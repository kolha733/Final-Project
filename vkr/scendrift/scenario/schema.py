"""Декларативная схема сценария дрейфа (pydantic).

Ответственный: Будаев К. В. (протокол оценки Ω — Гарифзянов Т. Р.,
см. :mod:`scendrift.evaluation.protocol`).

Формальная модель: сценарий S = ⟨B, E, Ω, s⟩, где

* B = (f, n, d, π, η, c) — базовая конфигурация потока
  (:class:`StreamSpec`): семейство генератора, длина, размерность, доля
  положительного (миноритарного) класса, шум меток, структурный seed c;
* E = (e₁, …, e_K) — упорядоченные события дрейфа (:class:`DriftEvent`),
  e_k = (τ_k, ℓ_k, φ_k, κ_k, m_k, α_k, ρ_k);
* Ω — протокол оценки (:class:`~scendrift.evaluation.protocol.EvaluationSpec`);
* s — seed реализации.

Здесь проверяются только *локальные* ограничения: домены полей и
согласованность полей одного события. Ограничения между полями разных
частей сценария и ограничения конкретного семейства генераторов проверяет
:mod:`scendrift.scenario.validation`. Благодаря такому разделению модули
формирования сценариев (этап 3) могут сначала построить недопустимого
кандидата, а затем отклонить или исправить его.

Кроме канонической формы (явный список событий) поддерживается компактная
форма :class:`DriftPlan`: расписание, параметры по умолчанию и
переопределения. Функция :func:`expand_plan` разворачивает её в явный список.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scendrift.evaluation.protocol import EvaluationSpec
from scendrift.scenario.enums import DriftForm, DriftKind, TransitionShape

__all__ = [
    "SCHEMA_VERSION",
    "MIN_CLASS_SHARE",
    "StreamSpec",
    "DriftEvent",
    "ScenarioSpec",
    "SchedulePlan",
    "EventTemplate",
    "EventOverride",
    "DriftPlan",
    "expand_plan",
    "layout_uniform",
]

#: Версия схемы сценария. Меняется при несовместимом изменении полей.
SCHEMA_VERSION = "1.0"

#: Минимально допустимая доля любого класса (для π и для prior-дрейфа).
MIN_CLASS_SHARE = 0.01

#: Метка подпотока генератора случайных чисел для расписания (см. expand_plan).
_SCHEDULE_STREAM_TAG = 1_000_003


class _Frozen(BaseModel):
    """Базовый класс: неизменяемые модели без лишних полей."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _plain(value: Any, path: str = "") -> Any:
    """Приводит произвольные параметры к JSON-совместимым значениям Python.

    numpy-скаляры и массивы превращаются в числа и списки, кортежи — в
    списки, −0,0 — в 0,0. Нечисловые (NaN, ±inf) значения и нестроковые ключи
    отклоняются: иначе разные сценарии получили бы одинаковый идентификатор.
    """
    where = path or "параметрах"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, bool) or value is None or isinstance(value, str | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"недопустимое нечисловое значение {value!r} в {where}")
        return value + 0.0
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"ключ {key!r} в {where} должен быть строкой")
            out[key] = _plain(item, f"{path}.{key}" if path else key)
        return out
    if isinstance(value, list | tuple):
        return [_plain(item, f"{path}[{i}]") for i, item in enumerate(value)]
    raise ValueError(f"неподдерживаемый тип {type(value).__name__} в {where}")


class StreamSpec(_Frozen):
    """Базовая конфигурация потока B = (f, n, d, π, η, c)."""

    family: str = Field(
        "hyperplane_gauss", min_length=1, description="f — семейство генератора потока."
    )
    n_samples: int = Field(20_000, ge=1_000, le=10_000_000, description="n — длина потока.")
    n_features: int = Field(10, ge=2, le=500, description="d — число признаков.")
    minority_share: float = Field(
        0.5,
        ge=MIN_CLASS_SHARE,
        le=0.5,
        description="π — исходная доля положительного (миноритарного) класса P(y=1).",
    )
    label_noise: float = Field(
        0.0, ge=0.0, lt=0.5, description="η — вероятность инверсии метки (шум меток)."
    )
    concept_seed: int = Field(
        0,
        ge=0,
        description=(
            "c — структурный seed: задаёт геометрию концептов (затронутые признаки, "
            "направления поворота и сдвига) и случайное расписание. От seed "
            "реализации не зависит, поэтому повторы сценария имеют одинаковые концепты."
        ),
    )
    family_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Параметры, специфичные для семейства (допустимые ключи задаёт семейство).",
    )

    @field_validator("minority_share", "label_noise")
    @classmethod
    def _floats(cls, value: float) -> float:
        """Заменяет −0,0 на 0,0, чтобы равные значения давали равный хэш."""
        return value + 0.0

    @field_validator("family_params", mode="before")
    @classmethod
    def _params(cls, value: Any) -> Any:
        return _plain(value, "family_params") if isinstance(value, Mapping) else value


class DriftEvent(_Frozen):
    """Событие дрейфа e = (τ, ℓ, φ, κ, m, α, ρ).

    Интервал перехода — полуинтервал [onset, end), где
    onset = τ − ⌊ℓ/2⌋ и end = onset + ℓ. При внезапном дрейфе onset = end = τ,
    и объект с индексом τ — первый объект нового концепта.

    Если задано ``returns_to`` (ρ), событие считается *повторяющимся*: поток
    возвращается к концепту с номером ρ. Концепт 0 — исходный, концепт k
    возникает после k-го события. Вид и величина такого события не задаются,
    а вычисляются по разности концептов.
    """

    position: int = Field(ge=0, description="τ — центр интервала перехода (индекс объекта).")
    width: int = Field(0, ge=0, description="ℓ — ширина перехода (в объектах).")
    form: DriftForm = Field(DriftForm.SUDDEN, description="φ — форма перехода.")
    kind: DriftKind | None = Field(None, description="κ — вид дрейфа (None для повторяющегося).")
    magnitude: float | None = Field(
        None, gt=0.0, le=1.0, description="m — величина дрейфа на единой шкале (0, 1]."
    )
    affected_share: float = Field(
        1.0, gt=0.0, le=1.0, description="α — доля признаков, затронутых дрейфом."
    )
    returns_to: int | None = Field(
        None, ge=0, description="ρ — номер концепта, к которому возвращается поток."
    )
    shape: TransitionShape = Field(
        TransitionShape.LINEAR, description="Профиль функции перехода p(t)."
    )

    @model_validator(mode="before")
    @classmethod
    def _default_kind(cls, data: Any) -> Any:
        """Подставляет вид ``real`` для обычного (неповторяющегося) события."""
        if (
            isinstance(data, Mapping)
            and data.get("returns_to") is None
            and data.get("kind") is None
        ):
            data = {**dict(data), "kind": DriftKind.REAL}
        return data

    @field_validator("magnitude", "affected_share")
    @classmethod
    def _floats(cls, value: float | None) -> float | None:
        """Заменяет −0,0 на 0,0, чтобы равные значения давали равный хэш."""
        return None if value is None else value + 0.0

    @model_validator(mode="after")
    def _check_local(self) -> DriftEvent:
        """Локальные ограничения события (C1–C3 каталога ограничений)."""
        if self.form is DriftForm.SUDDEN and self.width != 0:
            raise ValueError("C1: у внезапного дрейфа ширина перехода должна быть 0")
        if self.form is not DriftForm.SUDDEN and self.width < 2:
            raise ValueError("C1: у постепенного/инкрементального дрейфа ширина ≥ 2")
        if self.position - self.width // 2 < 0:
            raise ValueError("C1: интервал перехода начинается до начала потока")
        if self.returns_to is None:
            if self.magnitude is None:
                raise ValueError("C2: для обычного события обязательна величина magnitude")
            if self.kind is None:
                raise ValueError("C2: для обычного события обязателен вид kind")
        else:
            if self.kind is not None or self.magnitude is not None:
                raise ValueError(
                    "C3: у повторяющегося события (returns_to) вид и величина "
                    "не задаются — они вычисляются по разности концептов"
                )
        return self

    @property
    def onset(self) -> int:
        """Начало интервала перехода."""
        return self.position - self.width // 2

    @property
    def end(self) -> int:
        """Конец (исключительно) интервала перехода."""
        return self.onset + self.width

    @property
    def is_recurring(self) -> bool:
        """Признак повторяющегося события."""
        return self.returns_to is not None


class ScenarioSpec(_Frozen):
    """Сценарий дрейфа S = ⟨B, E, Ω, s⟩ в канонической (явной) форме.

    Поля ``name``, ``description``, ``tags`` и ``meta`` описательные: на
    порождаемые данные они не влияют и в идентификатор сценария не входят
    (см. :func:`scendrift.scenario.io.scenario_id`).
    """

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    name: str = Field("unnamed", min_length=1, max_length=200)
    description: str = ""
    tags: tuple[str, ...] = ()
    stream: StreamSpec = Field(default_factory=StreamSpec)
    events: tuple[DriftEvent, ...] = ()
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    seed: int = Field(0, ge=0, description="s — seed реализации (выборка X, y, шум).")
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Происхождение сценария (в хэш не входит)."
    )

    @property
    def n_events(self) -> int:
        """Число событий дрейфа K."""
        return len(self.events)

    def with_seed(self, seed: int) -> ScenarioSpec:
        """Копия сценария с другим seed реализации (для повторов), с проверкой."""
        return ScenarioSpec.model_validate({**self.model_dump(), "seed": int(seed)})

    @field_validator("meta", mode="before")
    @classmethod
    def _meta(cls, value: Any) -> Any:
        return _plain(value, "meta") if isinstance(value, Mapping) else value


# ---------------------------------------------------------------------------
# Компактная форма: расписание + значения по умолчанию + переопределения
# ---------------------------------------------------------------------------


class SchedulePlan(_Frozen):
    """Расписание дрейфов.

    * ``uniform`` — поток после разогрева делится на K равных блоков, и
      «след» события [onset, end + Δ) ставится в центр своего блока;
    * ``random`` — следы событий случайно размещаются без перекрытий
      (используется структурный seed c);
    * ``explicit`` — центры τ_k заданы явно.
    """

    mode: Literal["uniform", "random", "explicit"] = "uniform"
    count: int = Field(1, ge=0, le=100, description="K — число событий дрейфа.")
    positions: tuple[int, ...] | None = Field(
        None, description="Явные центры τ_k (только для mode=explicit)."
    )
    min_gap: int = Field(
        0, ge=0, description="Дополнительный зазор между следами событий (mode=random)."
    )

    @model_validator(mode="after")
    def _check(self) -> SchedulePlan:
        if self.mode == "explicit":
            if self.positions is None or len(self.positions) != self.count:
                raise ValueError("для mode=explicit нужно ровно count позиций")
        elif self.positions is not None:
            raise ValueError("positions допустимы только при mode=explicit")
        return self


class EventTemplate(_Frozen):
    """Параметры события по умолчанию: всё, кроме позиции."""

    width: int | None = Field(None, ge=0)
    form: DriftForm | None = None
    kind: DriftKind | None = None
    magnitude: float | None = Field(None, gt=0.0, le=1.0)
    affected_share: float | None = Field(None, gt=0.0, le=1.0)
    shape: TransitionShape | None = None


class EventOverride(EventTemplate):
    """Переопределение параметров события с номером ``index`` (с нуля)."""

    index: int = Field(ge=0)
    returns_to: int | None = Field(None, ge=0)
    position: int | None = Field(None, ge=0)


class DriftPlan(_Frozen):
    """Компактное описание событий дрейфа: расписание и параметры."""

    schedule: SchedulePlan = Field(default_factory=SchedulePlan)
    defaults: EventTemplate = Field(default_factory=EventTemplate)
    overrides: tuple[EventOverride, ...] = ()


def _event_params(plan: DriftPlan, index: int) -> dict[str, Any]:
    """Собирает параметры события: значения по умолчанию плюс переопределения."""
    params: dict[str, Any] = {k: v for k, v in plan.defaults.model_dump().items() if v is not None}
    for ov in plan.overrides:
        if ov.index == index:
            if ov.returns_to is not None:
                if ov.kind is not None or ov.magnitude is not None:
                    raise ValueError(
                        f"C3: событие {index} с returns_to не может задавать kind или magnitude"
                    )
                # Повторяющееся событие: вид и величина вычисляются, а не наследуются.
                params.pop("kind", None)
                params.pop("magnitude", None)
            params.update(
                {k: v for k, v in ov.model_dump(exclude={"index"}).items() if v is not None}
            )
    params.setdefault("form", DriftForm.SUDDEN)
    if params["form"] == DriftForm.SUDDEN:
        params["width"] = 0
    params.setdefault("width", 0)
    return params


def layout_uniform(widths: list[int], start: int, stop: int, delta: int) -> list[int]:
    """Центры событий при равномерном расписании.

    След события — полуинтервал [onset, end + Δ) длины ℓ_k + Δ. Свободное
    место F = (stop − start) − Σ(ℓ_k + Δ) делится поровну между K событиями,
    и каждый след ставится в центр своего блока длины (ℓ_k + Δ) + F/K. Если
    следы помещаются в поток (F ≥ 0), ограничения C4–C6 выполняются при
    любых ширинах переходов.

    Args:
        widths: ширины переходов ℓ_k.
        start: начало допустимой области (обычно W).
        stop: конец потока n.
        delta: окно допуска Δ.

    Returns:
        Список центров τ_k.
    """
    k_events = len(widths)
    if k_events == 0:
        return []
    footprints = [w + delta for w in widths]
    share = max(0.0, (stop - start) - sum(footprints)) / k_events
    centers, cursor = [], float(start)
    for w, fp in zip(widths, footprints, strict=True):
        onset = int(math.floor(cursor + share / 2.0))
        centers.append(onset + w // 2)
        cursor += fp + share
    return centers


def expand_plan(
    plan: DriftPlan,
    stream: StreamSpec,
    evaluation: EvaluationSpec,
) -> tuple[DriftEvent, ...]:
    """Разворачивает компактный план в явный список событий.

    Позиции рассчитываются так, чтобы при достаточной длине потока
    выполнялись ограничения C4–C6 (разогрев, хвост Δ после последнего
    перехода, непересечение следов [onset, end + Δ)). Если поток слишком
    короток, события всё равно строятся, а нарушения выявит валидатор
    (:func:`scendrift.scenario.validation.check`).

    Args:
        plan: компактное описание событий.
        stream: базовая конфигурация потока (нужны n и структурный seed).
        evaluation: протокол оценки (нужны W и Δ).

    Returns:
        Кортеж событий в порядке индексов. При расписаниях ``uniform`` и
        ``random`` он упорядочен по позиции; явные позиции не сортируются
        (на них ссылаются индексы ``returns_to``), а порядок проверяет C6.

    Raises:
        ValueError: при ссылке на несуществующее событие или противоречивом
            переопределении (returns_to вместе с kind или magnitude).
    """
    k_events = plan.schedule.count
    if k_events == 0:
        return ()
    for ov in plan.overrides:
        if ov.index >= k_events:
            raise ValueError(f"переопределение для несуществующего события {ov.index}")
    params = [_event_params(plan, i) for i in range(k_events)]
    widths = [int(p["width"]) for p in params]
    delta = evaluation.acceptance_window
    start, stop = evaluation.warmup, stream.n_samples
    footprints = [w + delta for w in widths]

    mode = plan.schedule.mode
    if mode == "explicit":
        assert plan.schedule.positions is not None
        centers = [int(t) for t in plan.schedule.positions]
    elif mode == "uniform":
        centers = layout_uniform(widths, start, stop, delta)
    else:  # random
        gap = plan.schedule.min_gap
        free = (stop - start) - sum(footprints) - gap * (k_events - 1)
        rng = np.random.default_rng([stream.concept_seed, _SCHEDULE_STREAM_TAG])
        cuts = np.sort(rng.uniform(0.0, max(free, 0), size=k_events))
        centers, cursor, prev = [], float(start), 0.0
        for i, (w, fp) in enumerate(zip(widths, footprints, strict=True)):
            cursor += cuts[i] - prev
            prev = cuts[i]
            centers.append(int(round(cursor)) + w // 2)
            cursor += fp + gap

    for i, p in enumerate(params):
        if "position" in p:  # явная позиция из переопределения имеет приоритет
            centers[i] = int(p.pop("position"))
    events = [DriftEvent(position=c, **p) for c, p in zip(centers, params, strict=True)]
    return tuple(events)

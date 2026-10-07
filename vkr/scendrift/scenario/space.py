"""Пространство параметров сценариев: домены, параметры, условная активность.

Ответственный: Будаев К. В.

Пространство параметров Θ = Θ₁ × … × Θ_D задаёт параметризованное семейство
сценариев. Каждая точка θ ∈ Θ отображается в сценарий S(θ)
(отображение реализуется на этапе 3 в :mod:`scendrift.formation`).
Модуль отвечает за домены и их отображение из единичного куба [0, 1)ᴰ:
это общая основа для перебора по сетке, случайной выборки,
латинского гиперкуба (LHS) и последовательностей Соболя. Обратное
отображение ``to_unit`` нужно, чтобы оценивать равномерность уже
сформированного набора (после исправления сценариев).

Некоторые параметры условные. Например, ширина перехода ℓ имеет смысл
только при φ ∈ {gradual, incremental}. Неактивный параметр получает
значение по умолчанию, но его координата в плане сохраняется
(см. :meth:`ParameterSpace.from_unit`).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

__all__ = [
    "space_id",
    "FloatDomain",
    "IntDomain",
    "Categorical",
    "Domain",
    "ParameterDef",
    "ParameterSpace",
    "AtLeast",
    "DEFAULT_SPACE",
]


def _fmt(x: float) -> str:
    """Число в русской записи (десятичная запятая)."""
    return f"{x:g}".replace(".", ",")


def _is_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


@dataclass(frozen=True)
class AtLeast:
    """Условие активности «значение не меньше bound» (например, K ≥ 2)."""

    bound: float

    def __contains__(self, value: object) -> bool:
        return _is_real(value) and value >= self.bound  # type: ignore[operator]

    def __str__(self) -> str:
        return f"≥ {_fmt(self.bound)}"


@dataclass(frozen=True)
class FloatDomain:
    """Непрерывный отрезок [low, high], при ``log`` — логарифмическая шкала."""

    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError("нужно low < high")
        if self.log and self.low <= 0:
            raise ValueError("для логарифмической шкалы нужно low > 0")

    def from_unit(self, u: float) -> float:
        """Отображение u ∈ [0, 1) → [low, high)."""
        if self.log:
            a, b = math.log(self.low), math.log(self.high)
            return math.exp(a + u * (b - a))
        return self.low + u * (self.high - self.low)

    def to_unit(self, value: float) -> float:
        """Обратное отображение: значение → координата u ∈ [0, 1]."""
        if self.log:
            a, b = math.log(self.low), math.log(self.high)
            return (math.log(float(value)) - a) / (b - a)
        return (float(value) - self.low) / (self.high - self.low)

    def contains(self, value: Any) -> bool:
        return _is_real(value) and self.low <= float(value) <= self.high

    def levels(self, n: int) -> list[float]:
        """N равноотстоящих (на выбранной шкале) уровней, включая концы."""
        if n == 1:
            return [self.from_unit(0.5)]
        return [self.from_unit(i / (n - 1)) if i < n - 1 else self.high for i in range(n)]

    def describe(self) -> str:
        scale = ", лог." if self.log else ""
        return f"[{_fmt(self.low)}; {_fmt(self.high)}]{scale}"


@dataclass(frozen=True)
class IntDomain:
    """Целочисленный отрезок {low, …, high}, при ``log`` — логарифмическая шкала."""

    low: int
    high: int
    log: bool = False

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError("нужно low < high")
        if self.log and self.low <= 0:
            raise ValueError("для логарифмической шкалы нужно low > 0")

    def from_unit(self, u: float) -> int:
        """Отображение u ∈ [0, 1) → {low, …, high} (равномерно на выбранной шкале)."""
        if self.log:
            a, b = math.log(self.low), math.log(self.high + 1)
            value = math.floor(math.exp(a + u * (b - a)))
        else:
            value = math.floor(self.low + u * (self.high - self.low + 1))
        return int(min(self.high, max(self.low, value)))

    def to_unit(self, value: int) -> float:
        """Обратное отображение: середина ячейки значения в [0, 1)."""
        v = int(value)
        if self.log:
            a, b = math.log(self.low), math.log(self.high + 1)
            lo, hi = (math.log(v) - a) / (b - a), (math.log(v + 1) - a) / (b - a)
        else:
            width = self.high - self.low + 1
            lo, hi = (v - self.low) / width, (v - self.low + 1) / width
        return (lo + hi) / 2.0

    def contains(self, value: Any) -> bool:
        return _is_int(value) and self.low <= int(value) <= self.high

    def levels(self, n: int) -> list[int]:
        """N уровней, включая концы (дубликаты при округлении удаляются)."""
        if n == 1:
            return [self.from_unit(0.5)]
        if self.log:
            a, b = math.log(self.low), math.log(self.high)
            raw = [round(math.exp(a + i / (n - 1) * (b - a))) for i in range(n)]
        else:
            raw = [round(self.low + i / (n - 1) * (self.high - self.low)) for i in range(n)]
        return sorted(set(int(v) for v in raw))

    def describe(self) -> str:
        scale = ", лог." if self.log else ""
        return f"{{{self.low}, …, {self.high}}}{scale}"


@dataclass(frozen=True)
class Categorical:
    """Конечное множество значений."""

    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("пустое множество значений")

    def from_unit(self, u: float) -> Any:
        """Отображение u ∈ [0, 1) → значение (равные доли отрезка на значение)."""
        return self.values[min(int(u * len(self.values)), len(self.values) - 1)]

    def to_unit(self, value: Any) -> float:
        """Обратное отображение: середина доли отрезка, отведённой значению."""
        return (self.values.index(value) + 0.5) / len(self.values)

    def contains(self, value: Any) -> bool:
        return value in self.values

    def levels(self, n: int | None = None) -> list[Any]:
        """Все значения (аргумент n игнорируется)."""
        return list(self.values)

    def describe(self) -> str:
        return "{" + ", ".join(str(v) for v in self.values) + "}"


Domain = FloatDomain | IntDomain | Categorical


@dataclass(frozen=True)
class ParameterDef:
    """Параметр пространства сценариев.

    Attributes:
        name: имя параметра (ключ в точке пространства).
        symbol: обозначение в формальной модели.
        group: часть сценария: stream, schedule, event или evaluation.
        domain: домен значений.
        default: значение по умолчанию (используется и для неактивного параметра).
        unit: единицы измерения.
        description: смысл параметра.
        active_if: условия активности: пары (имя другого параметра, допустимые
            значения). Допустимые значения — кортеж или :class:`AtLeast`.
    """

    name: str
    symbol: str
    group: str
    domain: Domain
    default: Any
    unit: str
    description: str
    active_if: tuple[tuple[str, tuple[Any, ...] | AtLeast], ...] = ()

    def is_active(self, values: Mapping[str, Any]) -> bool:
        """Активен ли параметр при заданных значениях остальных параметров."""
        return all(values.get(dep) in allowed for dep, allowed in self.active_if)


class ParameterSpace:
    """Упорядоченный набор параметров с операциями фиксации и сужения."""

    def __init__(self, params: Iterable[ParameterDef]) -> None:
        self._params: dict[str, ParameterDef] = {}
        for p in params:
            if p.name in self._params:
                raise ValueError(f"повтор параметра {p.name!r}")
            self._params[p.name] = p
        for p in self._params.values():
            for dep, _ in p.active_if:
                if dep not in self._params:
                    raise ValueError(f"{p.name}: условие ссылается на неизвестный {dep!r}")
                if list(self._params).index(dep) > list(self._params).index(p.name):
                    raise ValueError(f"{p.name}: параметр условия {dep!r} должен идти раньше")

    def __getitem__(self, name: str) -> ParameterDef:
        return self._params[name]

    def __iter__(self) -> Iterator[ParameterDef]:
        return iter(self._params.values())

    def __len__(self) -> int:
        return len(self._params)

    def __contains__(self, name: object) -> bool:
        return name in self._params

    @property
    def names(self) -> list[str]:
        """Имена параметров в порядке объявления."""
        return list(self._params)

    def subset(self, names: Sequence[str]) -> ParameterSpace:
        """Подпространство из указанных параметров (с сохранением порядка)."""
        keep = set(names)
        return ParameterSpace(p for p in self if p.name in keep)

    def with_domain(self, name: str, domain: Domain) -> ParameterSpace:
        """Копия пространства с другим доменом параметра ``name``."""
        if name not in self._params:
            raise KeyError(name)
        return ParameterSpace(replace(p, domain=domain) if p.name == name else p for p in self)

    def fix(self, **values: Any) -> ParameterSpace:
        """Фиксирует параметры: домен становится одноэлементным множеством."""
        space = self
        for name, value in values.items():
            space = space.with_domain(name, Categorical((value,)))
            space = ParameterSpace(
                replace(p, default=value) if p.name == name else p for p in space
            )
        return space

    def defaults(self) -> dict[str, Any]:
        """Точка пространства со значениями по умолчанию."""
        return {p.name: p.default for p in self}

    def free_names(self) -> list[str]:
        """Параметры, которые реально варьируются (домен из более чем одного значения)."""
        return [
            p.name
            for p in self
            if not (isinstance(p.domain, Categorical) and len(p.domain.values) == 1)
        ]

    def from_unit(self, u: Sequence[float]) -> dict[str, Any]:
        """Отображение точки единичного куба в точку пространства.

        Координаты u соответствуют :meth:`free_names` (по порядку). Каждый
        неактивный параметр получает значение по умолчанию, хотя его
        координата всё равно расходуется: так сохраняются свойства плана
        (стратификация LHS, равномерность последовательности Соболя) по
        остальным координатам.
        """
        free = self.free_names()
        if len(u) != len(free):
            raise ValueError(f"нужно {len(free)} координат, получено {len(u)}")
        coord = dict(zip(free, u, strict=True))
        values: dict[str, Any] = {}
        for p in self:
            if p.name in coord and p.is_active(values):
                values[p.name] = p.domain.from_unit(float(coord[p.name]))
            elif isinstance(p.domain, Categorical) and len(p.domain.values) == 1:
                values[p.name] = p.domain.values[0]
            else:
                values[p.name] = p.default
        return values

    def to_unit(
        self, values: Mapping[str, Any], fallback: Sequence[float] | None = None
    ) -> list[float]:
        """Обратное отображение точки пространства в единичный куб (по :meth:`free_names`).

        Координата неактивного параметра не определяется значением. Она
        берётся из ``fallback`` (обычно исходная точка плана), а без него
        равна 0,5.
        """
        free = self.free_names()
        out = []
        for j, name in enumerate(free):
            p = self[name]
            if p.is_active(values) and name in values:
                out.append(float(p.domain.to_unit(values[name])))
            else:
                out.append(float(fallback[j]) if fallback is not None else 0.5)
        return out

    def out_of_domain(self, values: Mapping[str, Any]) -> list[str]:
        """Имена активных параметров, значения которых вне домена."""
        bad = []
        for p in self:
            if p.name in values and p.is_active(values) and not p.domain.contains(values[p.name]):
                bad.append(p.name)
        return bad

    def table(self) -> list[dict[str, str]]:
        """Описание пространства для таблиц ноутбука и пояснительной записки."""
        rows = []
        for p in self:
            cond = "; ".join(
                f"{dep} {allowed}"
                if isinstance(allowed, AtLeast)
                else f"{dep} ∈ {{{', '.join(map(str, allowed))}}}"
                for dep, allowed in p.active_if
            )
            rows.append(
                {
                    "параметр": p.name,
                    "обозначение": p.symbol,
                    "часть": p.group,
                    "домен": p.domain.describe(),
                    "по умолчанию": str(p.default).replace(".", ","),
                    "единицы": p.unit,
                    "активен, если": cond or "всегда",
                    "смысл": p.description,
                }
            )
        return rows


def space_id(space: ParameterSpace) -> str:
    """Идентификатор пространства ``spc-xxxxxxxxxxxx``: хэш доменов и условий активности.

    Попадает в манифест набора сценариев: по нему видно, из какого
    пространства параметров сформирован набор.
    """
    import hashlib
    import json

    rows = [
        {
            "name": p.name,
            "domain": type(p.domain).__name__,
            "spec": repr(p.domain),
            "default": repr(p.default),
            "active_if": repr(p.active_if),
        }
        for p in space
    ]
    text = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "spc-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


#: Пространство параметров по умолчанию для семейства ``hyperplane_gauss``.
DEFAULT_SPACE = ParameterSpace(
    [
        ParameterDef(
            "n_samples",
            "n",
            "stream",
            IntDomain(10_000, 50_000),
            20_000,
            "объектов",
            "длина потока",
        ),
        ParameterDef(
            "n_features",
            "d",
            "stream",
            IntDomain(2, 50, log=True),
            10,
            "признаков",
            "размерность пространства признаков",
        ),
        ParameterDef(
            "minority_share",
            "π",
            "stream",
            FloatDomain(0.05, 0.5),
            0.5,
            "доля",
            "исходная доля положительного класса P(y=1)",
        ),
        ParameterDef(
            "label_noise",
            "η",
            "stream",
            FloatDomain(0.0, 0.3),
            0.0,
            "вероятность",
            "вероятность инверсии метки",
        ),
        ParameterDef(
            "n_drifts", "K", "schedule", IntDomain(1, 5), 1, "событий", "число событий дрейфа"
        ),
        ParameterDef(
            "recurring",
            "alt",
            "schedule",
            Categorical((False, True)),
            False,
            "флаг",
            "чередование концептов A → B → A → …",
            active_if=(("n_drifts", AtLeast(2)),),
        ),
        ParameterDef(
            "form",
            "φ",
            "event",
            Categorical(("sudden", "gradual", "incremental")),
            "sudden",
            "—",
            "форма перехода",
        ),
        ParameterDef(
            "width",
            "ℓ",
            "event",
            IntDomain(100, 4_000, log=True),
            1_000,
            "объектов",
            "ширина перехода",
            active_if=(("form", ("gradual", "incremental")),),
        ),
        ParameterDef(
            "kind",
            "κ",
            "event",
            Categorical(("real", "virtual", "prior")),
            "real",
            "—",
            "вид дрейфа (какая компонента P(X, y) меняется)",
        ),
        ParameterDef(
            "magnitude",
            "m",
            "event",
            FloatDomain(0.02, 0.5),
            0.2,
            "единая шкала",
            "величина дрейфа",
        ),
        ParameterDef(
            "affected_share",
            "α",
            "event",
            FloatDomain(0.2, 1.0),
            1.0,
            "доля",
            "доля признаков, затронутых дрейфом",
            active_if=(("kind", ("real", "virtual")),),
        ),
        ParameterDef(
            "warmup", "W", "evaluation", IntDomain(0, 5_000), 1_000, "объектов", "длина разогрева"
        ),
        ParameterDef(
            "acceptance_window",
            "Δ",
            "evaluation",
            IntDomain(500, 5_000),
            2_000,
            "объектов",
            "окно допуска обнаружения",
        ),
    ]
).fix(warmup=1_000, acceptance_window=2_000)

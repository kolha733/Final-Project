"""Параметризация сценариев: точка пространства параметров → сценарий; учёт ограничений.

Ответственный: Будаев К. В.

Отображение ξ ↦ S(ξ) строится по шаблону T в компактной форме: значения
параметров ξ записываются в поля шаблона (:data:`PARAM_PATHS`), после чего
компактная форма разворачивается в каноническую. Флаг ``recurring``
превращает события 2, 3, … в возвраты A → B → A → B → …

Точка плана может оказаться недопустимой (нарушены C4–C14). Реализованы
три способа учёта ограничений (``policy``):

* ``reject`` — недопустимые точки отбрасываются. Распределение остальных
  точек не искажается, но набор получается меньше бюджета;
* ``repair`` — недопустимая точка проецируется на допустимую область
  (:func:`scendrift.scenario.validation.repair`): число точек сохраняется,
  но исправленные точки скапливаются на границе области (например, m = m_max);
* ``adapt`` — **условные домены**. Для зависимых параметров домен
  сужается с учётом уже выбранных значений, и координата плана
  отображается в суженный домен:

      ξ_j = D_j(ξ_{<j}).from_unit(u_j),  D_j(ξ_{<j}) = [lo_j, min(hi_j, b_j(ξ_{<j}))].

  Границы b_j: число событий K и ширина ℓ — из условия, что следы событий
  помещаются в поток (C4–C6); доля признаков α — снизу, по C9; величина
  m — сверху, по C10–C12. Границы для α и m семейство сообщает через
  предложения валидатора (``Violation.suggestion``), поэтому способ не
  зависит от семейства. Точки не скапливаются на границе, и почти все
  точки допустимы. Цена — равномерность по u переходит в равномерность
  внутри условных доменов, а не по всей допустимой области.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from pydantic import ValidationError

from scendrift.formation.templates import resolve
from scendrift.scenario import io
from scendrift.scenario.report import ScenarioValidationError
from scendrift.scenario.schema import ScenarioSpec
from scendrift.scenario.space import IntDomain, ParameterSpace
from scendrift.scenario.validation import check, repair

__all__ = [
    "PARAM_PATHS",
    "Policy",
    "POLICIES",
    "Candidate",
    "derive_seed",
    "point_to_dict",
    "point_to_spec",
    "spec_to_point",
    "realize",
    "summarize",
    "magnitude_on_boundary",
]

#: Куда записывается параметр пространства в компактной форме сценария.
#: Имя с точками (например, ``stream.family_params.variants``) — путь напрямую.
PARAM_PATHS: dict[str, str] = {
    "family": "stream.family",
    "n_samples": "stream.n_samples",
    "n_features": "stream.n_features",
    "minority_share": "stream.minority_share",
    "label_noise": "stream.label_noise",
    "n_drifts": "drifts.schedule.count",
    "schedule": "drifts.schedule.mode",
    "form": "drifts.defaults.form",
    "width": "drifts.defaults.width",
    "shape": "drifts.defaults.shape",
    "kind": "drifts.defaults.kind",
    "magnitude": "drifts.defaults.magnitude",
    "affected_share": "drifts.defaults.affected_share",
    "warmup": "evaluation.warmup",
    "acceptance_window": "evaluation.acceptance_window",
}

Policy = Literal["reject", "repair", "adapt"]

#: Способы учёта ограничений.
POLICIES: tuple[str, ...] = ("reject", "repair", "adapt")

_SEED_TAG = 7_000_003
_MAGNITUDE_CODES = {"C10", "C11", "C12"}


def derive_seed(seed: int, index: int) -> int:
    """Seed сценария с номером ``index`` в наборе с seed ``seed``."""
    return int(np.random.default_rng([int(seed), _SEED_TAG, int(index)]).integers(2**31 - 1))


def _set(data: dict[str, Any], path: str, value: Any) -> None:
    node = data
    *parents, leaf = path.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def point_to_dict(point: Mapping[str, Any], template: Mapping[str, Any] | None = None) -> dict:
    """Описание сценария в компактной форме: шаблон + значения параметров.

    Raises:
        ValueError: если шаблон задаёт события явно (``events`` или ``blocks``).
        KeyError: если параметр не отображается в поле сценария.
    """
    data = resolve(template or {})
    if "events" in data or "blocks" in data:
        raise ValueError("шаблон для параметризации задаётся в компактной форме (drifts)")
    data.setdefault("drifts", {})
    for name, value in point.items():
        if name == "recurring":
            continue
        path = name if "." in name else PARAM_PATHS.get(name)
        if path is None:
            raise KeyError(f"параметр {name!r} не отображается в поле сценария")
        _set(data, path, value)
    if point.get("recurring"):
        count = int(data["drifts"].get("schedule", {}).get("count", 1))
        overrides = list(data["drifts"].get("overrides", []))
        # Чередование A → B → A → B: событие k ≥ 1 возвращает к концепту (k + 1) mod 2.
        overrides += [{"index": k, "returns_to": (k + 1) % 2} for k in range(1, count)]
        data["drifts"]["overrides"] = overrides
    return data


def point_to_spec(
    point: Mapping[str, Any],
    template: Mapping[str, Any] | None = None,
    *,
    concept_seed: int | None = None,
    seed: int | None = None,
    name: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> ScenarioSpec:
    """Сценарий по точке пространства параметров (без проверки ограничений C4–C14)."""
    data = point_to_dict(point, template)
    if concept_seed is not None:
        data.setdefault("stream", {})["concept_seed"] = int(concept_seed)
    if seed is not None:
        data["seed"] = int(seed)
    if name is not None:
        data["name"] = name
    if meta is not None:
        data["meta"] = {**data.get("meta", {}), **dict(meta)}
    return io.from_dict(data)


def spec_to_point(spec: ScenarioSpec, space: ParameterSpace) -> dict[str, Any]:
    """Значения параметров пространства, фактически заданные сценарием.

    Параметры событий берутся у первого события (первого обычного — для
    вида, величины и доли признаков). Неактивные параметры получают
    значения по умолчанию.
    """
    regular = [e for e in spec.events if e.returns_to is None]
    first = spec.events[0] if spec.events else None
    reg = regular[0] if regular else None
    dumped = spec.model_dump(mode="json")
    values: dict[str, Any] = {}
    for p in space:
        name = p.name
        if name == "n_drifts":
            v: Any = spec.n_events
        elif name == "recurring":
            v = any(e.returns_to is not None for e in spec.events)
        elif name in ("form", "width", "shape") and first is not None:
            v = getattr(first, name)
            v = v.value if hasattr(v, "value") else v
        elif name in ("kind", "magnitude", "affected_share") and reg is not None:
            v = getattr(reg, name)
            v = v.value if hasattr(v, "value") else v
        else:
            path = name if "." in name else PARAM_PATHS.get(name, "")
            if not path.startswith(("stream.", "evaluation.")):
                v = p.default  # не восстанавливается по канонической форме
            else:
                node: Any = dumped
                for key in path.split("."):
                    node = node[key]
                v = node
        values[name] = v
    for p in space:
        if not p.is_active(values):
            values[p.name] = p.default
    return values


@dataclass(frozen=True)
class Candidate:
    """Точка плана и результат её превращения в сценарий.

    Attributes:
        index: номер точки в плане.
        unit: координаты точки в единичном кубе.
        point: исходная точка пространства параметров.
        spec: итоговый сценарий (None, если точка отклонена).
        status: valid — допустима без изменений; repaired — исправлена
            проекцией; adapted — изменена условными доменами; rejected — отклонена.
        codes: коды ошибок исходной точки (до учёта ограничений).
        final_codes: коды ошибок, из-за которых точка отклонена.
        actions: журнал изменений.
    """

    index: int
    unit: tuple[float, ...]
    point: dict[str, Any]
    spec: ScenarioSpec | None
    status: Literal["valid", "repaired", "adapted", "rejected"]
    codes: tuple[str, ...] = ()
    final_codes: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        """Точка дала сценарий в наборе."""
        return self.spec is not None


def _errors(spec: ScenarioSpec) -> tuple[str, ...]:
    return tuple(sorted({v.code for v in check(spec).errors}))


def _narrow(domain: Any, low: float | None, high: float | None) -> Any:
    """Суженный домен [max(lo, low), min(hi, high)]; None — пустое пересечение."""
    lo = domain.low if low is None else max(domain.low, low)
    hi = domain.high if high is None else min(domain.high, high)
    if isinstance(domain, IntDomain):
        lo, hi = math.ceil(lo), math.floor(hi)
    if hi < lo:
        return None
    if hi == lo:
        return ("point", lo)
    if (lo, hi) == (domain.low, domain.high):
        return domain
    return replace(domain, low=lo, high=hi)


def _map(domain: Any, u: float) -> Any:
    return domain[1] if isinstance(domain, tuple) else domain.from_unit(u)


def _probe(
    point: dict[str, Any], template: Mapping[str, Any] | None, cseed: int, name: str, codes: set
) -> list[float]:
    """Предложения валидатора для полей с кодами ``codes`` при данной точке."""
    spec = point_to_spec(point, template, concept_seed=cseed, seed=cseed)
    return [
        float(v.suggestion)
        for v in check(spec).errors
        if v.code in codes and v.suggestion is not None and v.path.endswith("." + name)
    ]


def _adapt(
    space: ParameterSpace,
    u: Sequence[float],
    point: dict[str, Any],
    template: Mapping[str, Any] | None,
    cseed: int,
) -> tuple[dict[str, Any] | None, list[str], tuple[str, ...]]:
    """Условные домены: возвращает точку (или None), журнал и причину отказа."""
    coord = {k: float(v) for k, v in zip(space.free_names(), u, strict=True)}
    pt, actions = dict(point), []
    base = point_to_spec(pt, template, concept_seed=cseed, seed=cseed)
    n, warmup = base.stream.n_samples, base.evaluation.warmup
    delta = base.evaluation.acceptance_window
    plan = resolve(template or {}).get("drifts", {}).get("schedule", {})
    gap = int(plan.get("min_gap", 0)) if plan.get("mode") == "random" else 0
    avail = n - warmup

    def refresh() -> None:
        for p in space:
            if not p.is_active(pt):
                pt[p.name] = p.default

    def set_value(name: str, domain: Any, why: str) -> bool:
        if domain is None:
            return False
        new = _map(domain, coord[name])
        if new != pt[name]:
            actions.append(f"{why}: {name} {pt[name]!r} → {new!r}")
            pt[name] = new
        return True

    sudden = pt.get("form", "sudden") == "sudden"
    width_free = "width" in coord and space["width"].is_active(pt)
    # 1. Число событий: K·(ℓ_min + Δ) + (K − 1)·gap ≤ n − W.
    if "n_drifts" in coord and space["n_drifts"].is_active(pt):
        l_min = 0 if sudden else (space["width"].domain.low if width_free else int(pt["width"]))
        k_max = (avail + gap) // max(1, l_min + delta + gap)
        if not set_value("n_drifts", _narrow(space["n_drifts"].domain, None, k_max), "C4–C6"):
            return None, actions, ("C5",)
        refresh()
    # 2. Ширина перехода: ℓ ≤ (n − W + gap)/K − Δ − gap.
    if width_free and space["width"].is_active(pt):
        k = int(pt.get("n_drifts", base.n_events) or 1)
        l_max = (avail + gap) // k - delta - gap
        if not set_value("width", _narrow(space["width"].domain, None, l_max), "C4–C6"):
            return None, actions, ("C5",)
    # 3. Доля затронутых признаков: снизу, по предложению C9 при α = lo.
    if "affected_share" in coord and space["affected_share"].is_active(pt):
        dom = space["affected_share"].domain
        probe = {**pt, "affected_share": dom.low}
        lows = _probe(probe, template, cseed, "affected_share", {"C9"})
        if not set_value("affected_share", _narrow(dom, max(lows, default=None), None), "C9"):
            return None, actions, ("C9",)
    # 4. Величина: сверху, по предложениям C10–C12 при m = hi (итерации до согласования).
    if "magnitude" in coord and space["magnitude"].is_active(pt):
        dom = space["magnitude"].domain
        bound = dom.high
        for _ in range(20):
            sugg = _probe(
                {**pt, "magnitude": bound}, template, cseed, "magnitude", _MAGNITUDE_CODES
            )
            if not sugg or min(sugg) >= bound:
                break
            bound = min(sugg) * (1.0 - 1e-9)
        if not set_value("magnitude", _narrow(dom, None, bound), "C10–C12"):
            return None, actions, ("C10–C12",)
    refresh()
    return pt, actions, ()


def realize(
    space: ParameterSpace,
    U: np.ndarray,
    *,
    template: Mapping[str, Any] | None = None,
    policy: Policy = "adapt",
    seed: int = 0,
    prefix: str = "scn",
) -> list[Candidate]:
    """Превращает точки плана в сценарии с учётом ограничений.

    Args:
        space: пространство параметров.
        U: точки единичного куба, N × D (D = число свободных параметров).
        template: шаблон сценария в компактной форме (может содержать ``extends``).
        policy: способ учёта ограничений: reject, repair или adapt.
        seed: seed набора; структурный seed и seed реализации сценария i
            выводятся из пары (seed, i).
        prefix: префикс имён сценариев.

    Returns:
        Кандидаты в порядке точек плана (включая отклонённые).
    """
    if policy not in POLICIES:
        raise ValueError(f"неизвестный способ учёта ограничений {policy!r}")
    out = []
    for i, u in enumerate(np.asarray(U, dtype=float)):
        cseed = derive_seed(seed, i)
        point = space.from_unit(u)
        meta = {"formation": {"index": i, "policy": policy}}
        name = f"{prefix}-{i:04d}"
        try:
            spec0 = point_to_spec(
                point, template, concept_seed=cseed, seed=cseed, name=name, meta=meta
            )
            codes = _errors(spec0)
        except (ValidationError, ValueError) as exc:
            out.append(
                Candidate(i, tuple(u), point, None, "rejected", ("C1",), ("C1",), (str(exc),))
            )
            continue
        if policy == "adapt":
            pt, actions, why = _adapt(space, u, point, template, cseed)
            if pt is None:
                out.append(
                    Candidate(i, tuple(u), point, None, "rejected", codes, why, tuple(actions))
                )
                continue
            spec = point_to_spec(pt, template, concept_seed=cseed, seed=cseed, name=name, meta=meta)
            final = _errors(spec)
            if final:
                out.append(
                    Candidate(i, tuple(u), point, None, "rejected", codes, final, tuple(actions))
                )
            else:
                status = "adapted" if actions else "valid"
                out.append(Candidate(i, tuple(u), point, spec, status, codes, (), tuple(actions)))
            continue
        if not codes:
            out.append(Candidate(i, tuple(u), point, spec0, "valid"))
        elif policy == "reject":
            out.append(Candidate(i, tuple(u), point, None, "rejected", codes, codes))
        else:
            try:
                fixed = repair(spec0)
            except ScenarioValidationError as exc:
                final = tuple(sorted(exc.report.codes() - {"W1", "W2", "W3", "W4"}))
                out.append(Candidate(i, tuple(u), point, None, "rejected", codes, final))
                continue
            out.append(
                Candidate(
                    i, tuple(u), point, fixed.spec, "repaired", codes, (), tuple(fixed.actions)
                )
            )
    return out


def summarize(candidates: Sequence[Candidate]) -> dict[str, Any]:
    """Сводка формирования: доли статусов и причины (коды) недопустимости."""
    n = len(candidates)
    status = Counter(c.status for c in candidates)
    initial = Counter(code for c in candidates for code in c.codes)
    rejected = Counter(code for c in candidates if not c.accepted for code in c.final_codes)
    return {
        "points": n,
        "accepted": sum(c.accepted for c in candidates),
        "accepted_share": sum(c.accepted for c in candidates) / n if n else math.nan,
        **{s: status.get(s, 0) for s in ("valid", "repaired", "adapted", "rejected")},
        "initial_codes": dict(sorted(initial.items())),
        "rejection_codes": dict(sorted(rejected.items())),
    }


def magnitude_on_boundary(spec: ScenarioSpec, rel: float = 1e-6) -> bool:
    """Стоит ли величина какого-либо события на границе достижимости.

    Событие на границе, если увеличение его величины в (1 + rel) раз
    нарушает C10–C12 у этого же события. Так обнаруживается «атом на
    границе», который создаёт проекция (repair).
    """
    events = list(spec.events)
    for k, event in enumerate(events):
        if event.magnitude is None:
            continue
        bumped = events.copy()
        bumped[k] = event.model_copy(update={"magnitude": min(1.0, event.magnitude * (1 + rel))})
        report = check(spec.model_copy(update={"events": tuple(bumped)}))
        path = f"events[{k}].magnitude"
        if any(v.code in _MAGNITUDE_CODES and v.path == path for v in report.errors):
            return True
    return False

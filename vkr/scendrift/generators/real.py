r"""Внедрение дрейфа в реальные наборы данных: семейства ``real:bananas`` и ``real:phishing``.

Ответственный: Будаев К. В.

**Базовое распределение.** Концепт — дискретное распределение на строках
набора: веса p_i и метки ℓ_i, i = 1…N. Поток порождается бутстрепом: объект
t — строка i с вероятностью p_i, его признаки — исходные x_i, метка — ℓ_i.
Исходный концепт: равные веса и исходные метки, перевзвешенные по классам
так, чтобы P(y = 1) = π₀.

**Виды дрейфа.** Каждый вид меняет ровно один множитель в своём
разложении P(X, y) и сохраняет другой точно:

* real — инверсия меток в области {x : uᵀz(x) > c}, где z — стандартизованные
  признаки, u — направление в подпространстве затронутых признаков A.
  Веса строк не меняются, поэтому P(X) сохраняется точно;
* virtual — экспоненциальный наклон весов p_i ∝ p_i·exp(λ·vᵀz_i). Метки
  строк не меняются, поэтому P(y|X) сохраняется точно (у дубликатов одного
  x множитель общий);
* prior — перевзвешивание классов p_i ∝ p_i·r(ℓ_i). Сохраняется P(X|y).

**Величина** — расстояние полной вариации между совместными
распределениями, вычисленное точно на эмпирической мере (с группировкой
строк-дубликатов). Оно калибруется:

* для real — выбором размера области (точный перебор);
* для virtual — бисекцией по λ;
* для prior — аналитически, TV = |Δπ|.

Для real заданная величина достигается с точностью до массы одной строки.
Фактическое значение записывается в ground truth.

Инкрементальный переход дискретизируется на ``STEPS`` промежуточных концептов.
Возврат к концепту поддерживает только формы sudden и gradual: метки нельзя
непрерывно интерполировать между произвольными состояниями.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import river.datasets

from scendrift.generators.base import FamilyGenerator, register_generator
from scendrift.generators.transitions import Timeline
from scendrift.scenario.enums import DriftForm, DriftKind
from scendrift.scenario.families import FamilyInfo, register_family
from scendrift.scenario.report import Violation
from scendrift.scenario.schema import MIN_CLASS_SHARE, DriftEvent, ScenarioSpec
from scendrift.scenario.semantics import (
    MAX_VIRTUAL_MAGNITUDE,
    ZERO_DRIFT,
    affected_count,
    prior_target,
)

__all__ = [
    "DATASETS",
    "STEPS",
    "PRIOR_SIDE_EFFECT",
    "RealData",
    "RowState",
    "RealEventGeometry",
    "RealChain",
    "load",
    "joint_tv",
    "build_real_chain",
    "RealGenerator",
]

#: Число ступеней дискретизации инкрементального перехода.
STEPS = 64

#: Порог побочного изменения P(y) для предупреждения W4.
PRIOR_SIDE_EFFECT = 0.02

DATASETS: dict[str, type] = {
    "real:bananas": river.datasets.Bananas,
    "real:phishing": river.datasets.Phishing,
}

_EVENT_TAG = 5_000_011


@dataclass(frozen=True, eq=False)
class RealData:
    """Реальный набор данных, подготовленный для внедрения дрейфа.

    Attributes:
        X: исходные признаки, N × d.
        y: исходные метки 0/1, N.
        Z: стандартизованные признаки (для направлений u и v).
        ux: номер уникального вектора признаков для каждой строки
            (дубликаты получают один номер).
        names: имена признаков.
    """

    X: np.ndarray
    y: np.ndarray
    Z: np.ndarray
    ux: np.ndarray
    names: tuple[str, ...]

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


@lru_cache(maxsize=8)
def load(family: str) -> RealData:
    """Загружает набор данных семейства (наборы входят в river, сеть не нужна)."""
    rows = list(itertools.islice(DATASETS[family](), 10_000_000))
    names = tuple(str(k) for k in rows[0][0])
    X = np.array([[float(v) for v in x.values()] for x, _ in rows])
    y = np.array([int(bool(label)) for _, label in rows], dtype=np.int8)
    std = X.std(axis=0)
    Z = (X - X.mean(axis=0)) / np.where(std > 0, std, 1.0)
    _, ux = np.unique(X, axis=0, return_inverse=True)
    return RealData(X=X, y=y, Z=Z, ux=ux.ravel().astype(np.int64), names=names)


@dataclass(frozen=True, eq=False)
class RowState:
    """Концепт: распределение на строках (веса p и метки lab)."""

    p: np.ndarray
    lab: np.ndarray
    concept_id: int

    @property
    def prior(self) -> float:
        """Доля положительного класса P(y = 1)."""
        return float(self.p[self.lab == 1].sum())


def _key_masses(state: RowState, ux: np.ndarray) -> np.ndarray:
    return np.bincount(2 * ux + state.lab, weights=state.p, minlength=2 * (int(ux.max()) + 1))


def joint_tv(a: RowState, b: RowState, ux: np.ndarray) -> float:
    """TV(P_a(X, y), P_b(X, y)) на эмпирической мере (дубликаты сгруппированы)."""
    return float(0.5 * np.abs(_key_masses(a, ux) - _key_masses(b, ux)).sum())


def marginal_tv(a: RowState, b: RowState, ux: np.ndarray) -> float:
    """TV(P_a(X), P_b(X)) на эмпирической мере."""
    m = int(ux.max()) + 1
    return float(
        0.5 * np.abs(np.bincount(ux, a.p, minlength=m) - np.bincount(ux, b.p, minlength=m)).sum()
    )


def _reweight_prior(state: RowState, target: float, concept_id: int) -> RowState:
    pi = state.prior
    factor = np.where(state.lab == 1, target / pi, (1.0 - target) / (1.0 - pi))
    p = state.p * factor
    return RowState(p / p.sum(), state.lab, concept_id)


def _tilt(p: np.ndarray, scores: np.ndarray, lam: float) -> np.ndarray:
    logw = np.log(np.maximum(p, 1e-300)) + lam * scores
    logw -= logw.max()
    w = np.exp(logw) * (p > 0)
    return w / w.sum()


@dataclass(frozen=True, eq=False)
class RealEventGeometry:
    """Параметры события на реальных данных (для генерации и инкрементального перехода).

    Attributes:
        kind: вид события (None — возврат).
        order: порядок строк по проекции на u, по убыванию (real).
        n_flip: число инвертированных строк (real).
        scores: проекции vᵀz_i (virtual).
        lam: параметр наклона λ (virtual).
        prior_from: доля класса до события (prior).
        prior_to: доля класса после события (prior).
        realized: фактическая величина по компонентам.
    """

    kind: DriftKind | None
    order: np.ndarray | None = None
    n_flip: int = 0
    scores: np.ndarray | None = None
    lam: float = 0.0
    prior_from: float = 0.0
    prior_to: float = 0.0
    realized: Mapping[str, float] | None = None


@dataclass
class RealChain:
    """Цепочка концептов на реальных данных и найденные нарушения."""

    states: list[RowState]
    geometry: list[RealEventGeometry]
    violations: list[Violation]


def _flip_prefix(state: RowState, order: np.ndarray, n_flip: int, concept_id: int) -> RowState:
    lab = state.lab.copy()
    lab[order[:n_flip]] ^= 1
    return RowState(state.p, lab, concept_id)


def _calibrate_flip(
    state: RowState, order: np.ndarray, ux: np.ndarray, m: float
) -> tuple[int, float, float]:
    """Точный перебор размера области: (L*, TV(L*), TV(N)).

    TV пересчитывается приращениями: инверсия строки меняет массы двух
    ключей (x, 0) и (x, 1) одного вектора признаков.
    """
    base = _key_masses(state, ux)
    cur = base.copy()
    diff = 0.0
    best_n, best_tv = 0, 0.0
    for i, r in enumerate(order, start=1):
        u, lab, w = int(ux[r]), int(state.lab[r]), float(state.p[r])
        k_from, k_to = 2 * u + lab, 2 * u + 1 - lab
        diff -= abs(base[k_from] - cur[k_from]) + abs(base[k_to] - cur[k_to])
        cur[k_from] -= w
        cur[k_to] += w
        diff += abs(base[k_from] - cur[k_from]) + abs(base[k_to] - cur[k_to])
        tv = 0.5 * diff
        if abs(tv - m) < abs(best_tv - m):
            best_n, best_tv = i, tv
    return best_n, best_tv, 0.5 * diff


def _direction(rng: np.random.Generator, d: int, k: int) -> np.ndarray:
    subset = rng.choice(d, size=k, replace=False)
    v = np.zeros(d)
    v[subset] = rng.standard_normal(k)
    return v / np.linalg.norm(v)


def _apply(
    idx: int, event: DriftEvent, state: RowState, spec: ScenarioSpec, data: RealData, new_id: int
) -> tuple[RowState, RealEventGeometry, list[Violation]]:
    path = f"events[{idx}]"
    kind = DriftKind(event.kind)
    m = float(event.magnitude)  # type: ignore[arg-type]
    rng = np.random.default_rng([spec.stream.concept_seed, _EVENT_TAG, idx])
    unchanged = RowState(state.p, state.lab, new_id)

    if kind is DriftKind.PRIOR:
        target = prior_target(state.prior, m)
        if target is None:
            best = max(1.0 - MIN_CLASS_SHARE - state.prior, state.prior - MIN_CLASS_SHARE)
            v = Violation(
                "C12",
                f"prior-дрейф {m:.3g} выводит P(y=1) за [0,01; 0,99]",
                f"{path}.magnitude",
                suggestion=max(0.0, best),
            )
            return unchanged, RealEventGeometry(kind), [v]
        new = _reweight_prior(state, target, new_id)
        real = {"real": 0.0, "virtual": 0.0, "prior": abs(target - state.prior)}
        return (
            new,
            RealEventGeometry(kind, prior_from=state.prior, prior_to=target, realized=real),
            [],
        )

    d = data.n_features
    k = affected_count(event.affected_share, d)
    if k < 1:
        v = Violation(
            "C9", "не затронут ни один признак", f"{path}.affected_share", suggestion=1.0 / d
        )
        return unchanged, RealEventGeometry(kind), [v]
    direction = _direction(rng, d, k)
    scores = data.Z @ direction

    if kind is DriftKind.REAL:
        order = np.argsort(-scores, kind="stable")
        n_flip, tv, tv_max = _calibrate_flip(state, order, data.ux, m)
        if m > tv_max + 1.0 / data.n_rows:
            v = Violation(
                "C10",
                f"величина {m:.4g} недостижима инверсией меток в полупространстве: "
                f"максимум {tv_max:.4g}",
                f"{path}.magnitude",
                suggestion=tv_max,
            )
            return unchanged, RealEventGeometry(kind), [v]
        new = _flip_prefix(state, order, n_flip, new_id)
        real = {"real": tv, "virtual": 0.0, "prior": 0.0}
        return new, RealEventGeometry(kind, order=order, n_flip=n_flip, realized=real), []

    # virtual: наклон весов, бисекция по λ
    if m > MAX_VIRTUAL_MAGNITUDE:
        v = Violation(
            "C11",
            f"величина {m:.3g} > {MAX_VIRTUAL_MAGNITUDE}",
            f"{path}.magnitude",
            suggestion=MAX_VIRTUAL_MAGNITUDE,
        )
        return unchanged, RealEventGeometry(kind), [v]

    def tv_of(lam: float) -> float:
        return float(0.5 * np.abs(_tilt(state.p, scores, lam) - state.p).sum())

    hi = 1.0
    while tv_of(hi) < m and hi < 1e4:
        hi *= 2.0
    if tv_of(hi) < m:
        best = tv_of(hi)
        v = Violation(
            "C11",
            f"наклоном достижимо не более {best:.4g}",
            f"{path}.magnitude",
            suggestion=best * 0.999,
        )
        return unchanged, RealEventGeometry(kind), [v]
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if tv_of(mid) < m else (lo, mid)
        if hi - lo < 1e-13 * max(1.0, hi):
            break
    lam = 0.5 * (lo + hi)
    new = RowState(_tilt(state.p, scores, lam), state.lab, new_id)
    real = {"real": 0.0, "virtual": joint_tv(state, new, data.ux), "prior": 0.0}
    return new, RealEventGeometry(kind, scores=scores, lam=lam, realized=real), []


def build_real_chain(spec: ScenarioSpec) -> RealChain:
    """Цепочка концептов сценария на реальных данных (без порождения потока)."""
    data = load(spec.stream.family)
    base = RowState(np.full(data.n_rows, 1.0 / data.n_rows), data.y.copy(), 0)
    state = _reweight_prior(base, spec.stream.minority_share, 0)
    states, geometry, violations = [state], [], []
    next_id = 1
    for idx, event in enumerate(spec.events):
        if event.returns_to is not None:
            rho = event.returns_to
            if rho > idx:
                violations.append(
                    Violation("C7", f"концепта {rho} ещё нет", f"events[{idx}].returns_to")
                )
                new, geom = RowState(state.p, state.lab, next_id), RealEventGeometry(None)
                next_id += 1
            else:
                ref = states[rho]
                new = RowState(ref.p, ref.lab, ref.concept_id)
                changed = state.p[state.lab != ref.lab].sum()
                real = {
                    "real": float(changed),
                    "virtual": marginal_tv(state, ref, data.ux),
                    "prior": abs(ref.prior - state.prior),
                }
                geom = RealEventGeometry(None, realized=real)
                if joint_tv(state, ref, data.ux) < ZERO_DRIFT:
                    violations.append(
                        Violation(
                            "C7",
                            f"возврат к концепту {rho} не меняет распределение",
                            f"events[{idx}].returns_to",
                        )
                    )
        else:
            new, geom, issues = _apply(idx, event, state, spec, data, next_id)
            next_id += 1
            violations.extend(issues)
            side = abs(new.prior - state.prior)
            if geom.kind in (DriftKind.REAL, DriftKind.VIRTUAL) and side > PRIOR_SIDE_EFFECT:
                violations.append(
                    Violation(
                        "W4",
                        f"событие меняет и P(y): {state.prior:.3f} → {new.prior:.3f}",
                        f"events[{idx}].kind",
                        level="warning",
                    )
                )
        states.append(new)
        geometry.append(geom)
        state = new
    return RealChain(states, geometry, violations)


def _check(spec: ScenarioSpec) -> list[Violation]:
    data = load(spec.stream.family)
    out: list[Violation] = []
    if spec.stream.n_features != data.n_features:
        out.append(
            Violation(
                "C8",
                f"размерность задаётся набором данных: d = {data.n_features}",
                "stream.n_features",
                suggestion=data.n_features,
            )
        )
        return out
    for k, event in enumerate(spec.events):
        if event.returns_to is not None and event.form is DriftForm.INCREMENTAL:
            out.append(
                Violation(
                    "C8",
                    "возврат на реальных данных: только sudden или gradual",
                    f"events[{k}].form",
                )
            )
    return out + build_real_chain(spec).violations


def _incremental_state(chain: RealChain, e: int, level: float, data: RealData) -> RowState:
    """Промежуточный концепт инкрементального перехода на доле пути level."""
    a, geom = chain.states[e], chain.geometry[e]
    if geom.kind is DriftKind.REAL:
        return _flip_prefix(a, geom.order, int(round(level * geom.n_flip)), -1)  # type: ignore[arg-type]
    if geom.kind is DriftKind.VIRTUAL:
        return RowState(_tilt(a.p, geom.scores, level * geom.lam), a.lab, -1)  # type: ignore[arg-type]
    target = (1.0 - level) * geom.prior_from + level * geom.prior_to
    return _reweight_prior(a, target, -1)


class RealGenerator(FamilyGenerator):
    """Генератор полусинтетических потоков на реальных данных."""

    def __init__(self, family: str) -> None:
        self.family = family

    def feature_names(self, spec: ScenarioSpec) -> tuple[str, ...]:
        return load(self.family).names

    def realized(self, spec: ScenarioSpec) -> list[Mapping[str, float]]:
        return [dict(g.realized or {}) for g in build_real_chain(spec).geometry]

    def _sample(
        self, spec: ScenarioSpec, tl: Timeline, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = load(self.family)
        chain = build_real_chain(spec)
        n = spec.stream.n_samples
        take_new = rng.random(n) < tl.p
        state = np.where(take_new, tl.dst, tl.src).astype(np.int32)
        group = state.astype(np.int64) * (STEPS + 1)  # ключ: состояние × ступень
        for e, event in enumerate(spec.events):
            if event.form is DriftForm.INCREMENTAL:
                inside = (tl.event == e) & (tl.p > 0.0) & (tl.p < 1.0)
                level = np.ceil(tl.p[inside] * STEPS).astype(np.int64)
                state[inside] = e + 1
                group[inside] = e * (STEPS + 1) + level + len(chain.states) * (STEPS + 1)
        rows = np.empty(n, dtype=np.int64)
        labels = np.empty(n, dtype=np.int8)
        for key in np.unique(group):
            idx = np.flatnonzero(group == key)
            if key < len(chain.states) * (STEPS + 1):
                st = chain.states[key // (STEPS + 1)]
            else:
                rel = key - len(chain.states) * (STEPS + 1)
                e, level = divmod(int(rel), STEPS + 1)
                st = _incremental_state(chain, e, level / STEPS, data)
            pick = rng.choice(data.n_rows, size=len(idx), p=st.p)
            rows[idx] = pick
            labels[idx] = st.lab[pick]
        return data.X[rows], labels, state


for _name in DATASETS:
    register_family(
        FamilyInfo(
            name=_name,
            description=f"Внедрение дрейфа в реальный набор {_name.split(':')[1]} (бутстреп строк)",
            kinds=frozenset(DriftKind),
            forms=frozenset(DriftForm),
            supports_recurring=True,
            calibrated=True,
            check=_check,
        )
    )
    register_generator(RealGenerator(_name))

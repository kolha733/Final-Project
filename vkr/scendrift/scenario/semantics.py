r"""Семантика величины дрейфа и цепочка концептов семейства ``hyperplane_gauss``.

Ответственный: Будаев К. В.

**Модель семейства.** Концепт k задаётся состоянием (w_k, μ_k, π_k):

* w_k ∈ ℝᵈ, ‖w_k‖ = 1 — нормаль разделяющей гиперплоскости;
* μ_k ∈ ℝᵈ — центр базового распределения признаков N(μ_k, I);
* π_k — текущая доля положительного класса P(y = 1).

Метка задаётся правилом g_k(x) = 1[w_kᵀ(x − μ_k) > z], где z = Φ⁻¹(1 − π₀),
а π₀ — исходная доля класса (``minority_share``). Объекты порождаются по
схеме label shift: сначала y ~ Bernoulli(π_k), затем x | y из N(μ_k, I),
усечённого на полупространство класса y. Плотность P_k(X) равна
N(x; μ_k, I), перевзвешенной множителями π_k/π₀ и (1 − π_k)/(1 − π₀) на
полупространствах классов.

Каждый вид дрейфа меняет ровно одну **группу параметров** концепта:

* **real** — поворот w внутри подпространства затронутых признаков A.
  Меняется P(y|X), P(y) сохраняется. P(X) сохраняется при π_k = π₀, а при
  π_k ≠ π₀ меняется через перевзвешивание классов (предупреждение W3);
* **virtual** — сдвиг μ на δv, где v ⊥ w и supp v ⊆ A. Меняется P(X),
  а P(y|X) и P(y) сохраняются;
* **prior** — изменение π_k. Меняется P(y), а P(X|y) и P(y|X) сохраняются
  (для чистых меток; шум меток η > 0 делает наблюдаемое P(X|ỹ) зависящим от π).

**Единая шкала величины: расстояние полной вариации (TV).** Величина
обычного события — m = TV(P_k(X, y), P_{k+1}(X, y)) для чистых меток. Для
каждого вида это расстояние имеет точную форму:

* real: m = P_{x∼P_k}(g_k(x) ≠ g_{k+1}(x)), то есть доля смены метки
  (severity по Minku et al., 2010). Для угла θ между нормалями

  .. math:: m(θ) = (π₀ − Φ₂(−z, −z; \cos θ))\,(π_k/π₀ + (1 − π_k)/(1 − π₀)),

  что при π₀ = π_k = 0,5 даёт тождество m = θ/π. При повороте внутри A
  наибольший угол равен θ_max = arccos(1 − 2q), где q = ‖w_A‖², откуда
  m_max = m(θ_max) (ограничение C10);
* virtual: m = 2Φ(δ/2) − 1 (TV сдвинутых гауссиан), откуда
  δ = 2Φ⁻¹((1 + m)/2). Равенство точное и при перевзвешивании классов:
  плотность раскладывается на компоненту вдоль w, от которой зависит
  класс, и независимую гауссову компоненту в w⊥, а сдвиг затрагивает
  только вторую;
* prior: m = |π_{k+1} − π_k|.

Вся случайность геометрии (выбор A, направления поворота и сдвига)
определяется структурным seed c и номером события. Поэтому цепочка
концептов не зависит от seed реализации, и проверку допустимости
(:func:`build_chain`) можно провести без порождения данных.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri, owens_t

from scendrift.scenario.enums import DriftKind
from scendrift.scenario.report import Violation
from scendrift.scenario.schema import MIN_CLASS_SHARE, DriftEvent, ScenarioSpec

__all__ = [
    "FAMILY",
    "MAX_VIRTUAL_MAGNITUDE",
    "ZERO_DRIFT",
    "bvn_cdf",
    "real_severity_from_angle",
    "max_real_severity",
    "angle_for_severity",
    "rotation_for_angle",
    "angle_for_rotation",
    "tv_shift",
    "tv_gauss",
    "prior_target",
    "affected_count",
    "ConceptState",
    "initial_state",
    "drift_components",
    "EventGeometry",
    "ConceptChain",
    "build_chain",
]

FAMILY = "hyperplane_gauss"

#: Верхняя граница величины виртуального дрейфа (TV = 1 недостижимо при конечном сдвиге).
MAX_VIRTUAL_MAGNITUDE = 0.99

#: Порог, ниже которого возврат к концепту считается не меняющим распределение (C7).
ZERO_DRIFT = 1e-9

#: Метка подпотока ГСЧ для геометрии события (см. :func:`_event_rng`).
_EVENT_STREAM_TAG = 2_000_003

_RHO_EPS = 1e-15
_TOL = 1e-9


# ---------------------------------------------------------------------------
# Двумерное нормальное распределение
# ---------------------------------------------------------------------------


def bvn_cdf(h: float, k: float, rho: float) -> float:
    """Φ₂(h, k; ρ) = P(U ≤ h, V ≤ k) для стандартной двумерной нормали.

    Используется представление через T-функцию Оуэна (Owen, 1956), точное
    до машинной точности, с явной обработкой вырожденных случаев.
    """
    h, k, rho = float(h), float(k), float(rho)
    if rho >= 1.0 - _RHO_EPS:
        return float(ndtr(min(h, k)))
    if rho <= -1.0 + _RHO_EPS:
        return float(max(0.0, ndtr(h) - ndtr(-k)))
    s = math.sqrt(1.0 - rho * rho)
    if h == 0.0 and k == 0.0:
        return 0.25 + math.asin(rho) / (2.0 * math.pi)
    if h == 0.0:
        return float(0.5 * ndtr(k) - owens_t(k, -rho / s))
    if k == 0.0:
        return float(0.5 * ndtr(h) - owens_t(h, -rho / s))
    a_h = (k - rho * h) / (h * s)
    a_k = (h - rho * k) / (k * s)
    beta = 0.0 if (h > 0.0) == (k > 0.0) else 0.5  # по знакам: h·k может обнулиться
    value = 0.5 * ndtr(h) + 0.5 * ndtr(k) - owens_t(h, a_h) - owens_t(k, a_k) - beta
    return float(min(1.0, max(0.0, value)))


# ---------------------------------------------------------------------------
# Величина реального дрейфа
# ---------------------------------------------------------------------------


def _class_weight(pi0: float, prior: float) -> float:
    """Множитель π/π₀ + (1 − π)/(1 − π₀) из перевзвешивания классов."""
    return prior / pi0 + (1.0 - prior) / (1.0 - pi0)


def real_severity_from_angle(theta: float, pi0: float, prior: float) -> float:
    """Величина реального дрейфа m(θ) при повороте нормали на угол θ.

    Args:
        theta: угол между старой и новой нормалью, θ ∈ [0, π].
        pi0: исходная доля положительного класса π₀ (задаёт порог z).
        prior: текущая доля положительного класса π_k.
    """
    z = float(ndtri(1.0 - pi0))
    joint = bvn_cdf(-z, -z, math.cos(theta))
    return float((pi0 - joint) * _class_weight(pi0, prior))


def max_real_severity(q: float, pi0: float, prior: float) -> float:
    """Наибольшая величина, достижимая поворотом внутри A при q = ‖w_A‖²."""
    if q <= 0.0:
        return 0.0
    theta_max = math.acos(min(1.0, max(-1.0, 1.0 - 2.0 * q)))
    return real_severity_from_angle(theta_max, pi0, prior)


def angle_for_severity(m: float, q: float, pi0: float, prior: float) -> float:
    """Угол θ, при котором величина реального дрейфа равна m.

    m(θ) строго возрастает на [0, π], поскольку Φ₂(−z, −z; ρ) возрастает
    по ρ (тождество Плакетта). Поэтому корень единственный и ищется
    методом Брента.

    Raises:
        ValueError: если m > m_max(s).
    """
    theta_max = math.acos(min(1.0, max(-1.0, 1.0 - 2.0 * q)))
    m_max = real_severity_from_angle(theta_max, pi0, prior)
    if m > m_max * (1.0 + _TOL) + _TOL:
        raise ValueError(f"величина {m:.6g} недостижима: m_max = {m_max:.6g}")
    if m >= m_max:
        return theta_max
    return float(
        brentq(
            lambda t: real_severity_from_angle(t, pi0, prior) - m,
            0.0,
            theta_max,
            xtol=1e-14,
            rtol=1e-13,
        )
    )


def rotation_for_angle(theta: float, q: float) -> float:
    """Угол поворота ψ внутри A, дающий угол θ между нормалями.

    Из cos θ = 1 − q(1 − cos ψ) следует cos ψ = 1 − (1 − cos θ)/q.
    """
    cos_psi = 1.0 - (1.0 - math.cos(theta)) / q
    return math.acos(min(1.0, max(-1.0, cos_psi)))


def angle_for_rotation(psi: float, q: float) -> float:
    """Угол θ между нормалями при повороте на ψ внутри A (‖w_A‖² = q)."""
    return math.acos(min(1.0, max(-1.0, 1.0 - q * (1.0 - math.cos(psi)))))


# ---------------------------------------------------------------------------
# Виртуальный и prior-дрейф
# ---------------------------------------------------------------------------


def tv_shift(m: float) -> float:
    """Норма сдвига среднего δ, при которой TV(N(μ, I), N(μ + δv, I)) = m.

    Из TV = 2Φ(δ/2) − 1 следует δ = 2Φ⁻¹((1 + m)/2).
    """
    if not 0.0 <= m < 1.0:
        raise ValueError("величина виртуального дрейфа должна лежать в [0, 1)")
    return float(2.0 * ndtri((1.0 + m) / 2.0))


def tv_gauss(delta: float) -> float:
    """TV-расстояние между N(μ, I) и N(μ + δv, I), где ‖v‖ = 1: 2Φ(δ/2) − 1."""
    return float(math.erf(delta / (2.0 * math.sqrt(2.0))))


def prior_target(prior: float, m: float) -> float | None:
    """Новая доля P(y=1) при prior-дрейфе величины m.

    Правило: доля увеличивается на m, если остаётся ≤ 1 − MIN_CLASS_SHARE,
    иначе уменьшается на m. Возвращает None, если оба варианта недопустимы.
    """
    up, down = prior + m, prior - m
    if up <= 1.0 - MIN_CLASS_SHARE + _TOL:
        return up
    if down >= MIN_CLASS_SHARE - _TOL:
        return down
    return None


def affected_count(alpha: float, n_features: int) -> int:
    """Число затронутых признаков ⌊α·d⌉ (округление половины вверх)."""
    return int(min(n_features, math.floor(alpha * n_features + 0.5)))


# ---------------------------------------------------------------------------
# Состояния концептов
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ConceptState:
    """Состояние концепта (w, μ, π) семейства ``hyperplane_gauss``.

    Attributes:
        w: единичная нормаль гиперплоскости.
        mu: центр распределения признаков.
        prior: текущая доля положительного класса P(y = 1).
        pi0: исходная доля класса π₀, задающая порог z = Φ⁻¹(1 − π₀).
        concept_id: идентификатор концепта; при возврате к концепту ρ берётся его id.
    """

    w: np.ndarray
    mu: np.ndarray
    prior: float
    pi0: float
    concept_id: int

    @property
    def z(self) -> float:
        """Стандартизованный порог z = Φ⁻¹(1 − π₀)."""
        return float(ndtri(1.0 - self.pi0))

    def label(self, x: np.ndarray) -> np.ndarray:
        """Чистая (без шума) метка: 1[wᵀ(x − μ) > z]."""
        return ((np.asarray(x) - self.mu) @ self.w > self.z).astype(np.int8)


def initial_state(n_features: int, pi0: float) -> ConceptState:
    """Исходный концепт: равные веса признаков w = 1/√d · 1, μ = 0, π = π₀."""
    w = np.full(n_features, 1.0 / math.sqrt(n_features))
    return ConceptState(w=w, mu=np.zeros(n_features), prior=pi0, pi0=pi0, concept_id=0)


def _real_component(a: ConceptState, b: ConceptState) -> float:
    """P_{x∼P_a}(g_a(x) ≠ g_b(x)) для произвольных состояний a и b.

    Для поворота нормали (μ и π не меняются) эта величина равна
    TV(P_a(X, y), P_b(X, y)), в том числе при π ≠ π₀.
    """
    rho = float(np.clip(a.w @ b.w, -1.0, 1.0))
    z = a.z
    z_b = z + float(b.w @ (b.mu - a.mu))
    p_u = float(ndtr(-z))  # P(g_a = 1) по базовой гауссиане
    p_v = float(ndtr(-z_b))  # P(g_b = 1) по базовой гауссиане N(μ_a, I)
    p_uv = bvn_cdf(-z, -z_b, rho)  # P(g_a = 1, g_b = 1)
    flip_pos = (p_u - p_uv) / p_u  # P(g_b = 0 | g_a = 1)
    flip_neg = (p_v - p_uv) / (1.0 - p_u)  # P(g_b = 1 | g_a = 0)
    return float(max(0.0, a.prior * flip_pos + (1.0 - a.prior) * flip_neg))


def drift_components(a: ConceptState, b: ConceptState) -> dict[str, float]:
    """Разложение перехода a → b по группам параметров концепта (шкала TV).

    * ``real`` — доля смены метки P_{x∼P_a}(g_a ≠ g_b) (поворот нормали);
    * ``virtual`` — TV между базовыми распределениями N(μ_a, I) и N(μ_b, I);
    * ``prior`` — |π_b − π_a|.

    Для обычного события ненулевой будет только компонента его вида, и она
    равна TV(P_a(X, y), P_b(X, y)), то есть заданной величине m. Для
    повторяющегося события компоненты описывают, какие группы параметров
    изменились.
    """
    return {
        DriftKind.REAL.value: _real_component(a, b),
        DriftKind.VIRTUAL.value: tv_gauss(float(np.linalg.norm(b.mu - a.mu))),
        DriftKind.PRIOR.value: abs(b.prior - a.prior),
    }


# ---------------------------------------------------------------------------
# Цепочка концептов
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class EventGeometry:
    """Геометрия события: всё, что нужно генератору для перехода между концептами.

    Attributes:
        index: номер события (с нуля).
        kind: вид события (None — повторяющееся).
        target: номер состояния в цепочке, к которому ведёт событие (index + 1).
        returns_to: номер концепта возврата (для повторяющегося события).
        affected: индексы затронутых признаков A.
        direction: единичное направление поворота (real) или сдвига (virtual) в A, ⊥ w.
        rotation: угол поворота ψ внутри A (real).
        angle: угол θ между нормалями до и после (real).
        shift: норма сдвига среднего δ (virtual).
        realized: фактическая величина по компонентам (см. drift_components).
    """

    index: int
    kind: DriftKind | None
    target: int
    returns_to: int | None = None
    affected: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    direction: np.ndarray | None = None
    rotation: float = 0.0
    angle: float = 0.0
    shift: float = 0.0
    realized: dict[str, float] = field(default_factory=dict)


@dataclass
class ConceptChain:
    """Цепочка концептов c₀ → c₁ → … → c_K и нарушения, найденные при её построении."""

    states: list[ConceptState]
    geometry: list[EventGeometry]
    violations: list[Violation]

    @property
    def ok(self) -> bool:
        """Все события реализуемы."""
        return not any(v.level == "error" for v in self.violations)


def _event_rng(concept_seed: int, index: int) -> np.random.Generator:
    """Независимый подпоток ГСЧ для геометрии события ``index``."""
    return np.random.default_rng([concept_seed, _EVENT_STREAM_TAG, index])


def _orthogonal_direction(
    w: np.ndarray, subset: np.ndarray, rng: np.random.Generator
) -> np.ndarray | None:
    """Случайное единичное направление в span(A), ортогональное w.

    Возвращает None, если такого направления нет (|A| = 1 и w_A ≠ 0).
    """
    g = rng.standard_normal(len(subset))
    w_a = w[subset]
    norm_a = float(w_a @ w_a)
    if norm_a > 1e-24:
        g = g - (g @ w_a) / norm_a * w_a
    norm_g = float(np.linalg.norm(g))
    if norm_g < 1e-12:
        return None
    out = np.zeros_like(w)
    out[subset] = g / norm_g
    return out


def _apply_event(
    idx: int,
    event: DriftEvent,
    state: ConceptState,
    spec: ScenarioSpec,
    new_id: int,
) -> tuple[ConceptState, EventGeometry, list[Violation]]:
    """Применяет обычное (неповторяющееся) событие к состоянию."""
    path = f"events[{idx}]"
    d = spec.stream.n_features
    kind = DriftKind(event.kind)
    m = float(event.magnitude)  # type: ignore[arg-type]
    rng = _event_rng(spec.stream.concept_seed, idx)
    unchanged = ConceptState(state.w, state.mu, state.prior, state.pi0, new_id)

    if kind is DriftKind.PRIOR:
        target = prior_target(state.prior, m)
        if target is None:
            best = max(1.0 - MIN_CLASS_SHARE - state.prior, state.prior - MIN_CLASS_SHARE)
            v = Violation(
                "C12",
                f"prior-дрейф величины {m:.3g} выводит P(y=1) за [0,01; 0,99] "
                f"(текущая доля {state.prior:.3g})",
                f"{path}.magnitude",
                suggestion=max(0.0, best),
            )
            return unchanged, EventGeometry(idx, kind, idx + 1), [v]
        new = ConceptState(state.w, state.mu, target, state.pi0, new_id)
        geom = EventGeometry(idx, kind, idx + 1)
        return new, geom, []

    k = affected_count(event.affected_share, d)
    if k < 2:
        v = Violation(
            "C9",
            f"затронуто признаков ⌊α·d⌉ = {k} < 2: нельзя построить направление ⊥ w",
            f"{path}.affected_share",
            suggestion=2.0 / d,
        )
        return unchanged, EventGeometry(idx, kind, idx + 1), [v]
    subset = np.sort(rng.choice(d, size=k, replace=False))
    direction = _orthogonal_direction(state.w, subset, rng)
    if direction is None:  # pragma: no cover - при k ≥ 2 невозможно
        v = Violation("C9", "нет направления ⊥ w внутри A", f"{path}.affected_share")
        return unchanged, EventGeometry(idx, kind, idx + 1), [v]

    if kind is DriftKind.VIRTUAL:
        if m > MAX_VIRTUAL_MAGNITUDE:
            v = Violation(
                "C11",
                f"величина виртуального дрейфа {m:.3g} > {MAX_VIRTUAL_MAGNITUDE}",
                f"{path}.magnitude",
                suggestion=MAX_VIRTUAL_MAGNITUDE,
            )
            return unchanged, EventGeometry(idx, kind, idx + 1, affected=subset), [v]
        delta = tv_shift(m)
        new = ConceptState(state.w, state.mu + delta * direction, state.prior, state.pi0, new_id)
        geom = EventGeometry(idx, kind, idx + 1, affected=subset, direction=direction, shift=delta)
        return new, geom, []

    # Реальный дрейф: поворот w внутри A.
    q = float(state.w[subset] @ state.w[subset])
    m_max = max_real_severity(q, state.pi0, state.prior)
    if m > m_max * (1.0 + _TOL) + _TOL:
        v = Violation(
            "C10",
            f"величина реального дрейфа {m:.4g} недостижима поворотом в A: "
            f"m_max = {m_max:.4g} (q = ‖w_A‖² = {q:.3g})",
            f"{path}.magnitude",
            suggestion=m_max,
        )
        return unchanged, EventGeometry(idx, kind, idx + 1, affected=subset), [v]
    theta = angle_for_severity(m, q, state.pi0, state.prior)
    psi = rotation_for_angle(theta, q)
    w_new = state.w.copy()
    w_new[subset] = (
        math.cos(psi) * state.w[subset] + math.sin(psi) * math.sqrt(q) * (direction[subset])
    )
    w_new /= np.linalg.norm(w_new)
    new = ConceptState(w_new, state.mu, state.prior, state.pi0, new_id)
    geom = EventGeometry(
        idx, kind, idx + 1, affected=subset, direction=direction, rotation=psi, angle=theta
    )
    issues = []
    if abs(state.prior - state.pi0) > _TOL:
        issues.append(
            Violation(
                "W3",
                f"реальный дрейф при π_k = {state.prior:.3g} ≠ π₀ = {state.pi0:.3g}: "
                "меняется также P(X) (через перевзвешивание классов)",
                f"{path}.kind",
                level="warning",
            )
        )
    return new, geom, issues


def build_chain(spec: ScenarioSpec) -> ConceptChain:
    """Строит цепочку концептов сценария семейства ``hyperplane_gauss``.

    Данные при этом не порождаются: вычисляются только состояния концептов,
    геометрия событий и фактические величины. Если событие нереализуемо,
    в цепочку записывается неизменённый концепт и нарушение (C7, C9–C12).
    Реальный дрейф при π_k ≠ π₀ сопровождается предупреждением W3.

    Args:
        spec: сценарий семейства ``hyperplane_gauss``.

    Returns:
        Цепочка из K + 1 состояний с геометрией и списком нарушений.
    """
    state = initial_state(spec.stream.n_features, spec.stream.minority_share)
    states, geometry, violations = [state], [], []
    next_id = 1
    for idx, event in enumerate(spec.events):
        if event.returns_to is not None:
            rho = event.returns_to
            if rho > idx or states[rho].concept_id == state.concept_id:
                violations.append(
                    Violation(
                        "C7",
                        f"возврат к концепту {rho} невозможен: он ещё не существует "
                        "или совпадает с текущим",
                        f"events[{idx}].returns_to",
                    )
                )
                new = ConceptState(state.w, state.mu, state.prior, state.pi0, next_id)
                next_id += 1
                geom = EventGeometry(idx, None, idx + 1, returns_to=rho)
            else:
                ref = states[rho]
                new = ConceptState(ref.w, ref.mu, ref.prior, ref.pi0, ref.concept_id)
                geom = EventGeometry(idx, None, idx + 1, returns_to=rho)
                if max(drift_components(state, new).values()) < ZERO_DRIFT:
                    violations.append(
                        Violation(
                            "C7",
                            f"возврат к концепту {rho} не меняет распределение: "
                            "текущий концепт совпадает с ним по всем параметрам",
                            f"events[{idx}].returns_to",
                        )
                    )
        else:
            new, geom, issues = _apply_event(idx, event, state, spec, next_id)
            next_id += 1
            violations.extend(issues)
        geom = EventGeometry(
            geom.index,
            geom.kind,
            geom.target,
            geom.returns_to,
            geom.affected,
            geom.direction,
            geom.rotation,
            geom.angle,
            geom.shift,
            drift_components(state, new),
        )
        states.append(new)
        geometry.append(geom)
        state = new
    return ConceptChain(states, geometry, violations)

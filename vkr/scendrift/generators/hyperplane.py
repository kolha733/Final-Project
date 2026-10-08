"""Калиброванное семейство ``hyperplane_gauss``: порождение данных.

Ответственный: Будаев К. В.

Цепочка концептов и калибровка величины вычисляются заранее в
:func:`scendrift.scenario.semantics.build_chain`. Генератор только порождает
объекты из рассчитанных состояний (w, μ, π) по схеме label shift:

1. Метка выбирается как y ~ Bernoulli(π_t).
2. Проекция s = wᵀ(x − μ) берётся из N(0, 1), усечённого на полупространство
   класса: s > z для y = 1, s ≤ z для y = 0. Используется обратная функция
   распределения, без отбора.
3. Ортогональная к w часть берётся из N(0, I − wwᵀ).
4. x = μ + (g − (gᵀw)w) + s·w.

Формы перехода:

* sudden и gradual — объект порождается старым концептом с вероятностью
  1 − p(t) и новым с вероятностью p(t);
* incremental — параметры интерполируются для каждого объекта:
  - real: поворот на угол ψ·p(t) в той же плоскости;
  - virtual: сдвиг δ·p(t);
  - prior: π(t) = (1 − p)π_{k−1} + pπ_k;
  - возврат: сферическая интерполяция нормали, линейная — центра и доли.

Всё векторизовано по объектам.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.special import ndtr, ndtri

from scendrift.generators.base import FamilyGenerator, register_generator
from scendrift.generators.transitions import Timeline
from scendrift.scenario.enums import DriftForm, DriftKind
from scendrift.scenario.schema import ScenarioSpec
from scendrift.scenario.semantics import FAMILY, ConceptChain, build_chain

__all__ = ["HyperplaneGenerator", "sample_states"]


def _slerp(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Сферическая интерполяция единичных векторов a → b для массива долей p."""
    cos = float(np.clip(a @ b, -1.0, 1.0))
    omega = float(np.arccos(cos))
    if omega < 1e-12:
        return np.repeat(a[None, :], len(p), axis=0)
    p = p[:, None]
    return (np.sin((1.0 - p) * omega) * a + np.sin(p * omega) * b) / np.sin(omega)


def sample_states(
    W: np.ndarray,
    MU: np.ndarray,
    PI: np.ndarray,
    z: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Порождает по одному объекту для каждой строки параметров (w_t, μ_t, π_t).

    Args:
        W: единичные нормали, n × d.
        MU: центры, n × d.
        PI: доли положительного класса, n.
        z: стандартизованный порог Φ⁻¹(1 − π₀).
        rng: генератор случайных чисел.

    Returns:
        Признаки X (n × d) и чистые метки y (n).
    """
    n, d = W.shape
    y = rng.random(n) < PI
    u = 1.0 - rng.random(n)  # (0, 1]
    s = np.where(y, -ndtri(u * ndtr(-z)), ndtri(u * ndtr(z)))
    g = rng.standard_normal((n, d))
    g -= np.sum(g * W, axis=1, keepdims=True) * W
    X = MU + g + s[:, None] * W
    return X, y.astype(np.int8)


class HyperplaneGenerator(FamilyGenerator):
    """Генератор калиброванного семейства ``hyperplane_gauss``."""

    family = FAMILY

    def realized(self, spec: ScenarioSpec) -> list[Mapping[str, float]]:
        return [g.realized for g in build_chain(spec).geometry]

    def parameters(
        self, spec: ScenarioSpec, tl: Timeline, rng: np.random.Generator, chain: ConceptChain
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Параметры порождающего концепта для каждого объекта.

        Returns:
            Массивы W (n × d), MU (n × d), PI (n) и номер состояния (n).
        """
        states = chain.states
        w_s = np.stack([st.w for st in states])
        mu_s = np.stack([st.mu for st in states])
        pi_s = np.array([st.prior for st in states])
        # постепенный и внезапный переход: выбор старого или нового концепта
        take_new = rng.random(len(tl.p)) < tl.p
        state = np.where(take_new, tl.dst, tl.src).astype(np.int32)
        W, MU, PI = w_s[state].copy(), mu_s[state].copy(), pi_s[state].copy()

        for e, event in enumerate(spec.events):
            if event.form is not DriftForm.INCREMENTAL:
                continue
            inside = (tl.event == e) & (tl.p > 0.0) & (tl.p < 1.0)
            if not inside.any():
                continue
            p = tl.p[inside]
            state[inside] = e + 1
            a, b, geom = states[e], states[e + 1], chain.geometry[e]
            if event.returns_to is not None:
                W[inside] = _slerp(a.w, b.w, p)
                MU[inside] = a.mu + p[:, None] * (b.mu - a.mu)
                PI[inside] = (1.0 - p) * a.prior + p * b.prior
            elif geom.kind is DriftKind.REAL:
                mask = np.zeros_like(a.w)
                mask[geom.affected] = 1.0
                w_a = a.w * mask
                q = float(w_a @ w_a)
                psi = geom.rotation * p[:, None]
                W[inside] = (
                    a.w + (np.cos(psi) - 1.0) * w_a + np.sin(psi) * np.sqrt(q) * geom.direction
                )
            elif geom.kind is DriftKind.VIRTUAL:
                MU[inside] = a.mu + (geom.shift * p)[:, None] * geom.direction
            else:
                PI[inside] = (1.0 - p) * a.prior + p * b.prior
        return W, MU, PI, state

    def _sample(
        self, spec: ScenarioSpec, tl: Timeline, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chain = build_chain(spec)
        W, MU, PI, state = self.parameters(spec, tl, rng, chain)
        X, y = sample_states(W, MU, PI, chain.states[0].z, rng)
        return X, y, state


register_generator(HyperplaneGenerator())

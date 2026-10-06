"""Тесты семантики величины дрейфа (формулы проверяются численно и методом Монте-Карло)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import multivariate_normal, truncnorm

from scendrift.scenario import io
from scendrift.scenario import semantics as S
from scendrift.scenario.schema import ScenarioSpec


def _sample(state: S.ConceptState, n: int, rng: np.random.Generator) -> np.ndarray:
    """Выборка X по схеме label shift (как будет в генераторе этапа 2)."""
    y = rng.random(n) < state.prior
    z = state.z
    s = np.where(
        y,
        truncnorm.rvs(z, np.inf, size=n, random_state=rng),
        truncnorm.rvs(-np.inf, z, size=n, random_state=rng),
    )
    g = rng.standard_normal((n, state.w.size))
    g -= np.outer(g @ state.w, state.w)
    return state.mu + g + np.outer(s, state.w)


@pytest.mark.parametrize("h", [-1.3, -0.4, 0.0, 0.7])
@pytest.mark.parametrize("k", [-0.9, 0.0, 0.25, 1.6])
@pytest.mark.parametrize("rho", [-0.95, -0.3, 0.0, 0.6, 0.99])
def test_bvn_cdf_matches_scipy(h: float, k: float, rho: float) -> None:
    ref = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([h, k])
    assert S.bvn_cdf(h, k, rho) == pytest.approx(ref, abs=1e-6)


def test_bvn_cdf_degenerate_correlations() -> None:
    assert S.bvn_cdf(0.3, -0.2, 1.0) == pytest.approx(0.4207403, abs=1e-6)  # Φ(-0.2)
    assert S.bvn_cdf(-0.5, -0.5, -1.0) == 0.0
    assert S.bvn_cdf(0.5, 0.5, -1.0) == pytest.approx(0.3829249, abs=1e-6)


@pytest.mark.parametrize("theta", [0.0, 0.3, 1.0, math.pi / 2, 2.5, math.pi])
def test_balanced_severity_is_theta_over_pi(theta: float) -> None:
    assert S.real_severity_from_angle(theta, 0.5, 0.5) == pytest.approx(theta / math.pi, abs=1e-12)


def test_severity_monotone_in_angle() -> None:
    thetas = np.linspace(0.0, math.pi, 200)
    for pi0, prior in [(0.5, 0.5), (0.2, 0.2), (0.1, 0.4), (0.3, 0.7)]:
        values = [S.real_severity_from_angle(t, pi0, prior) for t in thetas]
        assert np.all(np.diff(values) > -1e-12)


def test_max_severity_closed_forms() -> None:
    assert S.max_real_severity(1.0, 0.5, 0.5) == pytest.approx(1.0)
    assert S.max_real_severity(0.3, 0.5, 0.5) == pytest.approx(math.acos(0.4) / math.pi)
    # при π₀ < 0,5 и полном развороте m_max = π + (1 − π)·π₀/(1 − π₀)
    assert S.max_real_severity(1.0, 0.2, 0.2) == pytest.approx(0.4)
    assert S.max_real_severity(0.0, 0.5, 0.5) == 0.0


def test_angle_for_severity_inverts() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        pi0 = rng.uniform(0.05, 0.5)
        prior = rng.uniform(0.05, 0.95)
        s = rng.uniform(0.05, 1.0)
        m = rng.uniform(0.0, 1.0) * S.max_real_severity(s, pi0, prior)
        theta = S.angle_for_severity(m, s, pi0, prior)
        assert S.real_severity_from_angle(theta, pi0, prior) == pytest.approx(m, abs=1e-10)
        phi = S.rotation_for_angle(theta, s)
        assert S.angle_for_rotation(phi, s) == pytest.approx(theta, abs=1e-9)


def test_angle_for_infeasible_severity_raises() -> None:
    with pytest.raises(ValueError):
        S.angle_for_severity(0.5, 0.1, 0.5, 0.5)


@pytest.mark.parametrize("m", [0.01, 0.2, 0.5, 0.9, 0.99])
def test_tv_inverse(m: float) -> None:
    assert S.tv_gauss(S.tv_shift(m)) == pytest.approx(m, abs=1e-12)


def test_tv_shift_known_value() -> None:
    # TV(N(0,1), N(δ,1)) = 2Φ(δ/2) − 1; при m = 0,5 сдвиг δ = 2Φ⁻¹(0,75) ≈ 1,34898
    assert S.tv_shift(0.5) == pytest.approx(1.3489795, abs=1e-6)


def test_bvn_cdf_tiny_same_sign_arguments() -> None:
    ref = 0.25 + math.asin(0.3) / (2 * math.pi)
    assert S.bvn_cdf(1e-200, 1e-200, 0.3) == pytest.approx(ref, abs=1e-12)


def test_prior_target_rule() -> None:
    assert S.prior_target(0.3, 0.2) == pytest.approx(0.5)
    assert S.prior_target(0.9, 0.3) == pytest.approx(0.6)
    assert S.prior_target(0.5, 0.6) is None


def test_affected_count_rounding() -> None:
    assert S.affected_count(0.25, 10) == 3  # 2,5 → 3 (половина вверх)
    assert S.affected_count(0.15, 10) == 2
    assert S.affected_count(1.0, 7) == 7


def test_chain_realizes_specified_magnitudes(demo_spec: ScenarioSpec) -> None:
    chain = S.build_chain(demo_spec)
    assert chain.ok
    g0, g1, g2 = chain.geometry
    assert g0.realized == pytest.approx({"real": 0.2, "virtual": 0.0, "prior": 0.0}, abs=1e-9)
    assert g1.realized == pytest.approx({"real": 0.0, "virtual": 0.4, "prior": 0.0}, abs=1e-9)
    # возврат к c₀ отменяет и поворот, и сдвиг
    assert g2.realized["virtual"] == pytest.approx(0.4, abs=1e-9)
    assert g2.realized["real"] > 0.0
    assert chain.states[3].concept_id == chain.states[0].concept_id
    assert np.allclose(chain.states[3].w, chain.states[0].w)


def test_chain_geometry_invariants(demo_spec: ScenarioSpec) -> None:
    chain = S.build_chain(demo_spec)
    for st in chain.states:
        assert np.linalg.norm(st.w) == pytest.approx(1.0)
    shift = chain.states[2].mu - chain.states[1].mu
    assert abs(shift @ chain.states[1].w) < 1e-12  # сдвиг ортогонален нормали
    affected = chain.geometry[1].affected
    untouched = np.setdiff1d(np.arange(10), affected)
    assert np.allclose(shift[untouched], 0.0)  # сдвигаются только признаки из A
    rot = chain.geometry[0]
    untouched = np.setdiff1d(np.arange(10), rot.affected)
    assert np.allclose(chain.states[1].w[untouched], chain.states[0].w[untouched])


def test_chain_depends_on_concept_seed_not_on_seed(demo_spec: ScenarioSpec) -> None:
    a = S.build_chain(demo_spec)
    b = S.build_chain(demo_spec.with_seed(999))
    for sa, sb in zip(a.states, b.states, strict=True):
        assert np.array_equal(sa.w, sb.w) and np.array_equal(sa.mu, sb.mu)
    other = demo_spec.model_copy(
        update={"stream": demo_spec.stream.model_copy(update={"concept_seed": 8})}
    )
    c = S.build_chain(other)
    assert not np.allclose(a.states[1].w, c.states[1].w)


def test_monte_carlo_real_drift_imbalanced() -> None:
    spec = io.from_dict(
        {
            "stream": {"n_samples": 10_000, "n_features": 8, "minority_share": 0.15},
            "events": [
                {"position": 5_000, "kind": "real", "magnitude": 0.12, "affected_share": 0.5}
            ],
        }
    )
    chain = S.build_chain(spec)
    a, b = chain.states
    rng = np.random.default_rng(3)
    x = _sample(a, 300_000, rng)
    assert np.mean(a.label(x) != b.label(x)) == pytest.approx(0.12, abs=0.003)
    # реальный дрейф сохраняет P(y): доля положительных по новому концепту та же
    x_new = _sample(b, 300_000, rng)
    assert b.label(x_new).mean() == pytest.approx(0.15, abs=0.003)


def test_monte_carlo_virtual_and_prior_preserve_labels() -> None:
    spec = io.from_dict(
        {
            "stream": {"n_samples": 20_000, "n_features": 6, "minority_share": 0.3},
            "events": [
                {"position": 5_000, "kind": "virtual", "magnitude": 0.5, "affected_share": 0.5},
                {"position": 12_000, "kind": "prior", "magnitude": 0.25},
            ],
        }
    )
    chain = S.build_chain(spec)
    c0, c1, c2 = chain.states
    rng = np.random.default_rng(4)
    x = rng.standard_normal((200_000, 6)) * 3.0
    # P(y|X) не меняется ни при виртуальном, ни при prior-дрейфе
    assert np.array_equal(c0.label(x), c1.label(x))
    assert np.array_equal(c1.label(x), c2.label(x))
    assert c2.prior == pytest.approx(0.55)


def _sample_full(state: S.ConceptState, n: int, rng: np.random.Generator) -> np.ndarray:
    """Выборка X по схеме label shift с текущей долей класса state.prior."""
    return _sample(state, n, rng)


def test_real_after_prior_equals_joint_tv_and_warns() -> None:
    """После prior-дрейфа величина реального дрейфа остаётся TV совместных распределений.

    Для выборки из P_a отношение плотностей p_b(x, y)/p_a(x, y) = 1[g_b(x) = y],
    поэтому TV = P_a(g_a ≠ g_b), что и считает build_chain. P(X) при этом
    меняется (W3), что проверяется по среднему проекции на w_a.
    """
    spec = io.from_dict(
        {
            "stream": {"n_samples": 20_000, "n_features": 6, "minority_share": 0.5},
            "events": [
                {"position": 5_000, "kind": "prior", "magnitude": 0.3},
                {"position": 12_000, "kind": "real", "magnitude": 0.25},
            ],
        }
    )
    chain = S.build_chain(spec)
    assert [v.code for v in chain.violations] == ["W3"]
    a, b = chain.states[1], chain.states[2]
    rng = np.random.default_rng(11)
    xa = _sample_full(a, 300_000, rng)
    assert np.mean(a.label(xa) != b.label(xa)) == pytest.approx(0.25, abs=0.004)
    xb = _sample_full(b, 300_000, rng)
    shift = np.mean((xa - a.mu) @ a.w) - np.mean((xb - b.mu) @ a.w)
    assert abs(shift) > 0.05  # P(X) действительно изменилось


def test_virtual_tv_exact_under_reweighting() -> None:
    spec = io.from_dict(
        {
            "stream": {"n_samples": 20_000, "n_features": 6, "minority_share": 0.3},
            "events": [
                {"position": 5_000, "kind": "prior", "magnitude": 0.4},
                {"position": 12_000, "kind": "virtual", "magnitude": 0.35, "affected_share": 0.5},
            ],
        }
    )
    chain = S.build_chain(spec)
    a, b = chain.states[1], chain.states[2]
    rng = np.random.default_rng(12)
    x = _sample_full(a, 400_000, rng)
    delta_mu = b.mu - a.mu
    # классовые множители сокращаются (сдвиг ⊥ w), остаётся отношение гауссиан
    log_ratio = (x - a.mu) @ delta_mu - 0.5 * float(delta_mu @ delta_mu)
    tv = np.mean(np.clip(1.0 - np.exp(log_ratio), 0.0, None))
    assert tv == pytest.approx(0.35, abs=0.004)
    assert chain.geometry[1].realized["virtual"] == pytest.approx(0.35, abs=1e-12)


def test_return_without_change_is_rejected() -> None:
    spec = io.from_dict(
        {
            "stream": {"n_samples": 20_000, "minority_share": 0.5},
            "events": [
                {"position": 4_000, "kind": "prior", "magnitude": 0.3},
                {"position": 8_000, "kind": "prior", "magnitude": 0.3},
                {"position": 13_000, "returns_to": 0},
            ],
        }
    )
    chain = S.build_chain(spec)
    assert [st.prior for st in chain.states] == pytest.approx([0.5, 0.8, 0.5, 0.5])
    assert any(v.code == "C7" for v in chain.violations)

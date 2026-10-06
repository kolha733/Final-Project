"""Тесты генераторов потоков: переходы, данные, ground truth, воспроизводимость."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scendrift.generators import generate, timeline, transition
from scendrift.generators import real as real_family
from scendrift.generators.classic import concept_variants
from scendrift.generators.diagnostics import (
    digest,
    empirical_magnitude,
    stable_after,
    transition_curve,
)
from scendrift.scenario import io
from scendrift.scenario import semantics as S
from scendrift.scenario.report import ScenarioValidationError
from scendrift.scenario.schema import DriftEvent
from scendrift.scenario.validation import check


def _hp(events: list[dict[str, Any]], **stream: Any):
    base = {"n_samples": 20_000, "n_features": 10}
    base.update(stream)
    return io.from_dict({"stream": base, "events": events, "seed": 5})


# --- функции перехода ------------------------------------------------------


@pytest.mark.parametrize("shape", ["linear", "sigmoid"])
def test_transition_bounds_and_monotonicity(shape: str) -> None:
    ev = DriftEvent(position=5_000, width=1_000, form="gradual", magnitude=0.2, shape=shape)
    t = np.arange(4_000, 6_000)
    p = transition(t, ev)
    assert p[t < ev.onset].max() == 0.0 and p[t >= ev.end].min() == 1.0
    assert np.all(np.diff(p) >= 0)
    assert transition(np.array([ev.position]), ev)[0] == pytest.approx(0.5, abs=0.01)


def test_sudden_transition_is_step() -> None:
    ev = DriftEvent(position=5_000, magnitude=0.2)
    assert transition(np.array([4_999, 5_000]), ev).tolist() == [0.0, 1.0]


def test_timeline_states() -> None:
    spec = _hp([{"position": 5_000, "magnitude": 0.2}, {"position": 12_000, "magnitude": 0.2}])
    tl = timeline(spec)
    assert tl.src[0] == 0 and tl.dst[0] == 0 and tl.event[0] == -1
    assert tl.dst[6_000] == 1 and tl.p[6_000] == 1.0
    assert tl.src[12_000] == 1 and tl.dst[12_000] == 2


# --- hyperplane_gauss ------------------------------------------------------


def test_reproducibility_and_seed_separation(demo_spec) -> None:
    a, b = generate(demo_spec), generate(demo_spec)
    assert digest(a) == digest(b)
    c = generate(demo_spec, seed=99)
    assert digest(c) != digest(a)
    # геометрия концептов не зависит от seed реализации
    assert [iv.realized for iv in a.ground_truth.intervals] == [
        iv.realized for iv in c.ground_truth.intervals
    ]


def test_noise_does_not_change_features(demo_spec) -> None:
    clean = demo_spec.model_copy(
        update={"stream": demo_spec.stream.model_copy(update={"label_noise": 0.0})}
    )
    a, b = generate(demo_spec), generate(clean)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y_clean, b.y_clean)
    assert not np.array_equal(a.y, b.y)


def test_prior_and_noise_rates() -> None:
    spec = _hp([], minority_share=0.2, label_noise=0.1, n_samples=50_000)
    data = generate(spec)
    se = np.sqrt(0.2 * 0.8 / 50_000)
    assert abs(data.y_clean.mean() - 0.2) < 4 * se
    noise = np.mean(data.y != data.y_clean)
    assert abs(noise - 0.1) < 4 * np.sqrt(0.09 / 50_000)


def test_clean_labels_follow_current_concept(demo_spec) -> None:
    data = generate(demo_spec)
    chain = S.build_chain(demo_spec)
    for k in range(demo_spec.n_events):
        seg = stable_after(demo_spec, k)
        assert np.array_equal(chain.states[k + 1].label(data.X[seg]), data.y_clean[seg])


@pytest.mark.parametrize(
    "event",
    [
        {"kind": "real", "magnitude": 0.25, "affected_share": 0.5},
        {"kind": "virtual", "magnitude": 0.3, "affected_share": 0.5},
        {"kind": "prior", "magnitude": 0.2},
    ],
)
def test_empirical_magnitude_matches_target(event: dict[str, Any]) -> None:
    spec = _hp([{"position": 10_000, **event}], minority_share=0.3, n_samples=40_000)
    data = generate(spec)
    est, se = empirical_magnitude(spec, data, 0)
    assert abs(est - event["magnitude"]) < 4 * se + 1e-9


@pytest.mark.parametrize(
    "form,shape", [("gradual", "linear"), ("gradual", "sigmoid"), ("incremental", "linear")]
)
def test_transition_curve_follows_theory(form: str, shape: str) -> None:
    spec = _hp(
        [
            {
                "position": 15_000,
                "form": form,
                "width": 6_000,
                "shape": shape,
                "kind": "real",
                "magnitude": 0.3,
            }
        ],
        n_samples=40_000,
        minority_share=0.3,
    )
    curve = transition_curve(spec, generate(spec), 0, window=1_000).dropna()
    assert np.max(np.abs(curve["empirical"] - curve["theory_smoothed"])) < 0.05


def test_invalid_scenario_is_rejected() -> None:
    spec = _hp([{"position": 500, "magnitude": 0.2}])
    with pytest.raises(ScenarioValidationError):
        generate(spec)


def test_ground_truth_content(demo_spec) -> None:
    gt = generate(demo_spec).ground_truth
    assert [iv.kind for iv in gt.intervals] == ["real", "virtual", "recurring"]
    assert gt.intervals[2].magnitude is None and gt.intervals[2].returns_to == 0
    assert gt.drift_mask().sum() == sum(iv.end - iv.onset for iv in gt.intervals)
    assert gt.to_records()[0]["realized_real"] == pytest.approx(0.2)


# --- классические генераторы river -----------------------------------------


def _classic(family: str, d: int, **extra: Any):
    data = {"stream": {"family": family, "n_samples": 12_000, "n_features": d}}
    data["drifts"] = {"schedule": {"count": 2}, "defaults": {"form": "gradual", "width": 800}}
    data["drifts"].update(extra)
    return io.from_dict(data)


def test_classic_generation_and_realized() -> None:
    spec = _classic("river:SEA", 3)
    assert check(spec).ok
    data = generate(spec)
    assert data.X.shape == (12_000, 3) and set(np.unique(data.y)) <= {0, 1}
    assert concept_variants(spec) == [0, 1, 2]
    realized = [iv.realized["real"] for iv in data.ground_truth.intervals]
    assert all(0.0 < r < 1.0 for r in realized)
    assert digest(generate(spec)) == digest(data)


@pytest.mark.parametrize(
    "stream, events, code",
    [
        ({"n_features": 4}, None, "C8"),
        ({"minority_share": 0.3}, None, "C8"),
        ({"family_params": {"variants": [0, 0, 1]}}, None, "C8"),
        ({"family_params": {"variants": [0, 7, 1]}}, None, "C8"),
        ({}, [{"position": 4_000, "form": "incremental", "width": 500}], "C8"),
        ({}, [{"position": 4_000, "magnitude": 0.2}], "C2"),
        ({}, [{"position": 4_000, "kind": "virtual"}], "C8"),
    ],
)
def test_classic_constraints(stream: dict, events: list | None, code: str) -> None:
    base = {"family": "river:SEA", "n_samples": 12_000, "n_features": 3}
    base.update(stream)
    data: dict[str, Any] = {"stream": base}
    if events is None:
        data["drifts"] = {"schedule": {"count": 2}}
    else:
        data["events"] = events
    assert code in check(io.from_dict(data)).codes()


def test_classic_recurring_uses_previous_variant() -> None:
    spec = io.from_dict(
        {
            "stream": {"family": "river:STAGGER", "n_samples": 15_000, "n_features": 3},
            "events": [{"position": 4_000}, {"position": 9_000, "returns_to": 0}],
        }
    )
    assert concept_variants(spec) == [0, 1, 0]
    gt = generate(spec).ground_truth
    assert gt.intervals[0].realized["real"] == pytest.approx(gt.intervals[1].realized["real"])


# --- реальные данные -------------------------------------------------------


def _real(family: str, d: int, events: list[dict[str, Any]], **stream: Any):
    base = {"family": family, "n_samples": 20_000, "n_features": d, "minority_share": 0.45}
    base.update(stream)
    return io.from_dict({"stream": base, "events": events})


@pytest.mark.parametrize("family,d", [("real:bananas", 2), ("real:phishing", 9)])
def test_real_injection_magnitudes(family: str, d: int) -> None:
    spec = _real(
        family,
        d,
        [
            {"position": 4_000, "kind": "real", "magnitude": 0.2},
            {"position": 9_000, "kind": "virtual", "magnitude": 0.3},
            {"position": 14_000, "kind": "prior", "magnitude": 0.15},
        ],
    )
    chain = real_family.build_real_chain(spec)
    n_rows = real_family.load(family).n_rows
    g = chain.geometry
    assert abs(g[0].realized["real"] - 0.2) <= 1.0 / n_rows
    assert g[1].realized["virtual"] == pytest.approx(0.3, abs=1e-9)
    assert g[2].realized["prior"] == pytest.approx(0.15, abs=1e-12)
    s = chain.states
    assert np.array_equal(s[0].p, s[1].p)  # real: P(X) не меняется
    assert np.array_equal(s[1].lab, s[2].lab)  # virtual: метки строк (P(y|X)) не меняются
    data = generate(spec)
    assert data.X.shape == (20_000, d)
    rows = {tuple(r) for r in real_family.load(family).X}
    assert all(tuple(x) in rows for x in data.X[:200])


def test_real_constraints_and_warnings() -> None:
    bad_dim = _real("real:bananas", 3, [{"position": 5_000, "kind": "real", "magnitude": 0.2}])
    assert "C8" in check(bad_dim).codes()
    inc_return = _real(
        "real:bananas",
        2,
        [
            {"position": 4_000, "kind": "real", "magnitude": 0.2},
            {"position": 10_000, "returns_to": 0, "form": "incremental", "width": 1_000},
        ],
    )
    assert "C8" in check(inc_return).codes()
    side = _real("real:phishing", 9, [{"position": 5_000, "kind": "virtual", "magnitude": 0.5}])
    assert "W4" in check(side).codes()


def test_real_incremental_generation() -> None:
    spec = _real(
        "real:bananas",
        2,
        [
            {
                "position": 10_000,
                "form": "incremental",
                "width": 4_000,
                "kind": "prior",
                "magnitude": 0.3,
            }
        ],
    )
    data = generate(spec)
    before, after = data.y_clean[:7_000].mean(), data.y_clean[13_000:].mean()
    assert after - before == pytest.approx(0.3, abs=0.03)

"""Тесты схемы сценария: локальные ограничения C1–C3 и компактная форма."""

from __future__ import annotations

import pydantic
import pytest

from scendrift.evaluation.protocol import EvaluationSpec
from scendrift.scenario.enums import DriftForm, DriftKind
from scendrift.scenario.schema import (
    DriftEvent,
    DriftPlan,
    ScenarioSpec,
    StreamSpec,
    expand_plan,
    layout_uniform,
)


def test_event_defaults_and_interval() -> None:
    e = DriftEvent(position=5_000, magnitude=0.3)
    assert e.kind is DriftKind.REAL and e.form is DriftForm.SUDDEN
    assert (e.onset, e.end) == (5_000, 5_000)
    g = DriftEvent(position=5_000, width=1_001, form="gradual", magnitude=0.3)
    assert (g.onset, g.end) == (4_500, 5_501)
    assert g.end - g.onset == g.width


@pytest.mark.parametrize(
    "kwargs, code",
    [
        ({"position": 100, "width": 10, "magnitude": 0.1}, "C1"),  # sudden с шириной
        ({"position": 100, "form": "gradual", "width": 1, "magnitude": 0.1}, "C1"),
        ({"position": 10, "form": "gradual", "width": 100, "magnitude": 0.1}, "C1"),
        ({"position": 100, "returns_to": 0, "magnitude": 0.1}, "C3"),
        ({"position": 100, "returns_to": 0, "kind": "real"}, "C3"),
    ],
)
def test_event_local_constraints(kwargs: dict, code: str) -> None:
    with pytest.raises(pydantic.ValidationError, match=code):
        DriftEvent(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": 100, "magnitude": 0.0},
        {"position": 100, "magnitude": 1.5},
        {"position": 100, "magnitude": 0.1, "affected_share": 0.0},
        {"position": 100, "magnitude": 0.1, "kind": "concept"},
        {"position": -1, "magnitude": 0.1},
        {"position": 100, "magnitude": 0.1, "unknown_field": 1},
    ],
)
def test_event_domains(kwargs: dict) -> None:
    with pytest.raises(pydantic.ValidationError):
        DriftEvent(**kwargs)


def test_magnitude_is_checked_by_family() -> None:
    # величина не обязательна на уровне события: её требует калиброванное семейство (C2)
    from scendrift.scenario import io
    from scendrift.scenario.validation import check

    e = DriftEvent(position=5_000)
    assert e.magnitude is None and e.kind is DriftKind.REAL
    spec = io.from_dict({"events": [{"position": 9_000}]})
    assert "C2" in check(spec).codes()


def test_recurring_event_has_no_kind() -> None:
    e = DriftEvent(position=100, returns_to=0)
    assert e.is_recurring and e.kind is None and e.magnitude is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 999},
        {"n_features": 1},
        {"minority_share": 0.6},
        {"minority_share": 0.0},
        {"label_noise": 0.5},
    ],
)
def test_stream_domains(kwargs: dict) -> None:
    with pytest.raises(pydantic.ValidationError):
        StreamSpec(**kwargs)


def test_spec_is_frozen_and_strict() -> None:
    spec = ScenarioSpec()
    with pytest.raises(pydantic.ValidationError):
        spec.seed = 3  # type: ignore[misc]
    with pytest.raises(pydantic.ValidationError):
        ScenarioSpec(foo=1)  # type: ignore[call-arg]
    assert spec.with_seed(5).seed == 5 and spec.seed == 0


def test_layout_uniform_respects_footprints() -> None:
    centers = layout_uniform([0, 1_000, 500], 1_000, 20_000, 2_000)
    events = [
        DriftEvent(position=centers[0], magnitude=0.1),
        DriftEvent(position=centers[1], width=1_000, form="gradual", magnitude=0.1),
        DriftEvent(position=centers[2], width=500, form="gradual", magnitude=0.1),
    ]
    assert events[0].onset >= 1_000
    for a, b in zip(events, events[1:], strict=False):
        assert b.onset >= a.end + 2_000
    assert events[-1].end + 2_000 <= 20_000


def _expand(plan: dict, **stream) -> tuple[DriftEvent, ...]:
    return expand_plan(DriftPlan.model_validate(plan), StreamSpec(**stream), EvaluationSpec())


def test_expand_uniform_single_drift_centered() -> None:
    (e,) = _expand({"schedule": {"count": 1}, "defaults": {"magnitude": 0.2}})
    # след [onset, onset + Δ) по центру отрезка [W, n) = [1000, 20000)
    assert e.onset == 1_000 + (19_000 - 2_000) // 2


def test_expand_random_is_deterministic_and_valid() -> None:
    plan = {
        "schedule": {"mode": "random", "count": 4, "min_gap": 100},
        "defaults": {"form": "gradual", "width": 800, "magnitude": 0.2},
    }
    a = _expand(plan, concept_seed=3)
    b = _expand(plan, concept_seed=3)
    c = _expand(plan, concept_seed=4)
    assert a == b and a != c
    assert a[0].onset >= 1_000
    for x, y in zip(a, a[1:], strict=False):
        assert y.onset >= x.end + 2_000 + 100
    assert a[-1].end + 2_000 <= 20_000


def test_expand_explicit_and_overrides() -> None:
    events = _expand(
        {
            "schedule": {"mode": "explicit", "count": 3, "positions": [4_000, 9_000, 15_000]},
            "defaults": {"kind": "real", "magnitude": 0.2, "form": "gradual", "width": 600},
            "overrides": [
                {"index": 1, "kind": "prior", "magnitude": 0.1, "form": "sudden"},
                {"index": 2, "returns_to": 0},
            ],
        }
    )
    assert [e.position for e in events] == [4_000, 9_000, 15_000]
    assert events[1].kind is DriftKind.PRIOR and events[1].width == 0
    assert events[2].is_recurring and events[2].width == 600


def test_expand_bad_override_index() -> None:
    with pytest.raises(ValueError):
        _expand(
            {
                "schedule": {"count": 1},
                "defaults": {"magnitude": 0.2},
                "overrides": [{"index": 3, "magnitude": 0.1}],
            }
        )


def test_schedule_plan_validation() -> None:
    with pytest.raises(pydantic.ValidationError):
        DriftPlan.model_validate({"schedule": {"mode": "explicit", "count": 2, "positions": [1]}})
    with pytest.raises(pydantic.ValidationError):
        DriftPlan.model_validate({"schedule": {"mode": "uniform", "positions": [1]}})


def test_with_seed_validates() -> None:
    with pytest.raises(pydantic.ValidationError):
        ScenarioSpec().with_seed(-5)


def test_kind_default_for_any_mapping() -> None:
    from types import MappingProxyType

    e = DriftEvent.model_validate(MappingProxyType({"position": 100, "magnitude": 0.2}))
    assert e.kind is DriftKind.REAL


def test_family_params_are_normalized_and_finite() -> None:
    import numpy as np

    s = StreamSpec(family_params={"a": np.int64(3), "b": (1.0, np.float32(0.5)), "c": -0.0})
    assert s.family_params == {"a": 3, "b": [1.0, 0.5], "c": 0.0}
    assert type(s.family_params["a"]) is int
    for bad in (float("nan"), float("inf")):
        with pytest.raises(pydantic.ValidationError):
            StreamSpec(family_params={"x": bad})


def test_negative_zero_is_normalized() -> None:
    assert str(StreamSpec(label_noise=-0.0).label_noise) == "0.0"


def test_contradictory_recurring_override_rejected() -> None:
    with pytest.raises(ValueError, match="C3"):
        _expand(
            {
                "schedule": {"count": 2},
                "defaults": {"magnitude": 0.2},
                "overrides": [{"index": 1, "returns_to": 0, "kind": "virtual"}],
            }
        )


def test_layout_valid_for_unequal_widths() -> None:
    widths = [0, 10_000]
    centers = layout_uniform(widths, 1_000, 20_000, 2_000)
    onsets = [c - w // 2 for c, w in zip(centers, widths, strict=True)]
    ends = [o + w for o, w in zip(onsets, widths, strict=True)]
    assert onsets[0] >= 1_000 and onsets[1] >= ends[0] + 2_000 and ends[1] + 2_000 <= 20_000

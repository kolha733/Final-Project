"""Тесты ограничений валидности C4–C13, предупреждений W1–W2 и исправления сценариев."""

from __future__ import annotations

from typing import Any

import pytest

from scendrift.scenario import io
from scendrift.scenario import semantics as S
from scendrift.scenario.enums import DriftForm, DriftKind
from scendrift.scenario.families import FamilyInfo, get_family, register_family
from scendrift.scenario.report import CONSTRAINTS, ScenarioValidationError
from scendrift.scenario.schema import ScenarioSpec
from scendrift.scenario.validation import check, concept_ids, ensure_valid, is_valid, repair


def _spec(events: list[dict[str, Any]], **stream: Any) -> ScenarioSpec:
    base = {"n_samples": 20_000, "n_features": 10}
    base.update(stream)
    return io.from_dict({"stream": base, "events": events})


def test_demo_is_valid(demo_spec: ScenarioSpec) -> None:
    report = check(demo_spec)
    assert report.ok, str(report)
    assert report.codes() == {"W2"}
    assert ensure_valid(demo_spec) is demo_spec


@pytest.mark.parametrize(
    "events, stream, code",
    [
        ([{"position": 500, "magnitude": 0.2}], {}, "C4"),
        ([{"position": 19_000, "magnitude": 0.2}], {}, "C5"),
        ([{"position": 5_000, "magnitude": 0.2}, {"position": 6_000, "magnitude": 0.2}], {}, "C6"),
        ([{"position": 9_000, "magnitude": 0.2}, {"position": 5_000, "magnitude": 0.2}], {}, "C6"),
        ([{"position": 5_000, "returns_to": 0}], {}, "C7"),  # возврат к текущему концепту
        ([{"position": 5_000, "returns_to": 2}], {}, "C7"),  # концепта ещё нет
        ([{"position": 5_000, "magnitude": 0.2, "affected_share": 0.1}], {}, "C9"),
        ([{"position": 5_000, "magnitude": 0.5, "affected_share": 0.2}], {}, "C10"),
        ([{"position": 5_000, "magnitude": 0.995, "kind": "virtual"}], {}, "C11"),
        ([{"position": 5_000, "magnitude": 0.6, "kind": "prior"}], {}, "C12"),
        ([{"position": 5_000, "magnitude": 0.2}], {"family": "nope"}, "C13"),
        ([{"position": 5_000, "magnitude": 0.2}], {"family_params": {"noize": 0.3}}, "C8"),
    ],
)
def test_constraint_violations(events: list, stream: dict, code: str) -> None:
    report = check(_spec(events, **stream))
    assert not report.ok
    assert code in report.codes()
    assert code in CONSTRAINTS
    with pytest.raises(ScenarioValidationError):
        report.raise_if_invalid()


def test_recurring_after_two_events_is_valid() -> None:
    spec = _spec(
        [
            {"position": 4_000, "magnitude": 0.2},
            {"position": 9_000, "magnitude": 0.2},
            {"position": 14_000, "returns_to": 1},
        ]
    )
    assert is_valid(spec)
    assert concept_ids(spec) == [0, 1, 2, 1]
    # A → B → A → B: повторный возврат к уже текущему концепту недопустим
    bad = _spec(
        [
            {"position": 4_000, "magnitude": 0.2},
            {"position": 8_000, "returns_to": 0},
            {"position": 12_000, "returns_to": 0},
        ]
    )
    assert "C7" in check(bad).codes()


def test_warnings_do_not_invalidate() -> None:
    spec = _spec([{"position": 5_000, "magnitude": 0.03}], label_noise=0.4)
    report = check(spec)
    assert report.ok and "W1" in report.codes()


def test_family_support_c8() -> None:
    name = "test_only_real_sudden"
    if get_family(name) is None:
        register_family(
            FamilyInfo(
                name=name,
                description="тестовое семейство",
                kinds=frozenset({DriftKind.REAL}),
                forms=frozenset({DriftForm.SUDDEN}),
                supports_recurring=False,
                calibrated=False,
            )
        )
    spec = _spec(
        [
            {"position": 4_000, "magnitude": 0.2, "kind": "virtual"},
            {"position": 9_000, "magnitude": 0.2, "form": "gradual", "width": 100},
            {"position": 14_000, "returns_to": 0},
        ],
        family=name,
    )
    paths = {v.path for v in check(spec).errors if v.code == "C8"}
    assert paths == {"events[0].kind", "events[1].form", "events[2].returns_to"}
    with pytest.raises(KeyError):
        register_family(get_family(name))  # type: ignore[arg-type]


def test_repair_clips_magnitude_to_feasible_max() -> None:
    spec = _spec([{"position": 5_000, "magnitude": 0.5, "affected_share": 0.2}])
    result = repair(spec)
    m = result.spec.events[0].magnitude
    assert result.report.ok and m is not None and m < 0.5
    chain = S.build_chain(result.spec)
    assert chain.geometry[0].realized["real"] == pytest.approx(m, abs=1e-9)
    assert any(a.startswith("C10") for a in result.actions)


def test_repair_affected_share_and_timeline() -> None:
    spec = _spec(
        [
            {"position": 500, "magnitude": 0.2, "affected_share": 0.05},
            {"position": 1_200, "magnitude": 0.2, "kind": "virtual"},
        ]
    )
    result = repair(spec)
    assert result.report.ok
    assert result.spec.events[0].affected_share == pytest.approx(0.2)
    assert result.spec.events[0].onset >= 1_000


def test_repair_drops_events_that_do_not_fit() -> None:
    events = [{"position": 2_000 + 1_000 * i, "magnitude": 0.1} for i in range(12)]
    result = repair(_spec(events, n_samples=10_000))
    assert result.report.ok
    assert len(result.spec.events) == 4  # (10000 − 1000) // (0 + 2000)
    assert any("отброшено" in a for a in result.actions)


def test_repair_sequential_dependency() -> None:
    # оба prior-события исходно недопустимы; после исправления первого
    # (π: 0,5 → 0,99) второе становится допустимым (0,99 → 0,49) без изменений
    spec = _spec(
        [
            {"position": 4_000, "magnitude": 0.6, "kind": "prior"},
            {"position": 9_000, "magnitude": 0.5, "kind": "prior"},
        ],
        minority_share=0.5,
    )
    before = {v.path for v in check(spec).errors if v.code == "C12"}
    assert before == {"events[0].magnitude", "events[1].magnitude"}
    result = repair(spec)
    assert result.report.ok
    assert result.spec.events[0].magnitude == pytest.approx(0.49)
    assert result.spec.events[1].magnitude == pytest.approx(0.5)
    priors = [st.prior for st in S.build_chain(result.spec).states]
    assert priors == pytest.approx([0.5, 0.99, 0.49])


def test_repair_unrepairable_raises() -> None:
    with pytest.raises(ScenarioValidationError):
        repair(_spec([{"position": 5_000, "magnitude": 0.2}], family="nope"))
    with pytest.raises(ScenarioValidationError):
        repair(_spec([{"position": 5_000, "returns_to": 0}]))


def test_protocol_must_fit_stream_c14() -> None:
    spec = io.from_dict(
        {"stream": {"n_samples": 2_000}, "evaluation": {"warmup": 1_500}, "events": []}
    )
    assert "C14" in check(spec).codes()


def test_repair_handles_unequal_widths() -> None:
    # следы 2000 + 12000 ≤ n − W = 19000: допустимая раскладка существует
    spec = _spec(
        [
            {"position": 1_500, "magnitude": 0.2},
            {"position": 6_000, "form": "gradual", "width": 10_000, "magnitude": 0.2},
        ]
    )
    result = repair(spec)
    assert result.report.ok and len(result.spec.events) == 2


def test_repair_refuses_to_drop_all_events() -> None:
    spec = io.from_dict(
        {
            "stream": {"n_samples": 2_500},
            "evaluation": {"warmup": 0},
            "events": [{"position": 1_200, "form": "gradual", "width": 1_000, "magnitude": 0.2}],
        }
    )
    with pytest.raises(ScenarioValidationError):
        repair(spec)


def test_real_after_prior_gives_w3_warning() -> None:
    spec = _spec(
        [
            {"position": 4_000, "kind": "prior", "magnitude": 0.2},
            {"position": 9_000, "kind": "real", "magnitude": 0.2},
        ],
        minority_share=0.3,
    )
    report = check(spec)
    assert report.ok and "W3" in report.codes()

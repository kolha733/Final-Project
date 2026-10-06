"""Тесты пространства параметров и таксономии."""

from __future__ import annotations

import numpy as np
import pytest

from scendrift.scenario import io
from scendrift.scenario.space import (
    DEFAULT_SPACE,
    Categorical,
    FloatDomain,
    IntDomain,
    ParameterDef,
    ParameterSpace,
)
from scendrift.scenario.taxonomy import classify, signature


def test_float_domain() -> None:
    d = FloatDomain(0.1, 0.5)
    assert d.from_unit(0.0) == pytest.approx(0.1)
    assert d.from_unit(0.999999) == pytest.approx(0.5, abs=1e-6)
    assert d.levels(3) == pytest.approx([0.1, 0.3, 0.5])
    lg = FloatDomain(0.01, 1.0, log=True)
    assert lg.from_unit(0.5) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        FloatDomain(1.0, 1.0)


def test_int_domain_is_uniform_over_values() -> None:
    d = IntDomain(1, 5)
    u = (np.arange(10_000) + 0.5) / 10_000
    values = [d.from_unit(x) for x in u]
    counts = np.bincount(values)[1:]
    assert set(values) == {1, 2, 3, 4, 5}
    assert counts.min() == counts.max() == 2_000
    assert d.contains(3) and not d.contains(True) and not d.contains(6)
    lg = IntDomain(2, 50, log=True)
    assert lg.from_unit(0.0) == 2 and lg.from_unit(0.9999) == 50
    assert lg.levels(4)[0] == 2 and lg.levels(4)[-1] == 50


def test_categorical_domain() -> None:
    c = Categorical(("a", "b", "c"))
    assert [c.from_unit(u) for u in (0.0, 0.34, 0.99)] == ["a", "b", "c"]
    assert c.levels() == ["a", "b", "c"]


def test_conditional_activity() -> None:
    values = DEFAULT_SPACE.from_unit([0.1] * len(DEFAULT_SPACE.free_names()))
    assert values["form"] == "sudden"
    assert values["width"] == DEFAULT_SPACE["width"].default  # неактивен при sudden
    assert values["recurring"] is False  # K = 1 → чередование неактивно
    hi = DEFAULT_SPACE.from_unit([0.9] * len(DEFAULT_SPACE.free_names()))
    assert hi["form"] == "incremental" and hi["width"] != DEFAULT_SPACE["width"].default


def test_space_operations() -> None:
    assert "warmup" not in DEFAULT_SPACE.free_names()
    assert DEFAULT_SPACE.defaults()["acceptance_window"] == 2_000
    sub = DEFAULT_SPACE.subset(["form", "width", "magnitude"])
    assert sub.names == ["form", "width", "magnitude"]
    fixed = sub.fix(form="gradual")
    assert fixed.free_names() == ["width", "magnitude"]
    point = fixed.from_unit([0.0, 0.0])
    assert point["form"] == "gradual" and point["width"] == 100
    assert DEFAULT_SPACE.out_of_domain({"magnitude": 0.9, "form": "sudden"}) == ["magnitude"]
    rows = DEFAULT_SPACE.table()
    assert len(rows) == len(DEFAULT_SPACE) and rows[0]["параметр"] == "n_samples"
    with pytest.raises(ValueError):
        ParameterSpace(
            [ParameterDef("a", "a", "x", IntDomain(1, 2), 1, "", "", active_if=(("b", (1,)),))]
        )


def test_taxonomy(demo_spec) -> None:
    c = classify(demo_spec)
    assert c.n_drifts == 3 and c.recurring
    assert c.kinds == ("real", "virtual") and c.forms == ("gradual",)
    assert c.severity == "high" and c.speed == "slow"
    assert c.scope == "partial" and c.balance == "imbalanced" and c.noise == "noisy"
    sig = signature(io.from_dict({"events": [{"position": 9_000, "magnitude": 0.05}]}))
    assert sig == "real/sudden/K=1/low/abrupt/global/balanced/clean"

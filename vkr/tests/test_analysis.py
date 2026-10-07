"""Тесты анализа результатов: Спирмен, правила отказа, согласие ранжирований."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scendrift.evaluation.analysis import (
    failure_rules,
    monotone_share,
    rank_agreement,
    scenario_level,
    spearman_table,
)
from scendrift.evaluation.protocol import AdaptationPolicy
from scendrift.evaluation.runner import with_protocol
from scendrift.scenario import io


def _data(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.02, 0.5, n)
    width = rng.integers(100, 4_000, n)
    kind = rng.choice(["real", "virtual", "prior"], n)
    recall = np.clip(2.5 * m + rng.normal(0, 0.05, n) - 0.6 * (kind == "virtual"), 0, 1)
    return pd.DataFrame(
        {
            "magnitude": m,
            "width": width,
            "kind": kind,
            "recall": recall,
            "detector": "X",
            "scenario_id": [f"s{i}" for i in range(n)],
        }
    )


def test_scenario_level_averages_repeats() -> None:
    frame = pd.DataFrame(
        {
            "scenario_id": ["a", "a", "b"],
            "detector": ["X", "X", "X"],
            "f1": [0.2, 0.4, 1.0],
            "magnitude": [0.1, 0.1, 0.3],
        }
    )
    out = scenario_level(frame, ["f1"], keep=["magnitude"])
    assert out.set_index("scenario_id")["f1"].to_dict() == pytest.approx({"a": 0.3, "b": 1.0})
    assert out.set_index("scenario_id")["magnitude"]["a"] == 0.1


def test_spearman_table_signs() -> None:
    table = spearman_table(_data(), "recall", ["magnitude", "width"]).set_index("параметр")
    assert table.loc["magnitude", "ρ"] > 0.5 and table.loc["magnitude", "p (Холм)"] < 1e-6
    assert table.loc["width", "p (Холм)"] > 0.01


def test_failure_rules_recover_threshold() -> None:
    data = _data(300)
    data["отказ"] = data["recall"] < 0.5
    res = failure_rules(data, ["magnitude", "width", "kind"], "отказ", max_depth=2)
    top = res.rules.iloc[0]
    assert "m ≤" in top["правило"] and top["доля отказов"] > 0.9
    assert res.cv_accuracy > res.baseline_accuracy
    assert any("вид = virtual" in r for r in res.rules["правило"])


def test_rank_agreement_and_monotone_share() -> None:
    a = pd.Series({"x": 1.0, "y": 2.0, "z": 3.0, "w": 4.0})
    assert rank_agreement(a, a)["tau"] == pytest.approx(1.0)
    assert rank_agreement(a, -a)["tau"] == pytest.approx(-1.0)
    assert monotone_share([1.0, 0.8, 0.8, 0.9]) == pytest.approx(2 / 3)


def test_with_protocol_validates() -> None:
    spec = io.from_dict({"events": [{"position": 10_000, "magnitude": 0.2}]})
    changed = with_protocol(spec, on_detection="none")
    assert changed.evaluation.on_detection is AdaptationPolicy.NONE
    assert io.scenario_id(changed) != io.scenario_id(spec)
    with pytest.raises(ValueError):
        with_protocol(spec, on_detection="sometimes")

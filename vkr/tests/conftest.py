"""Общие фикстуры тестов."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scendrift.scenario import io  # noqa: E402
from scendrift.scenario.schema import ScenarioSpec  # noqa: E402


@pytest.fixture()
def demo_spec() -> ScenarioSpec:
    """Сценарий из трёх событий: real (gradual), virtual, возврат к концепту 0."""
    return io.from_dict(
        {
            "name": "demo",
            "stream": {
                "n_samples": 20_000,
                "n_features": 10,
                "minority_share": 0.3,
                "label_noise": 0.05,
                "concept_seed": 7,
            },
            "drifts": {
                "schedule": {"mode": "uniform", "count": 3},
                "defaults": {
                    "form": "gradual",
                    "kind": "real",
                    "width": 1_000,
                    "magnitude": 0.2,
                    "affected_share": 0.5,
                },
                "overrides": [
                    {"index": 1, "kind": "virtual", "magnitude": 0.4},
                    {"index": 2, "returns_to": 0},
                ],
            },
            "seed": 42,
        }
    )


@pytest.fixture()
def examples_dir() -> Path:
    return ROOT / "configs" / "examples"

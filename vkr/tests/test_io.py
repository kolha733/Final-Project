"""Тесты сериализации: round-trip, идентификаторы, JSON Schema, примеры конфигураций."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scendrift.scenario import io
from scendrift.scenario.schema import ScenarioSpec
from scendrift.scenario.validation import check


def test_yaml_roundtrip(demo_spec: ScenarioSpec, tmp_path: Path) -> None:
    path = io.save(demo_spec, tmp_path / "demo.yaml")
    text = path.read_text(encoding="utf-8")
    assert text.startswith(f"# id: {io.scenario_id(demo_spec)}")
    loaded = io.load(path)
    assert loaded == demo_spec
    assert io.scenario_id(loaded) == io.scenario_id(demo_spec)


def test_json_roundtrip(demo_spec: ScenarioSpec, tmp_path: Path) -> None:
    loaded = io.load(io.save(demo_spec, tmp_path / "demo.json"))
    assert loaded == demo_spec


def test_id_ignores_descriptive_fields(demo_spec: ScenarioSpec) -> None:
    renamed = demo_spec.model_copy(
        update={"name": "other", "description": "x", "tags": ("a",), "meta": {"by": "lhs"}}
    )
    assert io.scenario_id(renamed) == io.scenario_id(demo_spec)


def test_id_depends_on_semantics(demo_spec: ScenarioSpec) -> None:
    assert io.scenario_id(demo_spec.with_seed(1)) != io.scenario_id(demo_spec)
    other_eval = demo_spec.model_copy(
        update={"evaluation": demo_spec.evaluation.model_copy(update={"acceptance_window": 1500})}
    )
    assert io.scenario_id(other_eval) != io.scenario_id(demo_spec)
    # протокол оценки не влияет на данные, поэтому stream_id тот же
    assert io.stream_id(other_eval) == io.stream_id(demo_spec)


def test_canonical_form_is_stable() -> None:
    """«Золотой» идентификатор: изменение канонической формы требует повысить версию схемы."""
    spec = io.from_dict({"events": [{"position": 10_000, "magnitude": 0.25}]})
    canonical = io.canonical_json(spec)
    assert canonical == json.dumps(json.loads(canonical), sort_keys=True, separators=(",", ":"))
    assert io.scenario_id(spec) == "scn-ddfd54099929"


def test_compact_form_errors() -> None:
    with pytest.raises(NotImplementedError):
        io.from_dict({"extends": "catalog/medium"})
    with pytest.raises(ValueError):
        io.from_dict({"events": [], "drifts": {"schedule": {"count": 1}}})
    with pytest.raises(ValueError):
        io.loads_yaml("- 1\n- 2\n")
    with pytest.raises(ValueError):
        io.save(ScenarioSpec(), "scenario.txt")


def test_json_schema_export(tmp_path: Path) -> None:
    path = io.export_json_schema(tmp_path / "scenario.schema.json")
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert schema["title"] == "scendrift ScenarioSpec"
    assert {"stream", "events", "evaluation", "seed"} <= set(schema["properties"])
    assert "DriftEvent" in schema["$defs"]


def test_committed_schema_is_up_to_date() -> None:
    committed = Path(__file__).resolve().parents[1] / "configs" / "schema" / "scenario.schema.json"
    assert json.loads(committed.read_text(encoding="utf-8")) == io.json_schema()


def test_example_configs_load_and_validate(examples_dir: Path) -> None:
    files = sorted(examples_dir.glob("*.yaml"))
    assert len(files) >= 3
    for path in files:
        spec = io.load(path)
        report = check(spec)
        assert report.ok, f"{path.name}: {report}"

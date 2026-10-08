"""Сериализация сценариев: YAML/JSON, каноническая форма, идентификаторы, JSON Schema.

Ответственный: Будаев К. В.

Идентификатор сценария вычисляется по содержимому. Берётся SHA-256 от
канонического JSON (ключи отсортированы, без пробелов) по смысловым полям
⟨B, E, Ω, s⟩ и версии схемы. Описательные поля (``name``, ``description``,
``tags``, ``meta``) в хэш не входят. Поля, которые для события не имеют
смысла (доля признаков α у prior-события и у возврата, профиль перехода у
внезапного события), перед хэшированием приводятся к значениям по
умолчанию. Поэтому любое смысловое изменение меняет идентификатор,
сценарии с одинаковой цепочкой концептов получают один идентификатор, а
по идентификатору можно кэшировать и связывать результаты.

Наследование шаблонов (``extends``) и композиция из блоков (``blocks``)
реализованы в :mod:`scendrift.formation`. Функция :func:`from_dict`
вызывает их, если описание их использует (импорт ленивый: пакет
``scenario`` не зависит от ``formation`` при загрузке).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from scendrift.evaluation.protocol import EvaluationSpec
from scendrift.scenario.schema import (
    SCHEMA_VERSION,
    DriftPlan,
    ScenarioSpec,
    StreamSpec,
    expand_plan,
)

__all__ = [
    "semantic_dict",
    "canonical_json",
    "scenario_id",
    "stream_id",
    "to_dict",
    "from_dict",
    "dumps_yaml",
    "loads_yaml",
    "save",
    "load",
    "json_schema",
    "export_json_schema",
]

_DESCRIPTIVE = {"name", "description", "tags", "meta"}
_ID_LENGTH = 12


def _normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    """Приводит не влияющие на данные поля события к значениям по умолчанию."""
    event = dict(event)
    if event.get("kind") in (None, "prior"):
        event["affected_share"] = 1.0
    if event.get("form") == "sudden":
        event["shape"] = "linear"
    return event


def semantic_dict(spec: ScenarioSpec, *, include_evaluation: bool = True) -> dict[str, Any]:
    """Смысловые поля сценария в JSON-совместимом виде (с нормализацией событий)."""
    exclude = set(_DESCRIPTIVE)
    if not include_evaluation:
        exclude.add("evaluation")
    data = spec.model_dump(mode="json", exclude=exclude)
    data["events"] = [_normalize_event(e) for e in data["events"]]
    return data


def canonical_json(spec: ScenarioSpec, *, include_evaluation: bool = True) -> str:
    """Каноническая JSON-строка: отсортированные ключи, компактные разделители."""
    return json.dumps(
        semantic_dict(spec, include_evaluation=include_evaluation),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_ID_LENGTH]


def scenario_id(spec: ScenarioSpec) -> str:
    """Идентификатор сценария ``scn-xxxxxxxxxxxx`` (данные и протокол оценки)."""
    return "scn-" + _digest(canonical_json(spec))


def stream_id(spec: ScenarioSpec) -> str:
    """Идентификатор потока ``str-xxxxxxxxxxxx``: только то, что влияет на данные."""
    return "str-" + _digest(canonical_json(spec, include_evaluation=False))


def to_dict(spec: ScenarioSpec) -> dict[str, Any]:
    """Полное представление сценария (включая описательные поля)."""
    return spec.model_dump(mode="json")


def from_dict(data: dict[str, Any], *, base_dir: str | Path | None = None) -> ScenarioSpec:
    """Строит сценарий из словаря в канонической или компактной форме.

    В компактной форме вместо ``events`` задаётся ``drifts`` —
    объект :class:`~scendrift.scenario.schema.DriftPlan`. Он разворачивается
    в явный список событий с помощью
    :func:`~scendrift.scenario.schema.expand_plan`. Описание может
    наследовать шаблон (``extends``) и задавать события блоками (``blocks``).

    Args:
        data: описание сценария.
        base_dir: каталог, относительно которого ищутся файлы-родители ``extends``.

    Raises:
        ValueError: если одновременно заданы ``events`` и ``drifts``.
        TemplateError: при неизвестном шаблоне или цикле наследования.
    """
    data = dict(data)
    if "extends" in data or "blocks" in data:
        from scendrift.formation.compose import expand_blocks
        from scendrift.formation.templates import resolve

        data = expand_blocks(resolve(data, base_dir=base_dir))
    data.setdefault("schema_version", SCHEMA_VERSION)
    if "drifts" in data:
        if "events" in data:
            raise ValueError("нужно задать либо events, либо drifts, но не оба")
        plan = DriftPlan.model_validate(data.pop("drifts"))
        stream = StreamSpec.model_validate(data.get("stream", {}))
        evaluation = EvaluationSpec.model_validate(data.get("evaluation", {}))
        data["events"] = [e.model_dump(mode="json") for e in expand_plan(plan, stream, evaluation)]
    return ScenarioSpec.model_validate(data)


def dumps_yaml(spec: ScenarioSpec) -> str:
    """YAML-представление сценария с идентификатором в комментарии-заголовке."""
    body = yaml.safe_dump(
        to_dict(spec), allow_unicode=True, sort_keys=False, default_flow_style=None
    )
    return f"# id: {scenario_id(spec)}\n# schema_version: {spec.schema_version}\n{body}"


def loads_yaml(text: str, *, base_dir: str | Path | None = None) -> ScenarioSpec:
    """Читает сценарий из YAML-строки (каноническая или компактная форма)."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("YAML-описание сценария должно быть отображением")
    return from_dict(data, base_dir=base_dir)


def save(spec: ScenarioSpec, path: str | Path) -> Path:
    """Сохраняет сценарий в YAML (.yaml/.yml) или JSON (.json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in {".yaml", ".yml"}:
        path.write_text(dumps_yaml(spec), encoding="utf-8")
    elif path.suffix == ".json":
        path.write_text(
            json.dumps(to_dict(spec), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        raise ValueError(f"неизвестный формат файла: {path.suffix}")
    return path


def load(path: str | Path) -> ScenarioSpec:
    """Загружает сценарий из YAML или JSON."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        return loads_yaml(text, base_dir=path.parent)
    if path.suffix == ".json":
        return from_dict(json.loads(text), base_dir=path.parent)
    raise ValueError(f"неизвестный формат файла: {path.suffix}")


def json_schema() -> dict[str, Any]:
    """JSON Schema канонической формы сценария (генерируется из pydantic)."""
    schema = ScenarioSpec.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "scendrift ScenarioSpec"
    schema["description"] = (
        f"Сценарий концептуального дрейфа S = <B, E, Omega, s>, версия схемы {SCHEMA_VERSION}"
    )
    return schema


def export_json_schema(path: str | Path) -> Path:
    """Сохраняет JSON Schema в файл."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path

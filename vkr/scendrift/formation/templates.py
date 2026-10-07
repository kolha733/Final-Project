"""Шаблоны сценариев: наследование ``extends`` и глубокое слияние.

Ответственный: Будаев К. В.

Шаблон — неполное описание сценария (словарь в канонической или компактной
форме). Описание может ссылаться на родительский шаблон полем ``extends``
и уточнять его. Результат вычисляется так:

    resolve(T) = merge(resolve(parent(T)), T без extends).

Правила слияния :func:`deep_merge`:

1. словари сливаются рекурсивно;
2. остальные значения (числа, строки, списки) заменяются целиком;
3. значение ``null`` удаляет ключ родителя (например, ``magnitude: null``
   при переходе к некалиброванному семейству);
4. ``events``, ``drifts`` и ``blocks`` — три взаимоисключающих способа
   задать события. Если потомок задаёт один из них, остальные у родителя
   отбрасываются.

Ссылки ``extends``:

* ``catalog/<имя>`` — встроенный каталог (:func:`register_template`,
  профили easy/medium/hard регистрирует :mod:`scendrift.formation.difficulty`);
* иначе — путь к файлу YAML/JSON относительно файла-потомка
  (расширение можно не указывать).

Допустим и список родителей: они сливаются слева направо. Циклы
наследования обнаруживаются.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "EVENT_FORMS",
    "deep_merge",
    "register_template",
    "get_template",
    "catalog_names",
    "resolve",
    "TemplateError",
]

#: Взаимоисключающие способы задать события.
EVENT_FORMS = ("events", "drifts", "blocks")

_CATALOG: dict[str, dict[str, Any]] = {}
_PREFIX = "catalog/"


class TemplateError(ValueError):
    """Ошибка наследования: неизвестный шаблон, цикл, неверный формат."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Глубокое слияние описаний сценария (правила — в описании модуля).

    Аргументы не изменяются.
    """
    out = copy.deepcopy(dict(base))
    if any(key in override for key in EVENT_FORMS):
        for key in EVENT_FORMS:
            if key not in override:
                out.pop(key, None)
    for key, value in override.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def register_template(name: str, data: Mapping[str, Any]) -> None:
    """Добавляет шаблон во встроенный каталог (ссылка ``catalog/<name>``)."""
    if "extends" in data and _PREFIX + name in _as_list(data["extends"]):
        raise TemplateError(f"шаблон {name!r} наследует сам себя")
    _CATALOG[name] = copy.deepcopy(dict(data))


def get_template(name: str) -> dict[str, Any]:
    """Шаблон каталога по имени (без разрешения ``extends``).

    Raises:
        TemplateError: если шаблона нет.
    """
    _ensure_catalog()
    if name not in _CATALOG:
        raise TemplateError(
            f"в каталоге нет шаблона {name!r}; доступны: {', '.join(sorted(_CATALOG))}"
        )
    return copy.deepcopy(_CATALOG[name])


def catalog_names() -> list[str]:
    """Имена шаблонов встроенного каталога."""
    _ensure_catalog()
    return sorted(_CATALOG)


def _ensure_catalog() -> None:
    """Профили каталога регистрируются при импорте модуля шкалы сложности."""
    import scendrift.formation.difficulty  # noqa: F401


def _as_list(ref: str | Sequence[str]) -> list[str]:
    if isinstance(ref, str):
        return [ref]
    if isinstance(ref, Sequence) and all(isinstance(r, str) for r in ref):
        return list(ref)
    raise TemplateError("extends: нужна строка или список строк")


def _load_file(ref: str, base_dir: Path | None) -> tuple[dict[str, Any], Path, str]:
    path = Path(ref)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    candidates = (
        [path] if path.suffix else [path.with_suffix(s) for s in (".yaml", ".yml", ".json")]
    )
    for cand in candidates:
        if cand.exists():
            text = cand.read_text(encoding="utf-8")
            data = json.loads(text) if cand.suffix == ".json" else yaml.safe_load(text)
            if not isinstance(data, dict):
                raise TemplateError(f"{cand}: шаблон должен быть отображением")
            return data, cand.parent, str(cand.resolve())
    raise TemplateError(f"шаблон {ref!r} не найден (искали: {', '.join(map(str, candidates))})")


def resolve(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    _stack: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Разрешает наследование: возвращает описание без ``extends``.

    Блоки ``blocks`` (композиция) не разворачиваются: это делает
    :func:`scendrift.formation.compose.expand_blocks`.

    Args:
        data: описание сценария или шаблона.
        base_dir: каталог, относительно которого ищутся файлы-родители.

    Raises:
        TemplateError: при неизвестном шаблоне или цикле наследования.
    """
    data = dict(data)
    if "extends" not in data:
        return copy.deepcopy(data)
    refs = _as_list(data.pop("extends"))
    base_path = Path(base_dir) if base_dir is not None else None
    merged: dict[str, Any] = {}
    for ref in refs:
        if ref.startswith(_PREFIX):
            key = ref
            parent = get_template(ref[len(_PREFIX) :])
            parent_dir = base_path
        else:
            parent, parent_dir, key = _load_file(ref, base_path)
        if key in _stack:
            chain = " → ".join([*_stack, key])
            raise TemplateError(f"цикл наследования: {chain}")
        merged = deep_merge(merged, resolve(parent, base_dir=parent_dir, _stack=(*_stack, key)))
    return deep_merge(merged, data)

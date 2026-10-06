"""Ноутбук ВКР синхронизирован с модулями пакета и списком литературы.

Тест есть только в репозитории: в самом ноутбуке его нет, потому что
ноутбук не содержит сборщика. Без сборщика или ноутбука тест пропускается.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tools" / "build_notebook.py"
NOTEBOOK = ROOT / "VKR_Budaev_Garifzyanov.ipynb"

pytestmark = pytest.mark.skipif(
    not (BUILDER.exists() and NOTEBOOK.exists()), reason="нет сборщика или ноутбука"
)


def _builder():
    spec = importlib.util.spec_from_file_location("build_notebook", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_writefile_cells_match_modules() -> None:
    assert _builder().check(NOTEBOOK) == []


def test_citations_resolve_and_bibliography_is_rendered() -> None:
    builder = _builder()
    nb = builder.build()
    refs = builder.load_references()
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "[@" not in text  # все ключи заменены номерами
    assert "<!-- bibliography -->" not in text
    assert all("gost" in entry for entry in refs.values())


def test_unknown_citation_key_fails() -> None:
    builder = _builder()
    cells = builder.parse_source("Текст со ссылкой [@no_such_key].")
    with pytest.raises(KeyError):
        builder.number_citations(cells, builder.load_references())

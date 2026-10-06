"""Сборка ноутбука ВКР из исходников: markdown-разделов и модулей пакета.

Ответственные: Будаев К. В., Гарифзянов Т. Р.

Источник истины — модули ``scendrift/*.py`` и тексты разделов
``notebook/*.md``. Ноутбук собирается из них, поэтому код в ноутбуке и в
пакете не расходится. Ноутбук при этом самодостаточен: ячейки
``%%writefile`` заново создают пакет в чистом окружении (например, в Colab).

Правила разметки исходников ``notebook/*.md``:

* блок ```python … ``` превращается в ячейку кода;
* блок ```writefile путь``` (пустой) превращается в ячейку
  ``%%writefile путь`` с содержимым файла;
* строка ``<!-- cell -->`` разделяет markdown-ячейки;
* ссылка ``[@key]`` или ``[@key1; @key2]`` заменяется номером источника
  по порядку первого упоминания; в место ``<!-- bibliography -->``
  вставляется нумерованный список литературы из ``notebook/references.yaml``
  (тот же файл использует генератор пояснительной записки);
* остальной текст (включая блоки ```yaml, ```text и т. п.) остаётся
  markdown-ячейками.

Использование::

    python tools/build_notebook.py            # собрать ноутбук
    python tools/build_notebook.py --execute  # собрать и выполнить сверху вниз
    python tools/build_notebook.py --check    # сверить ноутбук с модулями
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat
import yaml
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "notebook"
OUTPUT = ROOT / "VKR_Budaev_Garifzyanov.ipynb"

REFERENCES = SOURCES / "references.yaml"

_FENCE = re.compile(r"^```(\S*)\s*(.*)$")
_SPLIT = "<!-- cell -->"
_CITE = re.compile(r"\[(@[\w-]+(?:\s*;\s*@[\w-]+)*)\]")
_BIBLIOGRAPHY = "<!-- bibliography -->"


def load_references(path: Path = REFERENCES) -> dict[str, dict]:
    """Читает список литературы: ключ → запись (поле ``gost`` обязательно)."""
    if not path.exists():
        return {}
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    refs = {}
    for entry in entries:
        if entry["key"] in refs:
            raise ValueError(f"повтор ключа источника {entry['key']!r}")
        refs[entry["key"]] = entry
    return refs


def number_citations(cells: list, refs: dict[str, dict]) -> list[str]:
    """Заменяет ``[@key]`` номерами по порядку первого упоминания.

    Returns:
        Ключи источников в порядке нумерации.
    """
    order: list[str] = []

    def replace(match: re.Match) -> str:
        numbers = []
        for key in (k.strip().lstrip("@") for k in match.group(1).split(";")):
            if key not in refs:
                raise KeyError(f"неизвестный источник [@{key}]")
            if key not in order:
                order.append(key)
            numbers.append(str(order.index(key) + 1))
        return "[" + "; ".join(numbers) + "]"

    for cell in cells:
        if cell.cell_type == "markdown":
            cell.source = _CITE.sub(replace, cell.source)
    for cell in cells:
        if cell.cell_type == "markdown" and _BIBLIOGRAPHY in cell.source:
            lines = []
            for i, key in enumerate(order, start=1):
                mark = " [ПРОВЕРИТЬ]" if refs[key].get("status") == "check" else ""
                lines.append(f"{i}. {refs[key]['gost']}{mark}")
            cell.source = cell.source.replace(_BIBLIOGRAPHY, "\n".join(lines))
    return order


def _flush_markdown(buffer: list[str], cells: list) -> None:
    text = "\n".join(buffer).strip("\n")
    if text.strip():
        cells.append(new_markdown_cell(text))
    buffer.clear()


def parse_source(text: str) -> list:
    """Разбирает markdown-исходник раздела на ячейки ноутбука."""
    cells: list = []
    markdown: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        match = _FENCE.match(line)
        if match and match.group(1) in {"python", "writefile"}:
            lang, arg = match.group(1), match.group(2).strip()
            body: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "```":
                body.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ValueError(f"незакрытый блок ```{lang}")
            _flush_markdown(markdown, cells)
            if lang == "python":
                cells.append(new_code_cell("\n".join(body).strip("\n")))
            else:
                path = ROOT / arg
                content = path.read_text(encoding="utf-8").rstrip("\n")
                cell = new_code_cell(f"%%writefile {arg}\n{content}")
                cell.metadata["scendrift_writefile"] = arg
                cell.metadata["jupyter"] = {"source_hidden": False}
                cells.append(cell)
            i += 1
            continue
        if match and line.strip() != "```":
            # прочие блоки кода остаются в markdown (копируем до закрывающей ```)
            markdown.append(line)
            i += 1
            while i < len(lines) and lines[i].strip() != "```":
                markdown.append(lines[i])
                i += 1
            if i < len(lines):
                markdown.append(lines[i])
            i += 1
            continue
        if line.strip() == _SPLIT:
            _flush_markdown(markdown, cells)
        else:
            markdown.append(line)
        i += 1
    _flush_markdown(markdown, cells)
    return cells


def build() -> nbformat.NotebookNode:
    """Собирает ноутбук из всех разделов ``notebook/*.md`` по порядку имён."""
    cells: list = []
    for path in sorted(SOURCES.glob("*.md")):
        cells.extend(parse_source(path.read_text(encoding="utf-8")))
    number_citations(cells, load_references())
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["title"] = (
        "Разработка фреймворка генерации бенчмарков систем выявления концептуального "
        "дрейфа с автоматизированным формированием и параметризацией сценариев"
    )
    nb.metadata["authors"] = [{"name": "Будаев К. В."}, {"name": "Гарифзянов Т. Р."}]
    return nb


def execute(nb: nbformat.NotebookNode, timeout: int = 3600) -> nbformat.NotebookNode:
    """Выполняет ноутбук сверху вниз в каталоге проекта."""
    from nbclient import NotebookClient

    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    return nb


def check(path: Path = OUTPUT) -> list[str]:
    """Сверяет ячейки ``%%writefile`` ноутбука с файлами на диске.

    Returns:
        Список путей, содержимое которых расходится (пустой — всё совпадает).
    """
    nb = nbformat.read(path, as_version=4)
    stale = []
    for cell in nb.cells:
        target = cell.get("metadata", {}).get("scendrift_writefile")
        if not target:
            continue
        expected = (ROOT / target).read_text(encoding="utf-8").rstrip("\n")
        body = cell.source.split("\n", 1)[1] if "\n" in cell.source else ""
        if body != expected:
            stale.append(target)
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--execute", action="store_true", help="выполнить ноутбук")
    parser.add_argument("--check", action="store_true", help="сверить с модулями")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.check:
        stale = check(args.output)
        for target in stale:
            print(f"расходится: {target}")
        return 1 if stale else 0
    nb = build()
    if args.execute:
        nb = execute(nb)
    nbformat.write(nb, args.output)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"{args.output.name}: {len(nb.cells)} ячеек, из них кода {n_code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Сборка ноутбука ВКР из исходников: markdown-разделов и модулей пакета.

Ответственные: Будаев К. В., Гарифзянов Т. Р.

Источник истины — модули ``scendrift/*.py`` и тексты разделов
``notebook/*.md``. Ноутбук собирается из них, поэтому код в ноутбуке и в
пакете не расходится. При этом ноутбук самодостаточен: ячейки
``%%writefile`` заново создают пакет в чистом окружении (например, в Colab).

Правила разметки исходников ``notebook/*.md``:

* блок ```python … ``` превращается в ячейку кода;
* блок ```writefile путь``` (пустой) превращается в ячейку
  ``%%writefile путь`` с содержимым файла;
* строка ``<!-- cell -->`` разделяет markdown-ячейки;
* ссылка ``[@key]`` или ``[@key1; @key2]`` заменяется номером источника
  по порядку первого упоминания. В место ``<!-- bibliography -->``
  вставляется нумерованный список литературы из ``notebook/references.yaml``
  (оформление по ГОСТ Р 7.0.100–2018 — ``tools/gost.py``);
* ``<!-- table:constraints -->`` и ``<!-- table:space -->`` заменяются
  таблицами, построенными из кода (каталог ограничений и пространство
  параметров), поэтому текст и код не расходятся;
* остальной текст, включая блоки ```yaml, ```text и др. (в том числе с
  длинными ограждениями из четырёх и более обратных апострофов), остаётся
  markdown-ячейками.

Использование::

    python tools/build_notebook.py            # собрать ноутбук
    python tools/build_notebook.py --execute  # собрать и выполнить сверху вниз
    python tools/build_notebook.py --check    # сверить ноутбук с исходниками
"""

from __future__ import annotations

import argparse
import importlib.util
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

_FENCE = re.compile(r"^(`{3,})(\S*)\s*(.*)$")
_SPLIT = "<!-- cell -->"
_CITE = re.compile(r"\[(@[\w-]+(?:\s*;\s*@[\w-]+)*)\]")
_BIBLIOGRAPHY = "<!-- bibliography -->"
_TABLE = re.compile(r"<!-- table:(\w+) -->")


def _gost():
    spec = importlib.util.spec_from_file_location("gost", ROOT / "tools" / "gost.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_references(path: Path = REFERENCES) -> dict[str, dict]:
    """Читает список литературы: ключ → запись с готовой строкой ``gost``."""
    if not path.exists():
        return {}
    gost = _gost()
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    refs = {}
    for entry in entries:
        if entry["key"] in refs:
            raise ValueError(f"повтор ключа источника {entry['key']!r}")
        refs[entry["key"]] = {**entry, "gost": gost.render(entry)}
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


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    def esc(text: str) -> str:
        return str(text).replace("|", "\\|")

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render_table(name: str) -> str:
    """Таблица из кода фреймворка для вставки в текст раздела."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if name == "constraints":
        from scendrift.scenario.report import CONSTRAINTS

        rows = [[code, scope, text] for code, (scope, text) in CONSTRAINTS.items()]
        return _md_table(["Код", "Область проверки", "Ограничение"], rows)
    if name == "space":
        from scendrift.scenario.space import DEFAULT_SPACE

        rows = [
            [
                r["смысл"],
                r["обозначение"],
                r["часть"],
                r["домен"],
                r["по умолчанию"],
                r["активен, если"],
            ]
            for r in DEFAULT_SPACE.table()
        ]
        header = ["Параметр", "Обозн.", "Часть", "Домен", "По умолч.", "Активен, если"]
        return _md_table(header, rows)
    raise KeyError(f"неизвестная таблица {name!r}")


def _flush_markdown(buffer: list[str], cells: list) -> None:
    text = "\n".join(buffer).strip("\n")
    if text.strip():
        text = _TABLE.sub(lambda m: render_table(m.group(1)), text)
        cells.append(new_markdown_cell(text))
    buffer.clear()


def parse_source(text: str) -> list:
    """Разбирает markdown-исходник раздела на ячейки ноутбука.

    Raises:
        ValueError: если блок кода любого языка не закрыт.
    """
    cells: list = []
    markdown: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        match = _FENCE.match(line)
        if not match:
            if line.strip() == _SPLIT:
                _flush_markdown(markdown, cells)
            else:
                markdown.append(line)
            i += 1
            continue
        fence, lang, arg = match.group(1), match.group(2), match.group(3).strip()
        body: list[str] = []
        j = i + 1
        while j < len(lines) and not (
            lines[j].strip().startswith(fence) and set(lines[j].strip()) == {"`"}
        ):
            body.append(lines[j])
            j += 1
        if j >= len(lines):
            raise ValueError(f"незакрытый блок {fence}{lang} (строка {i + 1})")
        if lang == "python" and len(fence) == 3:
            _flush_markdown(markdown, cells)
            cells.append(new_code_cell("\n".join(body).strip("\n")))
        elif lang == "writefile" and len(fence) == 3:
            _flush_markdown(markdown, cells)
            content = (ROOT / arg).read_text(encoding="utf-8").rstrip("\n")
            cell = new_code_cell(f"%%writefile {arg}\n{content}")
            cell.metadata["scendrift_writefile"] = arg
            cells.append(cell)
        else:
            markdown.extend(lines[i : j + 1])
        i = j + 1
    _flush_markdown(markdown, cells)
    return cells


def build() -> nbformat.NotebookNode:
    """Собирает ноутбук из всех разделов ``notebook/*.md`` по порядку имён."""
    cells: list = []
    for path in sorted(SOURCES.glob("*.md")):
        cells.extend(parse_source(path.read_text(encoding="utf-8")))
    number_citations(cells, load_references())
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
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

    client = NotebookClient(
        nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    )
    client.execute()
    return nb


def check(path: Path = OUTPUT) -> list[str]:
    """Сверяет сохранённый ноутбук со свежей сборкой из исходников.

    Сравниваются типы и тексты ячеек (выводы не учитываются). Так
    обнаруживаются и изменённый модуль, и правка текста, и удалённая или
    добавленная ячейка.

    Returns:
        Список расхождений (пустой — ноутбук актуален).
    """
    if not path.exists():
        return [f"нет файла {path.name}"]
    try:
        fresh = build()
    except (OSError, KeyError, ValueError) as error:
        return [f"сборка невозможна: {error}"]
    saved = nbformat.read(path, as_version=4)
    problems = []
    if len(saved.cells) != len(fresh.cells):
        problems.append(f"число ячеек: {len(saved.cells)} ≠ {len(fresh.cells)} (сборка)")
    for i, (a, b) in enumerate(zip(saved.cells, fresh.cells, strict=False)):
        if a.cell_type != b.cell_type or a.source != b.source:
            target = b.get("metadata", {}).get("scendrift_writefile")
            problems.append(f"ячейка {i}: {target or a.cell_type} расходится со сборкой")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--execute", action="store_true", help="выполнить ноутбук")
    parser.add_argument("--check", action="store_true", help="сверить с исходниками")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.check:
        problems = check(args.output)
        for problem in problems:
            print(problem)
        return 1 if problems else 0
    nb = build()
    if args.execute:
        nb = execute(nb)
    nbformat.write(nb, args.output)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"{args.output.name}: {len(nb.cells)} ячеек, из них кода {n_code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

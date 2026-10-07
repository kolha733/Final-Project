"""Сборка пояснительной записки (.docx) по ГОСТ 7.32-2017.

Ответственные: Будаев К. В., Гарифзянов Т. Р.

Источники:

* ``docs/zapiska/text/*.md`` — текст записки (Markdown, формулы в LaTeX);
* ``docs/zapiska/facts.py`` — числа и таблицы, вычисленные по результатам
  экспериментов (``results/*.csv``) и по коду пакета: записка не расходится
  с ноутбуком и после прогона FULL пересобирается без ручной правки чисел;
* ``notebook/references.yaml`` — список литературы; строки по
  ГОСТ Р 7.0.100-2018 формирует ``tools/gost.py``;
* ``docs/figures/zapiska/*.png`` — рисунки без общего заголовка;
* ``docs/zapiska/format_config.py`` — все параметры оформления.

Разметка текста сверх Markdown:

* ``{{ имя }}`` — значение из ``facts.py``;
* ``<!-- table:имя -->`` — таблица из ``facts.py``;
* ``[@ключ]``, ``[@a; @b]`` — ссылки на источники, нумеруются по первому упоминанию;
* ``![Название](путь){#fig:id}`` (ширина по умолчанию — ``FIGURE_WIDTH_CM``, иначе
  ``{#fig:id width=12}`` в сантиметрах), ``Table: Название {#tbl:id}``, ``$$ … $$ {#eq:id}`` —
  рисунки, таблицы и формулы, нумеруются в пределах главы; ссылки ``@fig:id``,
  ``@tbl:id``, ``@eq:id`` заменяются номерами;
* ``<!-- bibliography -->`` — место списка источников;
* заголовок ``# ПРИЛОЖЕНИЕ А | (справочное) | Название`` — приложение.

Сборка идёт в два прохода. Первый проход рендерится в PDF (LibreOffice), по
нему определяются номера страниц заголовков (для содержания) и объём
записки (для реферата). Второй проход подставляет эти значения.

Запуск: ``python docs/zapiska/build_docx.py`` (из каталога ``vkr``).
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import pypandoc
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import format_config as F  # noqa: E402, N812

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# Препроцессор Markdown
# ---------------------------------------------------------------------------


@dataclass
class Prepared:
    """Результат препроцессора: текст для pandoc и счётчики."""

    markdown: str
    figures: int = 0
    tables: int = 0
    equations: int = 0
    sources: int = 0
    appendices: int = 0
    headings: list[tuple[int, str]] = field(default_factory=list)


def _fmt_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}".replace(".", ",")
    return str(value)


def substitute_facts(text: str, facts: dict[str, object], tables: dict[str, str]) -> str:
    """Подставляет ``{{ имя }}`` и ``<!-- table:имя -->``; неизвестное имя — ошибка."""

    def fact(match: re.Match) -> str:
        key = match.group(1)
        if key not in facts:
            raise KeyError(f"нет значения {{{{ {key} }}}} в facts.py")
        return _fmt_value(facts[key])

    def table(match: re.Match) -> str:
        key = match.group(1)
        if key not in tables:
            raise KeyError(f"нет таблицы {key!r} в facts.py")
        return tables[key]

    text = re.sub(r"<!--\s*table:(\w+)\s*-->", table, text)
    return re.sub(r"\{\{\s*([\w.]+)\s*\}\}", fact, text)


def number_citations(text: str, known: set[str]) -> tuple[str, list[str]]:
    """``[@a; @b]`` → ``[1, 2]`` по порядку первого упоминания."""
    order: OrderedDict[str, int] = OrderedDict()

    def replace(match: re.Match) -> str:
        keys = [k.strip().lstrip("@") for k in match.group(1).split(";")]
        numbers = []
        for key in keys:
            if key not in known:
                raise KeyError(f"источник {key!r} отсутствует в references.yaml")
            order.setdefault(key, len(order) + 1)
            numbers.append(order[key])
        return "[" + "; ".join(str(n) for n in sorted(numbers)) + "]"

    text = re.sub(r"\[(@[\w\-]+(?:\s*;\s*@[\w\-]+)*)\]", replace, text)
    return text, list(order)


_CHAPTER = re.compile(r"^# (\d+) ", re.M)


def number_objects(text: str) -> tuple[str, dict[str, int]]:
    """Нумерует рисунки, таблицы и формулы по главам и подставляет ссылки."""
    labels: dict[str, str] = {}
    counts = {"fig": 0, "tbl": 0, "eq": 0}
    out_lines = []
    chapter = ""
    per_chapter = {"fig": 0, "tbl": 0, "eq": 0}

    def next_number(kind: str) -> str:
        counts[kind] += 1
        per_chapter[kind] += 1
        if F.NUMBER_BY_CHAPTER and chapter:
            return f"{chapter}.{per_chapter[kind]}"
        return str(counts[kind])

    for line in text.splitlines():
        head = re.match(r"^# (\d+) ", line)
        appendix = re.match(rf"^# {F.APPENDIX_PREFIX} (\w)", line)
        if head or appendix:
            chapter = head.group(1) if head else appendix.group(1)
            per_chapter = {"fig": 0, "tbl": 0, "eq": 0}
        elif re.match(r"^# ", line):
            chapter = ""
        fig = re.match(r"^!\[(.+)\]\(([^)]+)\)\{#fig:([\w\-]+)(?:\s+width=([\d.]+))?\}\s*$", line)
        tbl = re.match(r"^Table:\s*(.+?)\s*\{#tbl:([\w\-]+)\}\s*$", line)
        eq = re.match(r"^\$\$(.+)\$\$\s*\{#eq:([\w\-]+)\}\s*$", line)
        if fig:
            number = next_number("fig")
            labels[f"fig:{fig.group(3)}"] = number
            caption = F.FIGURE_CAPTION.format(number=number, title=fig.group(1))
            width = fig.group(4) or F.FIGURE_WIDTH_CM
            line = f"![{caption}]({fig.group(2)}){{width={width}cm}}"
        elif tbl:
            number = next_number("tbl")
            labels[f"tbl:{tbl.group(2)}"] = number
            line = "Table: " + F.TABLE_CAPTION.format(number=number, title=tbl.group(1))
        elif eq:
            number = next_number("eq")
            labels[f"eq:{eq.group(2)}"] = number
            line = f"$${eq.group(1)}$$\n\nEQNUM{F.EQUATION_NUMBER.format(number=number)}"
        out_lines.append(line)
    text = "\n".join(out_lines)

    def ref(match: re.Match) -> str:
        key = f"{match.group(1)}:{match.group(2)}"
        if key not in labels:
            raise KeyError(f"ссылка на неизвестный объект @{key}")
        number = labels[key]
        return f"({number})" if match.group(1) == "eq" else number

    text = re.sub(r"@(fig|tbl|eq):([\w\-]+)", ref, text)
    return text, counts


_MATH = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$")


def office_math(text: str) -> str:
    r"""Приводит формулы к виду, который одинаково отображают Word и LibreOffice.

    При импорте OMML LibreOffice показывает «|» как «∨», а «∣» (``\\mid``) — как
    «/». Текстовый символ внутри формулы (``\\text{|}``) отображается верно
    везде. В строках pipe-таблиц «|» разделяет ячейки, поэтому там используется
    внешне такой же символ U+01C0. Одиночная «|» в формуле — ошибка: модуль
    записывается как ``\\left| … \\right|``. Непарные скобки (полуинтервал
    ``[a, b)``) LibreOffice тоже искажает — их записывают через ``\\left[ … \\right)``.
    Скобки целой части (``\\lfloor``, ``\\rfloor``) заменяются текстовыми символами.
    """
    out = []
    for line in text.splitlines():
        bar = "\\text{ǀ}" if line.lstrip().startswith("|") else "\\text{|}"
        line = re.sub(r"\\mid(?![A-Za-z])", lambda _m, b=bar: b, line)
        for cmd, char in (("lfloor", "⌊"), ("rfloor", "⌋"), ("lceil", "⌈"), ("rceil", "⌉")):
            line = re.sub(
                rf"(\\left|\\right)?\\{cmd}(?![A-Za-z])", lambda _m, c=char: f"\\text{{{c}}}", line
            )
        for formula in _MATH.findall(line):
            bare = re.sub(r"\\(left|right)\||\\text\{\|\}|\\\|", "", formula)
            if "|" in bare and not line.lstrip().startswith("|"):
                raise ValueError(
                    f"одиночная «|» в формуле {formula!r}: используйте \\left| … \\right| или \\mid"
                )
            plain = re.sub(r"\\(left|right)[\[\]()|.]", "", formula)
            if plain.count("[") != plain.count("]") or plain.count("(") != plain.count(")"):
                raise ValueError(
                    f"непарные скобки в формуле {formula!r}: полуинтервал "
                    "записывается как \\left[ … \\right)"
                )
        out.append(line)
    return "\n".join(out)


def prepare(page_info: dict[str, object] | None = None) -> tuple[Prepared, list[str], dict]:
    """Препроцессор: факты, ссылки на источники, нумерация объектов, список литературы."""
    facts_module = _load_module("zapiska_facts", HERE / "facts.py")
    facts = facts_module.collect()
    facts.update(page_info or {"doc.pages": 0})
    tables = facts_module.tables()
    sources = sorted(HERE.joinpath("text").glob("*.md"))
    text = "\n\n".join(p.read_text(encoding="utf-8") for p in sources)

    build_nb = _load_module("build_notebook", ROOT / "tools" / "build_notebook.py")
    refs = build_nb.load_references()
    text, order = number_citations(text, set(refs))
    text, counts = number_objects(text)
    facts.update(
        {
            "doc.figures": counts["fig"],
            "doc.tables": counts["tbl"],
            "doc.sources": len(order),
            "doc.appendices": len(re.findall(rf"^# {F.APPENDIX_PREFIX} ", text, re.M)),
        }
    )
    lines = []
    for i, key in enumerate(order, 1):
        mark = " [ПРОВЕРИТЬ]" if refs[key].get("status") == "check" else ""
        lines.append(f"{i}\\. {refs[key]['gost']}{mark}")
    text = text.replace("<!-- bibliography -->", "\n\n".join(lines))
    text = substitute_facts(text, facts, tables)
    text = office_math(text)
    prepared = Prepared(
        markdown=text,
        figures=counts["fig"],
        tables=counts["tbl"],
        equations=counts["eq"],
        sources=len(order),
        appendices=facts["doc.appendices"],  # type: ignore[arg-type]
    )
    return prepared, order, facts


# ---------------------------------------------------------------------------
# Стили (reference.docx для pandoc)
# ---------------------------------------------------------------------------


def _set_font(
    rpr_owner,
    name: str,
    size: float | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
) -> None:
    font = rpr_owner.font
    font.name = name
    if size is not None:
        font.size = Pt(size)
    if bold is not None:
        font.bold = bold
    if italic is not None:
        font.italic = italic
    font.color.rgb = RGBColor(0, 0, 0)
    rpr = rpr_owner.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), name)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        fonts.attrib.pop(qn(attr), None)


def _paragraph_format(
    style, *, align=None, indent_cm=None, spacing=None, before=0.0, after=0.0, keep_next=False
) -> None:
    pf = style.paragraph_format
    if align is not None:
        pf.alignment = align
    pf.first_line_indent = Cm(indent_cm) if indent_cm is not None else Cm(0)
    pf.left_indent = Cm(0)
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = spacing if spacing is not None else F.LINE_SPACING
    pf.keep_with_next = keep_next
    pf.widow_control = True


def make_reference_docx(path: Path) -> Path:
    """reference.docx: стили pandoc, приведённые к параметрам format_config."""
    with tempfile.TemporaryDirectory() as tmp:
        default = Path(tmp) / "default.docx"
        subprocess.run(
            [
                pypandoc.get_pandoc_path(),
                "-o",
                str(default),
                "--print-default-data-file",
                "reference.docx",
            ],
            check=True,
        )
        doc = Document(default)
    justify = WD_ALIGN_PARAGRAPH.JUSTIFY if F.JUSTIFY else WD_ALIGN_PARAGRAPH.LEFT
    styles = doc.styles
    body_styles = ["Normal", "Body Text", "First Paragraph", "Compact", "Block Text"]
    for name in body_styles:
        if name in styles:
            st = styles[name]
            _set_font(st, F.FONT_NAME, F.FONT_SIZE_PT, bold=False, italic=False)
            _paragraph_format(st, align=justify, indent_cm=F.FIRST_LINE_INDENT_CM)
    for level in (1, 2, 3):
        st = styles[f"Heading {level}"]
        _set_font(st, F.FONT_NAME, F.HEADING_FONT_SIZE_PT, bold=F.HEADING_BOLD, italic=False)
        _paragraph_format(
            st,
            align=WD_ALIGN_PARAGRAPH.LEFT,
            indent_cm=F.FIRST_LINE_INDENT_CM,
            before=F.HEADING_SPACE_BEFORE_PT,
            after=F.HEADING_SPACE_AFTER_PT,
            keep_next=True,
        )
    for name, align in (
        ("Image Caption", WD_ALIGN_PARAGRAPH.CENTER),
        ("Table Caption", WD_ALIGN_PARAGRAPH.LEFT),
        ("Captioned Figure", WD_ALIGN_PARAGRAPH.CENTER),
        ("Figure", WD_ALIGN_PARAGRAPH.CENTER),
    ):
        if name in styles:
            st = styles[name]
            _set_font(st, F.FONT_NAME, F.CAPTION_FONT_SIZE_PT, bold=False, italic=False)
            _paragraph_format(
                st, align=align, indent_cm=0, before=6, after=6, keep_next=name != "Image Caption"
            )
    # Без подсветки pandoc ссылается на стили листингов, но не создаёт их; если
    # стиля нет, LibreOffice игнорирует и прямое форматирование абзаца.
    if "Source Code" not in styles:
        styles.add_style("Source Code", WD_STYLE_TYPE.PARAGRAPH).base_style = styles["Normal"]
    if "Verbatim Char" not in styles:
        styles.add_style("Verbatim Char", WD_STYLE_TYPE.CHARACTER)
    st = styles["Source Code"]
    _set_font(st, F.CODE_FONT_NAME, F.CODE_FONT_SIZE_PT)
    _paragraph_format(st, align=WD_ALIGN_PARAGRAPH.LEFT, indent_cm=0, spacing=1.0)
    _set_font(styles["Verbatim Char"], F.CODE_FONT_NAME, F.CODE_FONT_SIZE_PT)
    for name in ("Hyperlink",):
        if name in styles:
            styles[name].font.color.rgb = RGBColor(0, 0, 0)
            styles[name].font.underline = False
    # шрифт по умолчанию документа
    rpr_default = doc.styles.element.find(qn("w:docDefaults")).find(qn("w:rPrDefault"))
    rpr = rpr_default.find(qn("w:rPr"))
    fonts = rpr.find(qn("w:rFonts"))
    for attr in list(fonts.attrib):
        del fonts.attrib[attr]
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), F.FONT_NAME)
    _page_setup(doc)
    doc.save(path)
    return path


def _page_setup(doc) -> None:
    for section in doc.sections:
        section.page_width = Mm(F.PAGE_WIDTH_MM)
        section.page_height = Mm(F.PAGE_HEIGHT_MM)
        section.left_margin = Mm(F.MARGIN_LEFT_MM)
        section.right_margin = Mm(F.MARGIN_RIGHT_MM)
        section.top_margin = Mm(F.MARGIN_TOP_MM)
        section.bottom_margin = Mm(F.MARGIN_BOTTOM_MM)
        section.footer_distance = Mm(F.FOOTER_DISTANCE_MM)
        section.header_distance = Mm(F.FOOTER_DISTANCE_MM)
        section.gutter = Mm(0)


# ---------------------------------------------------------------------------
# Постобработка
# ---------------------------------------------------------------------------


# Порядок дочерних элементов по схеме OOXML (ECMA-376): вставка не на своё
# место делает документ невалидным, и LibreOffice может его не открыть.
_ORDER = {
    "pPr": [
        "pStyle",
        "keepNext",
        "keepLines",
        "pageBreakBefore",
        "framePr",
        "widowControl",
        "numPr",
        "suppressLineNumbers",
        "pBdr",
        "shd",
        "tabs",
        "suppressAutoHyphens",
        "kinsoku",
        "wordWrap",
        "overflowPunct",
        "topLinePunct",
        "autoSpaceDE",
        "autoSpaceDN",
        "bidi",
        "adjustRightInd",
        "snapToGrid",
        "spacing",
        "ind",
        "contextualSpacing",
        "mirrorIndents",
        "suppressOverlap",
        "jc",
        "textDirection",
        "textAlignment",
        "textboxTightWrap",
        "outlineLvl",
        "divId",
        "cnfStyle",
        "rPr",
        "sectPr",
        "pPrChange",
    ],
    "tblPr": [
        "tblStyle",
        "tblpPr",
        "tblOverlap",
        "bidiVisual",
        "tblStyleRowBandSize",
        "tblStyleColBandSize",
        "tblW",
        "jc",
        "tblCellSpacing",
        "tblInd",
        "tblBorders",
        "shd",
        "tblLayout",
        "tblCellMar",
        "tblLook",
        "tblCaption",
        "tblDescription",
    ],
    "trPr": [
        "cnfStyle",
        "divId",
        "gridBefore",
        "gridAfter",
        "wBefore",
        "wAfter",
        "cantSplit",
        "trHeight",
        "tblHeader",
        "tblCellSpacing",
        "jc",
        "hidden",
        "ins",
        "del",
        "trPrChange",
    ],
    "lvl": [
        "start",
        "numFmt",
        "lvlRestart",
        "pStyle",
        "isLgl",
        "suff",
        "lvlText",
        "lvlPicBulletId",
        "legacy",
        "lvlJc",
        "pPr",
        "rPr",
    ],
    "settings": [
        "updateFields",
        "hdrShapeDefaults",
        "footnotePr",
        "endnotePr",
        "compat",
        "docVars",
        "rsids",
        "mathPr",
        "attachedSchema",
        "themeFontLang",
        "clrSchemeMapping",
        "doNotIncludeSubdocsInStats",
        "doNotAutoCompressPictures",
        "forceUpgrade",
        "captions",
        "readModeInkLockDown",
        "smartTagType",
        "schemaLibrary",
        "shapeDefaults",
        "doNotEmbedSmartTags",
        "decimalSymbol",
        "listSeparator",
    ],
}


def _local(el) -> str:
    return el.tag.split("}")[-1]


def _place(parent, child) -> None:
    """Вставляет ``child`` в ``parent`` на место, требуемое схемой; заменяет одноимённый."""
    name = _local(child)
    for old in parent.findall(qn(f"w:{name}")):
        parent.remove(old)
    order = _ORDER[_local(parent)]
    later = set(order[order.index(name) + 1 :])
    for i, sibling in enumerate(parent):
        if _local(sibling) in later:
            parent.insert(i, child)
            return
    parent.append(child)


def _text_width_twips() -> int:
    return int((F.PAGE_WIDTH_MM - F.MARGIN_LEFT_MM - F.MARGIN_RIGHT_MM) / 25.4 * 1440)


def _add_field(paragraph, instr: str, cached: str = "") -> None:
    def run_with(child):
        r = OxmlElement("w:r")
        r.append(child)
        paragraph._p.append(r)
        return r

    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    run_with(begin)
    text = OxmlElement("w:instrText")
    text.set(qn("xml:space"), "preserve")
    text.text = f" {instr} "
    run_with(text)
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    run_with(sep)
    t = OxmlElement("w:t")
    t.text = cached
    run_with(t)
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run_with(end)


def _setup_footer(doc) -> None:
    section = doc.sections[0]
    section.different_first_page_header_footer = True
    footer = section.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.text = ""
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    _add_field(p, "PAGE", "1")
    for run in p._p.findall(qn("w:r")):
        rpr = OxmlElement("w:rPr")
        run.insert(0, rpr)
        fonts = OxmlElement("w:rFonts")
        for attr in ("w:ascii", "w:hAnsi", "w:cs"):
            fonts.set(qn(attr), F.FONT_NAME)
        rpr.append(fonts)
        size = OxmlElement("w:sz")
        size.set(qn("w:val"), str(int(F.FONT_SIZE_PT * 2)))
        rpr.append(size)


def _page_break_before(paragraph, value: bool = True) -> None:
    paragraph.paragraph_format.page_break_before = value


def _is_structural(text: str) -> bool:
    return text in F.STRUCTURAL_ELEMENTS or text.startswith(F.APPENDIX_PREFIX)


def _style_headings(doc) -> list:
    """Оформление заголовков по ГОСТ; возвращает заголовки для содержания."""
    toc_entries = []
    for p in doc.paragraphs:
        style = p.style.name if p.style is not None else ""
        if not style.startswith("Heading"):
            continue
        level = int(style.split()[-1])
        text = p.text.strip()
        pf = p.paragraph_format
        if level == 1:
            _page_break_before(p, True)
            pf.space_before = Pt(0)
            if _is_structural(text):
                pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
                pf.first_line_indent = Cm(0)
                if text.startswith(F.APPENDIX_PREFIX) and "|" in text:
                    parts = [part.strip() for part in text.split("|")]
                    for r in list(p.runs):
                        r._r.getparent().remove(r._r)
                    for i, part in enumerate(parts):
                        run = p.add_run(part)
                        run.bold = i != 1
                        if i < len(parts) - 1:
                            run.add_break(WD_BREAK.LINE)
                    text = f"{parts[0]} {parts[-1]}"
            if text in ("РЕФЕРАТ", "СОДЕРЖАНИЕ"):
                p.style = doc.styles["Normal"]
                pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
                pf.first_line_indent = Cm(0)
                pf.space_after = Pt(F.HEADING_SPACE_AFTER_PT)
                pf.keep_with_next = True
                for r in p.runs:
                    r.bold = True
                continue
        toc_entries.append((level, text, p))
    return toc_entries


def _insert_toc(doc, entries, pages: dict[str, int]) -> None:
    """Содержание: поле TOC с готовым результатом (номера страниц из первого прохода)."""
    target = None
    for p in doc.paragraphs:
        if p.text.strip() == "СОДЕРЖАНИЕ":
            target = p
            break
    if target is None:
        return
    anchor = target._p
    width = _text_width_twips()

    def make_par():
        par = OxmlElement("w:p")
        ppr = OxmlElement("w:pPr")
        par.append(ppr)
        return par, ppr

    first, ppr0 = make_par()
    begin = OxmlElement("w:r")
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), "begin")
    begin.append(fc)
    instr = OxmlElement("w:r")
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = ' TOC \\o "1-2" \\h \\z \\u '
    instr.append(it)
    sep = OxmlElement("w:r")
    fs = OxmlElement("w:fldChar")
    fs.set(qn("w:fldCharType"), "separate")
    sep.append(fs)
    paragraphs = []
    for i, (level, text, _p) in enumerate(entries):
        if i == 0:
            par, ppr = first, ppr0
            par.extend([begin, instr, sep])
        else:
            par, ppr = make_par()
        tabs = OxmlElement("w:tabs")
        tab = OxmlElement("w:tab")
        tab.set(qn("w:val"), "right")
        tab.set(qn("w:leader"), "dot")
        tab.set(qn("w:pos"), str(width))
        tabs.append(tab)
        ppr.append(tabs)
        ind = OxmlElement("w:ind")
        left = 0 if level == 1 else int(F.FIRST_LINE_INDENT_CM / 2.54 * 1440 / 2)
        ind.set(qn("w:left"), str(left))
        ind.set(qn("w:firstLine"), "0")
        ppr.append(ind)
        jc = OxmlElement("w:jc")
        jc.set(qn("w:val"), "left")
        ppr.append(jc)
        label = text.replace("|", " ")
        for content in (label, None, str(pages.get(text, ""))):
            r = OxmlElement("w:r")
            if content is None:
                r.append(OxmlElement("w:tab"))
            else:
                t = OxmlElement("w:t")
                t.set(qn("xml:space"), "preserve")
                t.text = content
                r.append(t)
            par.append(r)
        paragraphs.append(par)
    end = OxmlElement("w:r")
    fe = OxmlElement("w:fldChar")
    fe.set(qn("w:fldCharType"), "end")
    end.append(fe)
    paragraphs[-1].append(end)
    for par in reversed(paragraphs):
        anchor.addnext(par)


def _style_tables(doc) -> None:
    width = _text_width_twips()
    for table in doc.tables:
        tbl = table._tbl
        following = tbl.getnext()
        if following is not None and _local(following) == "p":
            ppr = following.find(qn("w:pPr"))
            if ppr is None:
                ppr = OxmlElement("w:pPr")
                following.insert(0, ppr)
            spacing = ppr.find(qn("w:spacing"))
            if spacing is None:
                spacing = OxmlElement("w:spacing")
                _place(ppr, spacing)
            spacing.set(qn("w:before"), str(int(F.TABLE_SPACE_AFTER_PT * 20)))
        tblpr = tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "4")
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "000000")
            borders.append(el)
        _place(tblpr, borders)
        tblw = OxmlElement("w:tblW")
        tblw.set(qn("w:w"), str(width))
        tblw.set(qn("w:type"), "dxa")
        _place(tblpr, tblw)
        # небольшая таблица не разрывается между страницами (ГОСТ требует иначе
        # подписи «Продолжение таблицы»); большая переносится с повтором шапки
        keep = len(table.rows) <= F.TABLE_KEEP_TOGETHER_ROWS
        for r_i, row in enumerate(table.rows):
            trpr = row._tr.get_or_add_trPr()
            _place(trpr, OxmlElement("w:cantSplit"))
            if r_i == 0:
                _place(trpr, OxmlElement("w:tblHeader"))
            for cell in row.cells:
                for p in cell.paragraphs:
                    pf = p.paragraph_format
                    pf.keep_with_next = keep and r_i < len(table.rows) - 1
                    pf.first_line_indent = Cm(0)
                    pf.left_indent = Cm(0)
                    pf.space_before = Pt(0)
                    pf.space_after = Pt(0)
                    pf.line_spacing = F.TABLE_LINE_SPACING
                    if pf.alignment == WD_ALIGN_PARAGRAPH.JUSTIFY or pf.alignment is None:
                        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    for r in p.runs:
                        r.font.size = Pt(F.TABLE_FONT_SIZE_PT)
                        if r_i == 0:
                            r.bold = True


def _number_equations(doc) -> None:
    """Формула по центру, номер справа: маркер EQNUM(...) сливается с формулой."""
    width = _text_width_twips()
    body = doc.element.body
    for p in list(body.iter(qn("w:p"))):
        text = "".join(t.text or "" for t in p.iter(qn("w:t")))
        if not text.startswith("EQNUM"):
            continue
        number = text[len("EQNUM") :]
        prev = p.getprevious()
        math_para = prev.find(f"{{{M_NS}}}oMathPara") if prev is not None else None
        if math_para is None:
            raise ValueError(f"номер {number} не следует за формулой")
        omaths = math_para.findall(f"{{{M_NS}}}oMath")
        prev.remove(math_para)
        ppr = prev.find(qn("w:pPr"))
        if ppr is None:
            ppr = OxmlElement("w:pPr")
            prev.insert(0, ppr)
        tabs = OxmlElement("w:tabs")
        for val, pos in (("center", width // 2), ("right", width)):
            tab = OxmlElement("w:tab")
            tab.set(qn("w:val"), val)
            tab.set(qn("w:pos"), str(pos))
            tabs.append(tab)
        _place(ppr, tabs)
        ind = OxmlElement("w:ind")
        ind.set(qn("w:left"), "0")
        ind.set(qn("w:firstLine"), "0")
        _place(ppr, ind)
        jc = OxmlElement("w:jc")
        jc.set(qn("w:val"), "left")
        _place(ppr, jc)
        tab_run = OxmlElement("w:r")
        tab_run.append(OxmlElement("w:tab"))
        prev.append(tab_run)
        for om in omaths:
            prev.append(om)
        tab_run2 = OxmlElement("w:r")
        tab_run2.append(OxmlElement("w:tab"))
        prev.append(tab_run2)
        num_run = OxmlElement("w:r")
        t = OxmlElement("w:t")
        t.text = number
        num_run.append(t)
        prev.append(num_run)
        p.getparent().remove(p)


def _fix_numbering(doc) -> None:
    """Перечисления: тире для маркированных, «1)» для нумерованных; отступ как у абзаца."""
    part = doc.part.numbering_part
    if part is None:
        return
    numbering = part.element
    indent = str(int(F.LIST_INDENT_CM / 2.54 * 1440))
    for lvl in numbering.iter(qn("w:lvl")):
        ilvl = int(lvl.get(qn("w:ilvl")))
        fmt = lvl.find(qn("w:numFmt"))
        text = lvl.find(qn("w:lvlText"))
        if fmt is None or text is None:
            continue
        if fmt.get(qn("w:val")) == "bullet":
            text.set(qn("w:val"), F.BULLET_CHAR)
        elif ilvl == 0:
            text.set(qn("w:val"), F.ORDERED_FORMAT)
        suff = OxmlElement("w:suff")
        suff.set(qn("w:val"), "space")
        _place(lvl, suff)
        ppr = lvl.find(qn("w:pPr"))
        if ppr is None:
            ppr = OxmlElement("w:pPr")
            _place(lvl, ppr)
        ind = OxmlElement("w:ind")
        ind.set(qn("w:left"), str(int(indent) * ilvl))
        ind.set(qn("w:firstLine"), indent)
        _place(ppr, ind)
        rpr = lvl.find(qn("w:rPr"))
        if rpr is None:
            rpr = OxmlElement("w:rPr")
            _place(lvl, rpr)
        fonts = rpr.find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            rpr.append(fonts)
        for attr in ("w:ascii", "w:hAnsi", "w:cs"):
            fonts.set(qn(attr), F.FONT_NAME)


def _fix_figures(doc) -> None:
    """Абзацы с рисунками — без отступа, по центру, вместе с подписью."""
    for p in doc.paragraphs:
        if p._p.findall(".//" + qn("w:drawing")):
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.line_spacing = 1.0


def _title_page(doc) -> None:
    """Титульный лист перед первым абзацем документа."""
    t = F.TITLE_PAGE
    first = doc.paragraphs[0]

    def add(
        text: str, *, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, before=0.0, caps=False, size=None
    ):
        p = first.insert_paragraph_before()
        p.style = doc.styles["Normal"]
        pf = p.paragraph_format
        pf.alignment = align
        pf.first_line_indent = Cm(0)
        pf.space_before = Pt(before)
        pf.line_spacing = 1.15
        lines = text.split("\n")
        for i, line in enumerate(lines):
            run = p.add_run(line.upper() if caps else line)
            run.bold = bold
            if size:
                run.font.size = Pt(size)
            if i < len(lines) - 1:
                run.add_break(WD_BREAK.LINE)
        return p

    add(t["ministry"])
    add(t["faculty"], before=12)
    add(t["department"])
    add(t["work_type"], bold=True, before=60)
    add("на тему:", before=6)
    add(f"«{t['topic']}»", bold=True)
    add(t["direction"], before=24)
    add(t["profile"])
    authors = "\n".join(f"{name}, {role}" for name, role in t["authors"])
    add(
        "Выполнили:\n" + authors + "\n(подписи) ____________",
        align=WD_ALIGN_PARAGRAPH.RIGHT,
        before=36,
    )
    sup_name, sup_role = t["supervisor"]
    add(
        f"Руководитель:\n{sup_role}\n{sup_name}\n(подпись) ____________",
        align=WD_ALIGN_PARAGRAPH.RIGHT,
        before=18,
    )
    add(
        t["approval"] + "\n____________ «___» __________ 2027 г.",
        align=WD_ALIGN_PARAGRAPH.LEFT,
        before=24,
    )
    last = add(t["city_year"], before=36)
    last.runs[-1].add_break(WD_BREAK.PAGE)


def _style_code(doc) -> None:
    """Листинги: моноширинный шрифт, без отступа и выравнивания по ширине, одинарный интервал.

    Стиль «Source Code» pandoc добавляет при конвертации, поэтому он не
    настраивается через reference.docx и оформляется здесь.
    """
    for p in doc.paragraphs:
        ppr = p._p.pPr
        if ppr is None or ppr.pStyle is None or ppr.pStyle.val != "SourceCode":
            continue
        pf = p.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.first_line_indent = Cm(0)
        pf.left_indent = Cm(0)
        pf.line_spacing = 1.0
        pf.space_before = Pt(6)
        pf.space_after = Pt(6)
        pf.keep_together = True
        for run in p.runs:
            _set_run_font(run, F.CODE_FONT_NAME, F.CODE_FONT_SIZE_PT)


def _set_run_font(run, name: str, size: float) -> None:
    run.font.name = name
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0, 0, 0)
    fonts = run._r.get_or_add_rPr().find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        run._r.get_or_add_rPr().insert(0, fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), name)


def _fix_math_rpr(doc) -> None:
    r"""В свойствах OMML ``m:nor`` и ``m:sty`` взаимоисключающие (ECMA-376).

    Pandoc для ``\text{…}`` пишет оба; Word такой документ может счесть
    повреждённым, поэтому ``m:sty`` удаляется.
    """
    for rpr in doc.element.body.iter(f"{{{M_NS}}}rPr"):
        if rpr.find(f"{{{M_NS}}}nor") is not None:
            for sty in rpr.findall(f"{{{M_NS}}}sty"):
                rpr.remove(sty)


def _fix_keywords(doc) -> None:
    """Ключевые слова реферата (абзац после сведений об объёме) — без выравнивания по ширине."""
    paragraphs = doc.paragraphs
    for i, p in enumerate(paragraphs[:-1]):
        if p.text.startswith("Пояснительная записка содержит"):
            paragraphs[i + 1].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            return


def postprocess(src: Path, dst: Path, pages: dict[str, int], markers: bool) -> list:
    """Оформление документа после pandoc; возвращает записи содержания."""
    doc = Document(src)
    _page_setup(doc)
    entries = _style_headings(doc)
    _insert_toc(doc, entries, pages)
    _style_tables(doc)
    _number_equations(doc)
    _fix_numbering(doc)
    _fix_figures(doc)
    _fix_keywords(doc)
    _style_code(doc)
    _fix_math_rpr(doc)
    if markers:  # первый проход: невидимые метки для поиска страниц заголовков в PDF
        for i, (_level, _text, p) in enumerate(entries):
            run = p.runs[0] if p.runs else p.add_run("")
            marker = run._r.getparent().makeelement(qn("w:r"), {})
            rpr = OxmlElement("w:rPr")
            color = OxmlElement("w:color")
            color.set(qn("w:val"), "FFFFFF")
            size = OxmlElement("w:sz")
            size.set(qn("w:val"), "2")
            rpr.extend([color, size])
            marker.append(rpr)
            t = OxmlElement("w:t")
            t.text = f"QQH{i:03d}QQ"
            marker.append(t)
            p._p.insert(1 if p._p.find(qn("w:pPr")) is not None else 0, marker)
    _title_page(doc)
    _setup_footer(doc)
    settings = doc.settings.element
    upd = OxmlElement("w:updateFields")
    upd.set(qn("w:val"), "true")
    _place(settings, upd)
    doc.core_properties.title = F.TITLE_PAGE["topic"]
    doc.core_properties.author = "Будаев К. В., Гарифзянов Т. Р."
    doc.save(dst)
    return entries


# ---------------------------------------------------------------------------
# Рендер и два прохода
# ---------------------------------------------------------------------------


def to_pdf(docx: Path, out_dir: Path) -> Path:
    """PDF через LibreOffice (для проверки и подсчёта страниц)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / (docx.stem + ".pdf")
    pdf.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="lo_profile_") as profile:
        result = subprocess.run(
            [
                "soffice",
                f"-env:UserInstallation={Path(profile).as_uri()}",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(out_dir),
                str(docx),
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env={**__import__("os").environ, "SAL_USE_VCLPLUGIN": "svp"},
        )
    if not pdf.exists():
        raise RuntimeError(f"LibreOffice не создал PDF: {result.stderr.strip()[-500:]}")
    return pdf


def pdf_pages(pdf: Path) -> list[str]:
    """Текст PDF по страницам (pdftotext)."""
    text = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True, text=True
    ).stdout
    return text.split("\f")


def run_pandoc(markdown: str, reference: Path, out: Path) -> None:
    """Markdown → docx: формулы в OMML, стили из reference.docx, без подсветки кода."""
    pypandoc.convert_text(
        markdown,
        "docx",
        format="markdown+tex_math_dollars+pipe_tables+implicit_figures+link_attributes"
        "-auto_identifiers",
        outputfile=str(out),
        extra_args=[
            f"--reference-doc={reference}",
            f"--resource-path={ROOT}",
            "--wrap=none",
            "--syntax-highlighting=none",
        ],
    )


def build() -> Path:
    """Два прохода сборки; возвращает путь к итоговому .docx."""
    out_docx = ROOT / F.OUTPUT_DOCX
    build_dir = ROOT / F.OUTPUT_PDF_DIR
    build_dir.mkdir(parents=True, exist_ok=True)
    reference = make_reference_docx(build_dir / "reference.docx")

    # проход 1: метки заголовков, объём
    prepared, _order, _facts = prepare()
    raw = build_dir / "pass1_raw.docx"
    run_pandoc(prepared.markdown, reference, raw)
    pass1 = build_dir / "pass1.docx"
    entries = postprocess(raw, pass1, {}, markers=True)
    pages_text = pdf_pages(to_pdf(pass1, build_dir))
    page_of: dict[str, int] = {}
    for i, (_level, text, _p) in enumerate(entries):
        marker = f"QQH{i:03d}QQ"
        for n, page in enumerate(pages_text, 1):
            if marker in page:
                page_of[text] = n
                break
    total_pages = len([p for p in pages_text if p.strip()])

    # проход 2: номера страниц в содержании и объём в реферате
    prepared, _order, _facts = prepare({"doc.pages": total_pages})
    run_pandoc(prepared.markdown, reference, raw)
    postprocess(raw, out_docx, page_of, markers=False)
    final_pdf = to_pdf(out_docx, build_dir)
    final_pages = len([p for p in pdf_pages(final_pdf) if p.strip()])
    shutil.copy(final_pdf, out_docx.with_suffix(".pdf"))
    print(
        f"{out_docx.relative_to(ROOT)}: страниц {final_pages} (в реферате {total_pages}), "
        f"рисунков {prepared.figures}, таблиц {prepared.tables}, формул {prepared.equations}, "
        f"источников {prepared.sources}, приложений {prepared.appendices}"
    )
    if final_pages != total_pages:
        print("ВНИМАНИЕ: число страниц изменилось между проходами — пересоберите записку")
    return out_docx


if __name__ == "__main__":
    build()

"""Оформление библиографических записей по ГОСТ Р 7.0.100–2018.

Ответственные: Будаев К. В., Гарифзянов Т. Р.

Записи хранятся в ``notebook/references.yaml`` в структурированном виде
(авторы, заглавие, источник, год, том, номер, страницы, DOI и т. д.).
Отсюда два следствия:

* оформление единообразно и при необходимости меняется в одном месте
  (константы ниже), например если кафедра потребует другой вариант
  разделителей [УТОЧНИТЬ У РУКОВОДИТЕЛЯ];
* один и тот же список используют ноутбук и пояснительная записка.

Правила (ГОСТ Р 7.0.100–2018):

* при одном–трёх авторах запись начинается с заголовка «Фамилия, И. О.»,
  при четырёх и более — с заглавия;
* сведения об ответственности приводятся после «/». При пяти и более
  авторах указываются первые три и «[et al.]» («[и др.]» для русских
  источников);
* области разделяются «. – », сведения о составной части — после «//»;
* у электронных ресурсов указываются URL, дата обращения и «Текст : электронный».
"""

from __future__ import annotations

import re
from typing import Any

#: Разделитель областей описания (по ГОСТ допускается заменять на «. »).
AREA = ". – "
#: Дата обращения к электронным ресурсам.
ACCESSED = "06.10.2026"
#: Сколько авторов перечислять при пяти и более.
MAX_LISTED = 3

_LABELS = {
    "en": {
        "vol": "Vol.",
        "no": "no.",
        "p": "P.",
        "art": "Art.",
        "etal": "[et al.]",
        "pages_total": "p.",
    },
    "ru": {"vol": "Т.", "no": "№", "p": "С.", "art": "Ст.", "etal": "[и др.]", "pages_total": "с."},
}


_INITIAL = re.compile(r"^[A-ZА-ЯЁÁ-Ž]\.(-[A-ZА-ЯЁÁ-Ž]\.)?$")


def _split(author: str) -> tuple[str, str]:
    """«Фамилия И. О.» → («Фамилия», «И. О.»); инициалы вида «F.-M.» поддерживаются."""
    parts = author.split()
    initials = [p for p in parts if _INITIAL.match(p)]
    surname = " ".join(p for p in parts if not _INITIAL.match(p))
    return surname, " ".join(initials)


def _heading(author: str) -> str:
    surname, initials = _split(author)
    return f"{surname}, {initials}" if initials else surname


def _responsibility(authors: list[str], lang: str, et_al: bool) -> str:
    names = []
    for author in authors:
        surname, initials = _split(author)
        names.append(f"{initials} {surname}".strip())
    if et_al or len(names) >= 5:
        return ", ".join(names[:MAX_LISTED]) + " " + _LABELS[lang]["etal"]
    return ", ".join(names)


def _end(text: str) -> str:
    return text if text.endswith((".", "?", "!")) else text + "."


def render(entry: dict[str, Any]) -> str:
    """Возвращает запись по ГОСТ Р 7.0.100–2018 (или поле ``gost``, если задано)."""
    if entry.get("gost"):
        return entry["gost"]
    lang = entry.get("lang", "en")
    lab = _LABELS[lang]
    authors: list[str] = entry.get("authors", [])
    kind = entry["type"]
    title = entry["title"]
    if entry.get("subtitle"):
        title += f" : {entry['subtitle']}"
    if kind == "thesis":
        title += f" : {entry.get('thesis', 'PhD thesis')}"
    if kind == "report":
        title += f" : {entry.get('report', 'technical report')}"

    et_al = bool(entry.get("et_al"))
    head = ""
    if 1 <= len(authors) <= 3 and not et_al:
        head = _heading(authors[0]) + " "
    resp = f" / {_responsibility(authors, lang, et_al)}" if authors else ""
    if entry.get("org"):
        resp += f" ; {entry['org']}" if resp else f" / {entry['org']}"
    out = f"{head}{title}{resp}"

    areas: list[str] = []
    if kind == "article":
        out += f" // {entry['journal']}"
        areas.append(str(entry["year"]))
        vol = []
        if entry.get("volume"):
            vol.append(f"{lab['vol']} {entry['volume']}")
        if entry.get("number"):
            vol.append(f"{lab['no']} {entry['number']}")
        if vol:
            areas.append(", ".join(vol))
        if entry.get("article"):
            areas.append(f"{lab['art']} {entry['article']}")
        if entry.get("pages"):
            areas.append(f"{lab['p']} {entry['pages']}")
    elif kind == "inproceedings":
        out += f" // {entry['booktitle']}"
        imprint = ""
        if entry.get("place"):
            imprint = entry["place"]
            if entry.get("publisher"):
                imprint += f" : {entry['publisher']}"
            imprint += f", {entry['year']}"
        else:
            imprint = str(entry["year"])
        areas.append(imprint)
        if entry.get("pages"):
            areas.append(f"{lab['p']} {entry['pages']}")
        if entry.get("series"):
            areas.append(f"({entry['series']})")
    elif kind in {"book", "thesis", "report"}:
        imprint = entry.get("place", "")
        if entry.get("publisher"):
            imprint += f" : {entry['publisher']}"
        imprint += f", {entry['year']}" if imprint else str(entry["year"])
        areas.append(imprint)
        if entry.get("pages_total"):
            areas.append(f"{entry['pages_total']} {lab['pages_total']}")
        if entry.get("series"):
            areas.append(f"({entry['series']})")
        if entry.get("isbn"):
            areas.append(f"ISBN {entry['isbn']}")
    elif kind == "online":
        if entry.get("container"):
            out += f" // {entry['container']}"
        if entry.get("version"):
            areas.append(f"Version {entry['version']}")
        if entry.get("year"):
            areas.append(str(entry["year"]))
        if entry.get("eprint"):
            areas.append(entry["eprint"])
    else:  # pragma: no cover - защитная ветка
        raise ValueError(f"неизвестный тип записи {kind!r}")

    if entry.get("doi"):
        areas.append(f"DOI {entry['doi']}")
    if kind == "online" or entry.get("url_required"):
        areas.append(f"URL: {entry['url']} (дата обращения: {ACCESSED})")
        areas.append("Текст : электронный")
    text = out
    for area in areas:
        text += (" – " if text.endswith(".") else AREA) + area
    return _end(text)

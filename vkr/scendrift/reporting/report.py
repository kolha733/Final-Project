"""Отчёты бенчмарка: сводные таблицы и экспорт в CSV, Markdown и HTML.

Ответственный: Гарифзянов Т. Р.

* :func:`summary_table` — среднее и стандартное отклонение метрик по
  детекторам (или другой группировке);
* :func:`format_table` — подписи «среднее ± ст. откл.» с десятичной запятой;
* :func:`export_tables` — сырые результаты в CSV и сводка в Markdown;
* :func:`html_report` — самодостаточный HTML-отчёт (таблицы и рисунки
  встраиваются в файл, внешних зависимостей нет).
"""

from __future__ import annotations

import base64
import html
import io as _io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.figure
import pandas as pd

__all__ = ["summary_table", "format_table", "export_tables", "html_report"]


def summary_table(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    *,
    by: str = "detector",
    order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Среднее и ст. откл. метрик по группам (MultiIndex столбцов: метрика × mean/std)."""
    table = frame.groupby(by)[list(metrics)].agg(["mean", "std"])
    if order is not None:
        table = table.reindex([o for o in order if o in table.index])
    return table


def _num(value: float, digits: int) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.{digits}f}".replace(".", ",")


def format_table(table: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """Строковая таблица «среднее ± ст. откл.» с десятичной запятой (ГОСТ 7.32)."""
    out = pd.DataFrame(index=table.index)
    for metric in table.columns.get_level_values(0).unique():
        mean, std = table[(metric, "mean")], table[(metric, "std")]
        out[metric] = [
            _num(m, digits) if pd.isna(s) else f"{_num(m, digits)} ± {_num(s, digits)}"
            for m, s in zip(mean, std, strict=True)
        ]
    return out


def export_tables(
    frame: pd.DataFrame, summary: pd.DataFrame, directory: str | Path, name: str
) -> dict[str, Path]:
    """Сохраняет сырые результаты (CSV) и сводку (CSV и Markdown)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "runs_csv": directory / f"{name}_runs.csv",
        "summary_csv": directory / f"{name}_summary.csv",
        "summary_md": directory / f"{name}_summary.md",
    }
    frame.to_csv(paths["runs_csv"], index=False)
    summary.to_csv(paths["summary_csv"])
    paths["summary_md"].write_text(format_table(summary).to_markdown() + "\n", encoding="utf-8")
    return paths


def _figure_html(fig: matplotlib.figure.Figure | str | Path) -> str:
    if isinstance(fig, str | Path):
        data = Path(fig).read_bytes()
    else:
        buffer = _io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        data = buffer.getvalue()
    encoded = base64.b64encode(data).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="рисунок">'


def html_report(title: str, blocks: Sequence[tuple[str, Any]], path: str | Path) -> Path:
    """Самодостаточный HTML-отчёт.

    Args:
        title: заголовок отчёта.
        blocks: пары (подзаголовок, содержимое). Содержимое — строка
            (абзац), ``DataFrame`` (таблица), рисунок matplotlib или путь к PNG.
        path: файл отчёта.
    """
    parts = [
        "<!doctype html><html lang='ru'><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:'Times New Roman',serif;max-width:1100px;margin:2em auto;"
        "color:#0b0b0b;background:#fff}table{border-collapse:collapse;margin:1em 0}"
        "td,th{border:1px solid #c9c8c4;padding:4px 8px;text-align:right}"
        "th{background:#f3f2ef}img{max-width:100%}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
    ]
    for heading, content in blocks:
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        if isinstance(content, pd.DataFrame):
            parts.append(content.to_html(border=0))
        elif isinstance(content, matplotlib.figure.Figure | Path) or (
            isinstance(content, str) and content.endswith(".png")
        ):
            parts.append(_figure_html(content))
        else:
            parts.append(f"<p>{html.escape(str(content))}</p>")
    parts.append("</body></html>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path

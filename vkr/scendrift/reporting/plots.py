"""Графики оценки детекторов: CD-диаграмма и растр срабатываний.

Ответственный: Гарифзянов Т. Р.

* :func:`cd_diagram` — диаграмма критической разности [Demšar, 2006]:
  средние ранги методов на оси; методы, попарно не различающиеся по
  критерию Неменьи, соединены жирной линией.
* :func:`detection_raster` — срабатывания каждого детектора на одной
  реализации потока относительно ground truth: переходы и окна допуска
  закрашены; обнаружения (TP), избыточные срабатывания и ложные тревоги
  отмечены разными маркерами.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from scendrift.evaluation.metrics import match_detections
from scendrift.evaluation.protocol import EvaluationSpec
from scendrift.evaluation.stats import nemenyi_cliques
from scendrift.interfaces import GroundTruth
from scendrift.reporting.style import CATEGORICAL, TEXT, TEXT_SECONDARY

__all__ = ["cd_diagram", "detection_raster"]


def cd_diagram(
    ranks: pd.Series,
    cd: float,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """CD-диаграмма: средние ранги (1 — лучший) и группы без значимых различий."""
    ranks = ranks.sort_values()
    k = len(ranks)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.9 + 0.32 * k))
    names, values = list(ranks.index), ranks.to_numpy()
    half = (k + 1) // 2
    step = 0.32
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-(half + 1.4) * step - 0.25, 0.75)
    ax.axis("off")
    ax.hlines(0, 1, k, color=TEXT, linewidth=1.2)
    for r in range(1, k + 1):
        ax.vlines(r, 0, 0.07, color=TEXT, linewidth=1)
        ax.text(r, 0.12, str(r), ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.hlines(0.55, 1, 1 + cd, color=TEXT, linewidth=2)
    ax.vlines([1, 1 + cd], 0.5, 0.6, color=TEXT, linewidth=1)
    ax.text(
        1 + cd / 2, 0.62, f"CD = {cd:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=9
    )
    for i, (name, value) in enumerate(zip(names, values, strict=True)):
        left = i < half
        level = (i if left else k - 1 - i) + 1
        y = -level * step - 0.25
        x_end = 0.5 if left else k + 0.5
        ax.plot([value, value, x_end], [0, y, y], color=TEXT_SECONDARY, linewidth=1)
        label = (
            f"{name} ({value:.2f})".replace(".", ",")
            if left
            else (f"({value:.2f}) {name}".replace(".", ","))
        )
        ax.text(
            x_end + (-0.05 if left else 0.05),
            y,
            label,
            ha="right" if left else "left",
            va="center",
            fontsize=9.5,
            color=TEXT,
        )
    for j, clique in enumerate(nemenyi_cliques(ranks, cd)):
        lo, hi = ranks[list(clique)].min(), ranks[list(clique)].max()
        y = -0.1 - 0.07 * j
        ax.hlines(y, lo - 0.03, hi + 0.03, color=CATEGORICAL[1], linewidth=4)
    if title:
        ax.set_title(title, fontsize=11)
    return ax


def detection_raster(
    ground_truth: GroundTruth,
    protocol: EvaluationSpec,
    detections: Mapping[str, Sequence[int]],
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Срабатывания детекторов на одной реализации потока относительно ground truth."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 0.6 + 0.35 * len(detections)))
    delta = protocol.acceptance_window
    for k, iv in enumerate(ground_truth.intervals):
        ax.axvspan(
            iv.onset,
            iv.end + delta,
            color=CATEGORICAL[1],
            alpha=0.10,
            linewidth=0,
            label="окно допуска Λₖ" if k == 0 else None,
        )
        ax.axvspan(
            iv.onset,
            max(iv.end, iv.onset + 1),
            color=CATEGORICAL[1],
            alpha=0.35,
            linewidth=0,
            label="переход" if k == 0 else None,
        )
    ax.axvspan(
        0, protocol.warmup, color=TEXT_SECONDARY, alpha=0.12, linewidth=0, label="разогрев W"
    )
    marks = {
        "tp": ("обнаружение (TP)", CATEGORICAL[2], "|", 12),
        "red": ("избыточное (FP_red)", CATEGORICAL[0], "|", 8),
        "fa": ("ложная тревога (FP_out)", TEXT, "x", 6),
    }
    used = set()
    names = list(detections)
    for row, name in enumerate(names):
        m = match_detections(detections[name], ground_truth, protocol)
        groups = {
            "tp": [h for h in m.hits if h is not None],
            "red": list(m.redundant),
            "fa": list(m.false_alarms),
        }
        for key, times in groups.items():
            if not times:
                continue
            label, color, marker, size = marks[key]
            ax.plot(
                times,
                [row] * len(times),
                linestyle="none",
                marker=marker,
                color=color,
                markersize=size,
                markeredgewidth=2 if marker == "|" else 1.2,
                label=label if key not in used else None,
            )
            used.add(key)
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlim(0, ground_truth.n_samples)
    ax.set_xlabel("номер объекта t")
    return ax

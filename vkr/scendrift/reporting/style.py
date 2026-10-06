"""Единый стиль графиков ноутбука, записки и презентации.

Ответственный: Гарифзянов Т. Р.

Категориальная палитра — первые слоты проверенной эталонной палитры.
Три первых цвета различимы при всех типах дальтонизма даже при попарном
сравнении всех серий. Поэтому на точечных графиках и малых мультипликаторах
используется не больше трёх серий, а на линейных графиках (сравниваются
соседние серии) — до восьми. Порядок слотов фиксирован: цвет закреплён
за сущностью (например, за детектором), а не за её рангом.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

__all__ = [
    "CATEGORICAL",
    "SEQUENTIAL",
    "DIVERGING",
    "TEXT",
    "TEXT_SECONDARY",
    "GRID",
    "SURFACE",
    "apply_style",
    "color_map",
    "decimal_comma",
]

#: Категориальные цвета (светлая тема), фиксированный порядок слотов.
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # 1 синий
    "#eb6834",  # 2 оранжевый
    "#1baf7a",  # 3 аква
    "#eda100",  # 4 жёлтый
    "#e87ba4",  # 5 пурпурный
    "#008300",  # 6 зелёный
    "#4a3aa7",  # 7 фиолетовый
    "#e34948",  # 8 красный
)

#: Последовательная шкала (один тон, от светлого к тёмному) — тепловые карты.
SEQUENTIAL: tuple[str, ...] = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)

#: Расходящаяся шкала: синий ↔ серый ↔ красный.
DIVERGING: tuple[str, str, str] = ("#2a78d6", "#f0efec", "#e34948")

TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"


def apply_style(font_size: float = 10.5) -> None:
    """Устанавливает параметры matplotlib: тонкие линии, ненавязчивая сетка."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.titlesize": font_size + 1.5,
            "axes.titleweight": "bold",
            "axes.labelsize": font_size,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=list(CATEGORICAL)),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "font.size": font_size,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.dpi": 110,
        }
    )


def color_map(names: Sequence[str]) -> dict[str, str]:
    """Закрепляет цвета за именами в заданном порядке (не по рангу).

    Raises:
        ValueError: если имён больше, чем цветов в палитре.
    """
    if len(names) > len(CATEGORICAL):
        raise ValueError("больше восьми серий: объедините лишние в «прочие» или разбейте график")
    return dict(zip(names, CATEGORICAL, strict=False))


def _comma(value: float, _pos: int | None = None) -> str:
    text = f"{value:.10g}"
    return text.replace(".", ",").replace("-", "−")


def decimal_comma(*axes: mpl.axes.Axes) -> None:
    """Подписи делений осей с десятичной запятой (ГОСТ 7.32)."""
    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(_comma))
        ax.yaxis.set_major_formatter(FuncFormatter(_comma))

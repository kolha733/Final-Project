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
from pathlib import Path

import matplotlib as mpl
from matplotlib.ticker import FixedFormatter, FuncFormatter

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
    "save_figure",
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
    """Подписи делений осей с десятичной запятой (ГОСТ 7.32).

    Оси с заданными текстовыми подписями (``set_yticks(ticks, labels)``) не
    трогаются: иначе названия заменились бы номерами делений. Matplotlib
    хранит такие подписи в ``FixedFormatter`` или в ``FuncFormatter`` со
    своей функцией.
    """
    for ax in axes:
        for axis in (ax.xaxis, ax.yaxis):
            current = axis.get_major_formatter()
            custom = isinstance(current, FixedFormatter) or (
                isinstance(current, FuncFormatter) and current.func is not _comma
            )
            if not custom:
                axis.set_major_formatter(FuncFormatter(_comma))


def save_figure(fig: mpl.figure.Figure, path: str | Path, *, clean_dir: str = "zapiska") -> Path:
    """Сохраняет рисунок и его «чистую» копию без общего заголовка.

    Общий заголовок (``fig.suptitle``) с номером рисунка нужен в ноутбуке. В
    пояснительной записке и презентации подпись по ГОСТ 7.32 ставится под
    рисунком и нумеруется иначе, поэтому туда идёт копия без заголовка:
    ``<каталог>/zapiska/<имя>``.

    Returns:
        Путь к основному файлу.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    clean = path.parent / clean_dir / path.name
    clean.parent.mkdir(parents=True, exist_ok=True)
    title = fig._suptitle  # noqa: SLF001 — у matplotlib нет публичного доступа
    if title is not None:
        title.set_visible(False)
    fig.savefig(clean)
    if title is not None:
        title.set_visible(True)
    return path

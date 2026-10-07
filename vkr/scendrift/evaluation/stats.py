"""Статистическое сравнение детекторов по набору сценариев (Demšar, 2006).

Ответственный: Гарифзянов Т. Р.

Сравниваются k методов на N блоках. Блок — сценарий; значение метрики
усредняется по повторам (seed), чтобы повторы не считались независимыми
наборами данных.

* Средние ранги: в каждом блоке методы ранжируются (1 — лучший, при
  равенстве — средний ранг), затем ранги усредняются.
* Критерий Фридмана [Friedman, 1937] и его F-поправка Имана–Давенпорта,
  которую рекомендует Demšar: F_F = (N − 1)χ²_F / (N(k − 1) − χ²_F),
  распределение F(k − 1, (k − 1)(N − 1)). χ²_F вычисляется
  ``scipy.stats.friedmanchisquare`` (с поправкой на связи).
* Пост-хок Неменьи [Nemenyi, 1963]: два метода различаются, если разность
  средних рангов больше критической разности
  CD = q_α·√(k(k + 1)/(6N)), где q_α — квантиль распределения
  стьюдентизированного размаха с k и ∞ степенями свободы, делённый на √2.
* Попарный критерий Уилкоксона [Wilcoxon, 1945] с поправкой Холма [Holm, 1979].

Блоки, в которых у какого-либо метода метрика не определена (NaN),
исключаются; их число возвращается.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "block_matrix",
    "average_ranks",
    "FriedmanResult",
    "friedman",
    "nemenyi_cd",
    "nemenyi_cliques",
    "holm",
    "wilcoxon_holm",
]


def block_matrix(
    frame: pd.DataFrame,
    metric: str,
    *,
    block: str = "scenario_id",
    method: str = "detector",
    methods: list[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Матрица «блок × метод» (среднее по повторам) и число исключённых блоков."""
    table = frame.pivot_table(index=block, columns=method, values=metric, aggfunc="mean")
    if methods is not None:
        table = table[methods]
    full = table.dropna(axis=0, how="any")
    return full, len(table) - len(full)


def average_ranks(matrix: pd.DataFrame, higher_is_better: bool = True) -> pd.Series:
    """Средние ранги методов (1 — лучший)."""
    values = -matrix.to_numpy() if higher_is_better else matrix.to_numpy()
    ranks = np.apply_along_axis(stats.rankdata, 1, values)
    return pd.Series(ranks.mean(axis=0), index=matrix.columns).sort_values()


@dataclass(frozen=True)
class FriedmanResult:
    """Результат критерия Фридмана."""

    k: int
    n_blocks: int
    chi2: float
    p_chi2: float
    f_stat: float
    p_f: float


def friedman(matrix: pd.DataFrame) -> FriedmanResult:
    """Критерий Фридмана и поправка Имана–Давенпорта.

    Raises:
        ValueError: если методов меньше трёх или блоков меньше двух.
    """
    n, k = matrix.shape
    if k < 3 or n < 2:
        raise ValueError("нужно не менее трёх методов и двух блоков")
    chi2, p = stats.friedmanchisquare(*[matrix[c].to_numpy() for c in matrix.columns])
    denom = n * (k - 1) - chi2
    f_stat = (n - 1) * chi2 / denom if denom > 0 else math.inf
    p_f = float(stats.f.sf(f_stat, k - 1, (k - 1) * (n - 1))) if denom > 0 else 0.0
    return FriedmanResult(k, n, float(chi2), float(p), float(f_stat), p_f)


def nemenyi_cd(k: int, n_blocks: int, alpha: float = 0.05) -> tuple[float, float]:
    """Критическая разность Неменьи и q_α.

    Returns:
        Пара (CD, q_α).
    """
    q_alpha = float(stats.studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2))
    return q_alpha * math.sqrt(k * (k + 1) / (6.0 * n_blocks)), q_alpha


def nemenyi_cliques(ranks: pd.Series, cd: float) -> list[tuple[str, ...]]:
    """Максимальные группы методов, попарно не различающихся по Неменьи.

    Методы упорядочены по среднему рангу; группа — отрезок подряд идущих
    методов с размахом рангов не больше CD. Вложенные группы отбрасываются.
    """
    ordered = ranks.sort_values()
    names, values = list(ordered.index), ordered.to_numpy()
    cliques = []
    for i in range(len(names)):
        j = i
        while j + 1 < len(names) and values[j + 1] - values[i] <= cd:
            j += 1
        if j > i:
            cliques.append((i, j))
    maximal = [
        c for c in cliques if not any(o != c and o[0] <= c[0] and c[1] <= o[1] for o in cliques)
    ]
    return [tuple(names[a : b + 1]) for a, b in maximal]


def holm(pvalues: list[float]) -> list[float]:
    """Поправка Холма на множественные сравнения (скорректированные p-значения)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def wilcoxon_holm(
    matrix: pd.DataFrame, higher_is_better: bool = True, alpha: float = 0.05
) -> pd.DataFrame:
    """Попарные критерии Уилкоксона с поправкой Холма.

    Returns:
        Таблица: пара методов, победы/ничьи/поражения первого, медиана
        разности, p-значение, p с поправкой Холма, вывод.
    """
    rows = []
    for a, b in itertools.combinations(matrix.columns, 2):
        diff = matrix[a].to_numpy() - matrix[b].to_numpy()
        if not higher_is_better:
            diff = -diff
        wins, ties = int((diff > 0).sum()), int((diff == 0).sum())
        tied = bool(np.all(diff == 0))
        p = 1.0 if tied else float(stats.wilcoxon(diff, zero_method="zsplit").pvalue)
        rows.append(
            {
                "A": a,
                "B": b,
                "победы A": wins,
                "ничьи": ties,
                "поражения A": len(diff) - wins - ties,
                "медиана разности": float(np.median(diff)),
                "p": p,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p (Холм)"] = holm(table["p"].tolist())
    table["различие"] = np.where(
        table["p (Холм)"] < alpha,
        np.where(table["медиана разности"] >= 0, "A лучше", "B лучше"),
        "не обнаружено",
    )
    return table

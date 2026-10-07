"""Анализ результатов бенчмарка: чувствительность, правила отказа, согласие ранжирований.

Ответственный: Гарифзянов Т. Р.

* :func:`scenario_level` — средние по повторам: строка — пара (сценарий, детектор).
  Повторы одного сценария не считаются независимыми наблюдениями.
* :func:`spearman_table` — ранговая корреляция Спирмена метрики с каждым
  параметром сценария для каждого детектора, с поправкой Холма по
  параметрам внутри детектора.
* :func:`failure_rules` — интерпретируемые правила отказа: дерево решений
  ограниченной глубины предсказывает отказ (например, recall < 0,5) по
  параметрам сценария, листья с преобладанием отказов выписываются как
  конъюнкции условий. Надёжность правил оценивается кросс-валидацией и
  сравнивается с долей большинства.
* :func:`rank_agreement` — согласие двух ранжирований (τ Кендалла [Kendall, 1938]
  и ρ Спирмена).
* :func:`monotone_share` — доля шагов, на которых кривая не возрастает.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from scendrift.evaluation.stats import holm

__all__ = [
    "PARAM_LABELS",
    "scenario_level",
    "spearman_table",
    "FailureRules",
    "failure_rules",
    "rank_agreement",
    "monotone_share",
]

#: Обозначения параметров сценария в таблицах и правилах.
PARAM_LABELS: dict[str, str] = {
    "magnitude": "m",
    "width": "ℓ",
    "alpha": "α",
    "pi0": "π₀",
    "eta": "η",
    "d": "d",
    "K": "K",
    "n": "n",
    "kind": "вид",
    "form": "форма",
    "recurring": "возврат",
}


def scenario_level(
    frame: pd.DataFrame, metrics: Sequence[str], keep: Sequence[str] = ()
) -> pd.DataFrame:
    """Средние метрик по повторам для каждой пары (сценарий, детектор).

    Args:
        frame: таблица прогонов (:func:`~scendrift.evaluation.results.runs_to_frame`).
        metrics: усредняемые метрики.
        keep: столбцы-описатели сценария, которые переносятся без изменений.
    """
    agg = frame.groupby(["scenario_id", "detector"])[list(metrics)].mean()
    if keep:
        desc = frame.groupby("scenario_id")[list(keep)].first()
        agg = agg.join(desc, on="scenario_id")
    return agg.reset_index()


def spearman_table(
    frame: pd.DataFrame, metric: str, params: Sequence[str], *, by: str = "detector"
) -> pd.DataFrame:
    """ρ Спирмена между метрикой и параметрами сценария для каждой группы.

    Ожидается таблица уровня сценариев (:func:`scenario_level`). Поправка
    Холма применяется к параметрам внутри группы.

    Returns:
        Таблица: группа, параметр, ρ, p, p (Холм), n.
    """
    rows = []
    for group, part in frame.groupby(by, sort=False):
        block = []
        for param in params:
            sub = part[[metric, param]].dropna()
            if sub[param].nunique() < 2 or sub[metric].nunique() < 2:
                rho, p = np.nan, np.nan
            else:
                rho, p = stats.spearmanr(sub[metric], sub[param])
            block.append(
                {by: group, "параметр": param, "ρ": float(rho), "p": float(p), "n": len(sub)}
            )
        valid = [r["p"] for r in block if not np.isnan(r["p"])]
        adjusted = iter(holm(valid)) if valid else iter(())
        for r in block:
            r["p (Холм)"] = np.nan if np.isnan(r["p"]) else next(adjusted)
        rows += block
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class FailureRules:
    """Правила отказа и их надёжность.

    Attributes:
        rules: листья с преобладанием отказов: условие, число сценариев,
            доля отказов среди них, доля всех отказов, покрытых правилом.
        failure_share: доля отказов во всей выборке.
        cv_accuracy: точность дерева при 5-кратной кросс-валидации (NaN, если
            отказов или успехов меньше пяти).
        baseline_accuracy: точность константного ответа «класс большинства».
        tree: обученное дерево.
        features: имена признаков дерева.
    """

    rules: pd.DataFrame
    failure_share: float
    cv_accuracy: float
    baseline_accuracy: float
    tree: DecisionTreeClassifier
    features: tuple[str, ...]


def _fmt(x: float) -> str:
    return f"{x:.3g}".replace(".", ",")


def _condition(feature: str, threshold: float, left: bool) -> str:
    if "=" in feature:  # one-hot: kind=virtual
        name, value = feature.split("=", 1)
        label = PARAM_LABELS.get(name, name)
        return f"{label} ≠ {value}" if left else f"{label} = {value}"
    label = PARAM_LABELS.get(feature, feature)
    return f"{label} ≤ {_fmt(threshold)}" if left else f"{label} > {_fmt(threshold)}"


def failure_rules(
    data: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    max_depth: int = 3,
    min_samples_leaf: int = 6,
    seed: int = 0,
) -> FailureRules:
    """Правила отказа по дереву решений.

    Args:
        data: таблица уровня сценариев.
        features: параметры сценария; нечисловые кодируются one-hot (``kind=virtual``).
        target: булев столбец «отказ».
        max_depth: глубина дерева (правило — не более max_depth условий).
        min_samples_leaf: минимальное число сценариев в листе.
        seed: seed дерева и разбиения кросс-валидации.
    """
    X = pd.get_dummies(data[list(features)], prefix_sep="=", dtype=float)
    X = X.astype(float)
    y = data[target].astype(bool).to_numpy()
    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed
    ).fit(X, y)
    n_fail, n_ok = int(y.sum()), int((~y).sum())
    if min(n_fail, n_ok) >= 5:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        clf = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed
        )
        cv_acc = float(cross_val_score(clf, X, y, cv=cv).mean())
    else:
        cv_acc = float("nan")
    t = tree.tree_
    names = list(X.columns)
    fail_index = list(tree.classes_).index(True) if True in tree.classes_ else None
    rules = []

    def walk(node: int, conditions: list[str]) -> None:
        if t.children_left[node] == -1:
            counts = t.value[node][0]
            total = int(t.n_node_samples[node])
            share = float(counts[fail_index] / counts.sum()) if fail_index is not None else 0.0
            if share > 0.5:
                rules.append(
                    {
                        "правило": " и ".join(conditions) or "всегда",
                        "сценариев": total,
                        "доля отказов": share,
                        "покрыто отказов": share * total / n_fail if n_fail else 0.0,
                    }
                )
            return
        f, thr = names[t.feature[node]], float(t.threshold[node])
        walk(t.children_left[node], [*conditions, _condition(f, thr, True)])
        walk(t.children_right[node], [*conditions, _condition(f, thr, False)])

    walk(0, [])
    table = pd.DataFrame(rules, columns=["правило", "сценариев", "доля отказов", "покрыто отказов"])
    table = table.sort_values("покрыто отказов", ascending=False, ignore_index=True)
    return FailureRules(
        rules=table,
        failure_share=float(y.mean()),
        cv_accuracy=cv_acc,
        baseline_accuracy=float(max(y.mean(), 1 - y.mean())),
        tree=tree,
        features=tuple(names),
    )


def rank_agreement(a: pd.Series, b: pd.Series) -> dict[str, float]:
    """Согласие двух ранжирований на общих методах: τ Кендалла, его p и ρ Спирмена."""
    common = a.index.intersection(b.index)
    tau, p = stats.kendalltau(a[common], b[common])
    rho, _ = stats.spearmanr(a[common], b[common])
    return {"tau": float(tau), "p": float(p), "rho": float(rho), "n": len(common)}


def monotone_share(values: Sequence[float]) -> float:
    """Доля шагов, на которых последовательность не возрастает."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float("nan")
    return float(np.mean(np.diff(v) <= 0))

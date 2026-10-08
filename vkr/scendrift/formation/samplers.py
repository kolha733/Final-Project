"""Планы вычислительного эксперимента над пространством параметров сценариев.

Ответственный: Будаев К. В.

План — конечное множество точек U = {u₁, …, u_N} единичного куба [0, 1)ᴰ,
где D — число варьируемых параметров (:meth:`ParameterSpace.free_names`).
Точка куба переводится в точку пространства параметров
(:meth:`ParameterSpace.from_unit`), а затем в сценарий
(:mod:`scendrift.formation.mapping`).

Реализованы четыре плана с одинаковым интерфейсом и бюджетом N:

* :class:`GridSampler` — полный факторный план (сетка). При бюджете N
  берётся наибольшее число уровней L, при котором число узлов не превышает
  N. Категориальные параметры перебираются полностью, поэтому при большой
  размерности сетка вырождается: на каждую числовую ось приходится один
  уровень;
* :class:`RandomSampler` — независимая равномерная выборка;
* :class:`LHSSampler` — латинский гиперкуб [McKay, Beckman, Conover, 1979]:
  проекция на любую ось содержит ровно одну точку в каждом из N слоёв;
* :class:`SobolSampler` — скремблированная последовательность Соболя
  (квази-Монте-Карло): малое расхождение по всем осям одновременно;
  баланс гарантируется при N = 2ᵏ.

LHS и Соболь берутся из ``scipy.stats.qmc`` (заимствованный компонент).
Собственная часть модуля — единый интерфейс планов над пространством с
условными параметрами и метрики равномерности (:func:`design_metrics`),
по которым планы сравниваются в п. 4.6.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import qmc

from scendrift.scenario.space import Categorical, IntDomain, ParameterSpace

__all__ = [
    "Sampler",
    "GridSampler",
    "RandomSampler",
    "LHSSampler",
    "SobolSampler",
    "SAMPLERS",
    "make_sampler",
    "grid_levels",
    "design_metrics",
]


class Sampler:
    """Базовый класс плана: точки единичного куба и точки пространства параметров.

    Реализует протокол :class:`~scendrift.interfaces.ScenarioSampler`.
    """

    name: str = ""

    def design(self, space: ParameterSpace, n: int, seed: int) -> np.ndarray:
        """Точки плана в единичном кубе, массив N' × D (N' ≤ n для сетки)."""
        raise NotImplementedError

    def sample(self, space: ParameterSpace, n: int, seed: int) -> list[dict[str, Any]]:
        """Точки пространства параметров (протокол ``ScenarioSampler``)."""
        return [space.from_unit(u) for u in self.design(space, n, seed)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def _cardinality(space: ParameterSpace, name: str) -> int | None:
    """Число значений параметра (None — непрерывный домен)."""
    domain = space[name].domain
    if isinstance(domain, Categorical):
        return len(domain.values)
    if isinstance(domain, IntDomain):
        return domain.high - domain.low + 1
    return None


def grid_levels(space: ParameterSpace, n: int) -> dict[str, int]:
    """Число уровней по каждой оси сетки при бюджете n.

    Категориальные оси берут все значения. Числовые оси получают
    одинаковое число уровней L (но не больше числа значений целочисленного
    домена); L — наибольшее, при котором число узлов не превышает n.

    Raises:
        ValueError: если даже при L = 1 узлов больше n.
    """
    free = space.free_names()
    card = {name: _cardinality(space, name) for name in free}

    def levels(big_l: int) -> dict[str, int]:
        out = {}
        for name in free:
            c = card[name]
            if isinstance(space[name].domain, Categorical):
                out[name] = int(c)  # type: ignore[arg-type]
            else:
                out[name] = big_l if c is None else min(big_l, c)
        return out

    def size(lv: dict[str, int]) -> int:
        return math.prod(lv.values())

    if size(levels(1)) > n:
        raise ValueError(
            f"сетке нужно не менее {size(levels(1))} узлов (все сочетания категориальных "
            f"параметров), а бюджет {n}; зафиксируйте часть параметров"
        )
    big_l = 1
    while size(levels(big_l + 1)) <= n and levels(big_l + 1) != levels(big_l):
        big_l += 1
    return levels(big_l)


class GridSampler(Sampler):
    """Полный факторный план: узлы — середины ячеек равномерного разбиения осей."""

    name = "grid"

    def design(self, space: ParameterSpace, n: int, seed: int = 0) -> np.ndarray:
        lv = grid_levels(space, n)
        axes = [(np.arange(k) + 0.5) / k for k in lv.values()]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=1)

    def sample(self, space: ParameterSpace, n: int, seed: int = 0) -> list[dict[str, Any]]:
        """Узлы сетки без повторов (неактивные параметры дают совпадающие точки)."""
        out, seen = [], set()
        for u in self.design(space, n, seed):
            point = space.from_unit(u)
            key = tuple(sorted((k, repr(v)) for k, v in point.items()))
            if key not in seen:
                seen.add(key)
                out.append(point)
        return out


class RandomSampler(Sampler):
    """Независимая равномерная выборка из единичного куба."""

    name = "random"

    def design(self, space: ParameterSpace, n: int, seed: int) -> np.ndarray:
        return np.random.default_rng(seed).random((n, len(space.free_names())))


class LHSSampler(Sampler):
    """Латинский гиперкуб (``scipy.stats.qmc.LatinHypercube``).

    Args:
        optimization: ``None`` — классический LHS; ``"random-cd"`` —
            перестановки, уменьшающие центрированное L2-расхождение.
    """

    name = "lhs"

    def __init__(self, optimization: str | None = None) -> None:
        self.optimization = optimization

    def design(self, space: ParameterSpace, n: int, seed: int) -> np.ndarray:
        engine = qmc.LatinHypercube(
            len(space.free_names()), optimization=self.optimization, rng=seed
        )
        return engine.random(n)

    def __repr__(self) -> str:
        return f"LHSSampler(optimization={self.optimization!r})"


class SobolSampler(Sampler):
    """Скремблированная последовательность Соболя (``scipy.stats.qmc.Sobol``).

    При n = 2ᵏ берётся сбалансированный отрезок последовательности; при
    другом n — первые n точек (предупреждение SciPy о нарушении баланса
    подавляется, а свойство отмечается в документации набора).
    """

    name = "sobol"

    def design(self, space: ParameterSpace, n: int, seed: int) -> np.ndarray:
        engine = qmc.Sobol(len(space.free_names()), scramble=True, rng=seed)
        if n > 0 and n & (n - 1) == 0:
            return engine.random_base2(int(math.log2(n)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return engine.random(n)


#: Реестр планов по имени.
SAMPLERS: dict[str, type[Sampler]] = {
    "grid": GridSampler,
    "random": RandomSampler,
    "lhs": LHSSampler,
    "sobol": SobolSampler,
}


def make_sampler(method: str, **options: Any) -> Sampler:
    """План по имени: grid, random, lhs, sobol.

    Raises:
        KeyError: если план неизвестен.
    """
    if method not in SAMPLERS:
        raise KeyError(f"неизвестный план {method!r}; доступны: {', '.join(SAMPLERS)}")
    return SAMPLERS[method](**options)


def design_metrics(U: np.ndarray) -> dict[str, float]:
    """Метрики равномерности плана в единичном кубе.

    * ``cd`` — центрированное L2-расхождение (меньше — равномернее);
    * ``maximin`` — наименьшее попарное расстояние (больше — точки не
      «слипаются»);
    * ``strata`` — средняя по осям доля занятых слоёв при разбиении оси на
      N равных слоёв (у латинского гиперкуба равна 1).

    Args:
        U: точки плана, N × D, N ≥ 2.
    """
    U = np.asarray(U, dtype=float)
    n = len(U)
    if n < 2:
        raise ValueError("для метрик нужно не менее двух точек")
    U = np.clip(U, 0.0, np.nextafter(1.0, 0.0))
    strata = float(np.mean([len(np.unique(np.floor(U[:, j] * n))) / n for j in range(U.shape[1])]))
    return {
        "cd": float(qmc.discrepancy(U, method="CD")),
        "maximin": float(pdist(U).min()),
        "strata": strata,
    }

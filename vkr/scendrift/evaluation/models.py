"""Базовая онлайн-модель h: векторизованный prequential гауссовский наивный Байес.

Ответственный: Гарифзянов Т. Р.

Протокол оценки (п. 2.6) требует prequential-ошибки базовой модели:
в момент t модель, обученная на объектах [t₀, t) после последнего сброса
t₀, предсказывает метку x_t, затем обучается на (x_t, y_t). Детектор
получает индикатор ошибки e_t = 1[ŷ_t ≠ y_t].

Модель повторяет математику ``river.naive_bayes.GaussianNB`` (river 0.26.1):

* априорная вероятность класса — доля его объектов;
* для каждого класса и признака — нормальная плотность с выборочным средним
  и **популяционной** дисперсией (ddof = 0); при нулевой дисперсии
  плотность считается равной 0;
* log-правдоподобие класса: log P(c) + Σ_f log(10⁻⁹ + pdf_cf(x_f));
* предсказание — argmax; при равенстве выигрывает класс, встреченный
  первым после сброса; пока модель не видела ни одного объекта, ответа
  нет, и это считается ошибкой.

Отличие только в способе вычисления. В river модель обучается по одному
объекту в цикле Python. Здесь достаточные статистики (число объектов,
суммы и суммы квадратов по классам) считаются кумулятивными суммами
numpy сразу для блока объектов. Поэтому предсказания для всего отрезка
после сброса вычисляются векторно, а цикл Python остаётся только у
детектора. Эквивалентность с river проверяется тестом (п. 6) и в п. 4.8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["PDF_EPS", "NBState", "PrequentialGaussianNB", "MODELS", "make_model"]

#: Добавка к плотности, как в river (``_PDF_EPS = 10e-10``).
PDF_EPS = 10e-10
_LOG_PDF_EPS = math.log(PDF_EPS)
_VAR_TOL = 1e-12


@dataclass
class NBState:
    """Достаточные статистики модели с момента последнего сброса.

    Attributes:
        count: число объектов каждого класса (2).
        total: сумма признаков по классам (2 × d).
        squares: сумма квадратов признаков по классам (2 × d).
        first: класс, встреченный первым после сброса (−1 — объектов не было).
    """

    count: np.ndarray
    total: np.ndarray
    squares: np.ndarray
    first: int = -1

    @classmethod
    def empty(cls, n_features: int) -> NBState:
        """Состояние только что сброшенной модели."""
        zeros = np.zeros((2, n_features))
        return cls(np.zeros(2), zeros.copy(), zeros.copy(), -1)


@dataclass
class PrequentialGaussianNB:
    """Prequential-ошибки гауссовского наивного Байеса на заданном потоке.

    Args:
        X: признаки, n × d.
        y: метки 0/1, n.
    """

    X: np.ndarray
    y: np.ndarray
    _X: np.ndarray = field(init=False, repr=False)
    _y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._X = np.asarray(self.X, dtype=float)
        self._y = np.asarray(self.y).astype(np.int64)

    @property
    def n_features(self) -> int:
        return int(self._X.shape[1])

    def empty_state(self) -> NBState:
        """Состояние только что сброшенной модели."""
        return NBState.empty(self.n_features)

    def block(self, start: int, stop: int, state: NBState) -> tuple[np.ndarray, NBState]:
        """Ошибки на объектах [start, stop) и состояние модели после них.

        Args:
            start: первый объект блока.
            stop: конец блока (исключительно).
            state: статистики модели по объектам до ``start`` (после сброса).

        Returns:
            Индикаторы ошибок (bool, длина stop − start) и новое состояние.
        """
        X, y = self._X[start:stop], self._y[start:stop]
        m = len(y)
        onehot = np.stack([y == 0, y == 1], axis=1).astype(float)  # m × 2
        # Статистики «до объекта t» (исключительные кумулятивные суммы).
        cnt = state.count + np.cumsum(onehot, axis=0) - onehot  # m × 2
        tot = state.total[None] + np.cumsum(onehot[:, :, None] * X[:, None, :], axis=0)
        tot -= onehot[:, :, None] * X[:, None, :]
        sq = state.squares[None] + np.cumsum(onehot[:, :, None] * (X**2)[:, None, :], axis=0)
        sq -= onehot[:, :, None] * (X**2)[:, None, :]

        seen = cnt > 0
        safe = np.where(seen, cnt, 1.0)
        mean = tot / safe[:, :, None]
        var = sq / safe[:, :, None] - mean**2
        positive = var > _VAR_TOL * np.maximum(1.0, mean**2)
        safe_var = np.where(positive, var, 1.0)
        with np.errstate(over="ignore", under="ignore"):
            pdf = np.exp(-((X[:, None, :] - mean) ** 2) / (2.0 * safe_var))
            pdf /= np.sqrt(2.0 * math.pi * safe_var)
        log_pdf = np.where(positive, np.log(PDF_EPS + np.where(positive, pdf, 0.0)), _LOG_PDF_EPS)
        n_seen = cnt.sum(axis=1)
        prior = np.log(np.where(seen, cnt, 1.0) / np.where(n_seen > 0, n_seen, 1.0)[:, None])
        jll = np.where(seen, prior + log_pdf.sum(axis=2), -np.inf)

        # Порядок классов для разрешения равенства: первый встреченный после сброса.
        first = state.first
        if first < 0 and m:
            first = int(y[0])
        first_t = np.full(m, first)
        if state.first < 0 and m:
            first_t[0] = -1  # до первого объекта модель пуста
        other = 1 - np.maximum(first_t, 0)
        pred = np.where(
            jll[np.arange(m), other] > jll[np.arange(m), np.maximum(first_t, 0)],
            other,
            np.maximum(first_t, 0),
        )
        errors = (pred != y) | (n_seen == 0)

        new = NBState(
            state.count + onehot.sum(axis=0),
            state.total + (onehot[:, :, None] * X[:, None, :]).sum(axis=0),
            state.squares + (onehot[:, :, None] * (X**2)[:, None, :]).sum(axis=0),
            first,
        )
        return errors, new

    def errors(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Ошибки модели, сброшенной в момент ``start``, на объектах [start, stop)."""
        stop = len(self._y) if stop is None else stop
        err, _ = self.block(start, stop, self.empty_state())
        return err


#: Реестр базовых моделей протокола оценки (поле ``EvaluationSpec.base_model``).
MODELS: dict[str, type[PrequentialGaussianNB]] = {"gaussian_nb": PrequentialGaussianNB}


def make_model(name: str, X: np.ndarray, y: np.ndarray) -> PrequentialGaussianNB:
    """Базовая модель по имени из реестра.

    Raises:
        KeyError: если модель не зарегистрирована.
    """
    if name not in MODELS:
        raise KeyError(f"неизвестная базовая модель {name!r}; доступны: {', '.join(MODELS)}")
    return MODELS[name](X, y)

"""Общие типы данных и интерфейсы модулей фреймворка.

Ответственные: Будаев К. В. (данные, генераторы, формирование сценариев) и
Гарифзянов Т. Р. (детекторы, метрики, раннер).

Контракты между модулями заданы через :class:`typing.Protocol`
(структурная типизация). Поэтому, например, детектор из любой библиотеки
подключается адаптером, без наследования от классов фреймворка.

Поток данных между модулями::

    ScenarioSpec ──generate──▶ StreamData(X, y, GroundTruth)
         │                          │
         └──────── Runner ◀─────────┘ ── DriftDetector, Metric ──▶ DetectionRun
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from scendrift.evaluation.protocol import EvaluationSpec
    from scendrift.scenario.schema import ScenarioSpec
    from scendrift.scenario.space import ParameterSpace

__all__ = [
    "DriftInterval",
    "GroundTruth",
    "StreamData",
    "StreamGenerator",
    "ScenarioSampler",
    "DriftDetector",
    "Metric",
    "DetectionRun",
]


# ---------------------------------------------------------------------------
# Данные (Будаев К. В.)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftInterval:
    """Истинное событие дрейфа в ground truth.

    Attributes:
        index: номер события (с нуля).
        onset: начало перехода (индекс первого объекта, затронутого дрейфом).
        center: центр перехода τ.
        end: конец перехода (исключительно); при внезапном дрейфе end = onset.
        form: форма перехода.
        kind: вид события или ``"recurring"``.
        magnitude: заданная величина (None для повторяющегося события).
        realized: фактическая величина по компонентам real/virtual/prior.
        returns_to: номер концепта возврата.
    """

    index: int
    onset: int
    center: int
    end: int
    form: str
    kind: str
    magnitude: float | None
    realized: Mapping[str, float] = field(default_factory=dict)
    returns_to: int | None = None

    def acceptance_window(self, delta: int) -> tuple[int, int]:
        """Полуинтервал [onset, end + Δ), на котором срабатывание засчитывается."""
        return self.onset, self.end + delta


@dataclass(frozen=True)
class GroundTruth:
    """Истинная разметка дрейфа потока: упорядоченный набор интервалов."""

    intervals: tuple[DriftInterval, ...]
    n_samples: int

    def __len__(self) -> int:
        return len(self.intervals)

    @property
    def onsets(self) -> np.ndarray:
        """Начала переходов."""
        return np.array([iv.onset for iv in self.intervals], dtype=np.int64)

    def drift_mask(self) -> np.ndarray:
        """Булев вектор длины n: объект находится внутри интервала перехода.

        При внезапном дрейфе отмечается ровно один объект (onset).
        """
        mask = np.zeros(self.n_samples, dtype=bool)
        for iv in self.intervals:
            mask[iv.onset : max(iv.end, iv.onset + 1)] = True
        return mask

    def to_records(self) -> list[dict[str, Any]]:
        """Список словарей (для pandas.DataFrame и экспорта в CSV)."""
        rows = []
        for iv in self.intervals:
            row = {
                "index": iv.index,
                "onset": iv.onset,
                "center": iv.center,
                "end": iv.end,
                "form": iv.form,
                "kind": iv.kind,
                "magnitude": iv.magnitude,
                "returns_to": iv.returns_to,
            }
            row.update({f"realized_{k}": v for k, v in iv.realized.items()})
            rows.append(row)
        return rows


@dataclass(frozen=True)
class StreamData:
    """Реализация потока по сценарию.

    Attributes:
        X: матрица признаков n × d.
        y: метки (с шумом), n.
        ground_truth: истинная разметка дрейфа.
        concept: номер концепта, породившего объект (латентная переменная), n.
        y_clean: метки без шума (для валидации генератора), n.
        scenario_id: идентификатор сценария.
        seed: seed реализации.
    """

    X: np.ndarray
    y: np.ndarray
    ground_truth: GroundTruth
    concept: np.ndarray
    y_clean: np.ndarray
    scenario_id: str
    seed: int

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


@runtime_checkable
class StreamGenerator(Protocol):
    """Генератор потока семейства ``family``: детерминирован при фиксированных (S, seed)."""

    family: str

    def generate(self, spec: ScenarioSpec, seed: int | None = None) -> StreamData:
        """Порождает поток. Если seed не задан, берётся ``spec.seed``."""
        ...


@runtime_checkable
class ScenarioSampler(Protocol):
    """План эксперимента над пространством параметров (этап 3)."""

    def sample(self, space: ParameterSpace, n: int, seed: int) -> list[dict[str, Any]]:
        """Возвращает n точек пространства параметров."""
        ...


# ---------------------------------------------------------------------------
# Оценка (Гарифзянов Т. Р.)
# ---------------------------------------------------------------------------


@runtime_checkable
class DriftDetector(Protocol):
    """Детектор дрейфа: получает поток значений (обычно индикатор ошибки 0/1)."""

    name: str

    def update(self, value: float) -> bool:
        """Обрабатывает очередное значение и возвращает True при срабатывании."""
        ...

    def reset(self) -> None:
        """Возвращает детектор в исходное состояние."""
        ...

    def clone(self) -> DriftDetector:
        """Новый экземпляр с теми же гиперпараметрами в исходном состоянии."""
        ...


@dataclass(frozen=True)
class DetectionRun:
    """Результат одного прогона: сценарий × seed × детектор.

    Attributes:
        scenario_id: идентификатор сценария.
        seed: seed реализации.
        detector: имя детектора (с гиперпараметрами).
        detections: моменты срабатываний (индексы объектов).
        n_samples: длина потока.
        accuracy: prequential accuracy базовой модели.
        runtime_s: время прогона, с.
        extra: дополнительные измерения.
    """

    scenario_id: str
    seed: int
    detector: str
    detections: tuple[int, ...]
    n_samples: int
    accuracy: float
    runtime_s: float
    extra: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Metric(Protocol):
    """Метрика качества обнаружения по срабатываниям и ground truth."""

    name: str

    def __call__(
        self,
        detections: Sequence[int],
        ground_truth: GroundTruth,
        protocol: EvaluationSpec,
    ) -> float:
        """Значение метрики (NaN, если метрика не определена)."""
        ...

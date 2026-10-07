"""Метрики обнаружения дрейфа по срабатываниям и ground truth (п. 2.6).

Ответственный: Гарифзянов Т. Р.

Правило сопоставления. Окно допуска события k — Λ_k = [onset_k, end_k + Δ).
Окна не пересекаются (ограничение C6), поэтому каждое срабатывание
относится не более чем к одному событию:

* первое срабатывание в Λ_k — обнаружение (TP) с моментом t*_k;
* следующие срабатывания в Λ_k — избыточные FP_red;
* срабатывания вне всех окон — ложные тревоги FP_out;
* событие без срабатываний в окне — пропуск (FN).

Задержка: при r = onset δ_k = t*_k − onset_k + 1 ≥ 1; при r = center
δ_k = t*_k − τ_k (может быть отрицательной).

Метрики (определения — в п. 2.6): precision = TP/|D̂|, recall = TP/K,
F1, MDR = 1 − recall, MTD — средняя задержка, медианная задержка,
MTFA = T_stable / FP_out, MTR = MTFA/MTD·(1 − MDR) (только при r = onset),
FAR = 1000·FP_out / T_stable, где T_stable = (n − W) − Σ|Λ_k|.

Соглашения для вырожденных случаев (п. 2.6): при |D̂| = 0 precision не
определена (NaN); при TP = 0 F1 = 0, а MTD и MTR не определены; без
ложных тревог MTFA = MTR = +∞, а их «ограниченные» варианты используют
верхнюю границу T_stable. Если событий нет (K = 0), recall, MDR и F1 не
определены.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from scendrift.evaluation.protocol import DelayReference, EvaluationSpec
from scendrift.interfaces import GroundTruth

__all__ = [
    "MatchResult",
    "match_detections",
    "detection_metrics",
    "METRIC_NAMES",
    "HIGHER_IS_BETTER",
    "MetricFunction",
    "METRICS",
]


@dataclass(frozen=True)
class MatchResult:
    """Результат сопоставления срабатываний с событиями.

    Attributes:
        detections: учтённые срабатывания (t ≥ W), по возрастанию.
        hits: момент обнаружения t*_k каждого события (None — пропуск).
        delays: задержка обнаружения каждого события (None — пропуск).
        false_alarms: срабатывания вне окон допуска (FP_out).
        redundant: повторные срабатывания внутри окон (FP_red).
        t_stable: длина стабильной части потока T_stable.
    """

    detections: tuple[int, ...]
    hits: tuple[int | None, ...]
    delays: tuple[int | None, ...]
    false_alarms: tuple[int, ...]
    redundant: tuple[int, ...]
    t_stable: int

    @property
    def tp(self) -> int:
        return sum(h is not None for h in self.hits)

    @property
    def fn(self) -> int:
        return sum(h is None for h in self.hits)


def match_detections(
    detections: Sequence[int], ground_truth: GroundTruth, protocol: EvaluationSpec
) -> MatchResult:
    """Сопоставляет срабатывания с событиями по окнам допуска."""
    delta, warmup = protocol.acceptance_window, protocol.warmup
    dets = tuple(sorted(int(t) for t in detections if t >= warmup))
    windows = [iv.acceptance_window(delta) for iv in ground_truth.intervals]
    hits: list[int | None] = [None] * len(windows)
    false_alarms, redundant = [], []
    starts = np.array([w[0] for w in windows], dtype=np.int64)
    for t in dets:
        k = int(np.searchsorted(starts, t, side="right")) - 1
        if k >= 0 and t < windows[k][1]:
            if hits[k] is None:
                hits[k] = t
            else:
                redundant.append(t)
        else:
            false_alarms.append(t)
    delays: list[int | None] = []
    for iv, hit in zip(ground_truth.intervals, hits, strict=True):
        if hit is None:
            delays.append(None)
        elif DelayReference(protocol.delay_reference) is DelayReference.ONSET:
            delays.append(hit - iv.onset + 1)
        else:
            delays.append(hit - iv.center)
    covered = sum(end - start for start, end in windows)
    t_stable = (ground_truth.n_samples - warmup) - covered
    return MatchResult(
        dets, tuple(hits), tuple(delays), tuple(false_alarms), tuple(redundant), int(t_stable)
    )


def _safe_div(a: float, b: float) -> float:
    return a / b if b else math.nan


def detection_metrics(
    detections: Sequence[int], ground_truth: GroundTruth, protocol: EvaluationSpec
) -> dict[str, float]:
    """Все метрики обнаружения одного прогона (см. описание модуля)."""
    m = match_detections(detections, ground_truth, protocol)
    k_events = len(ground_truth.intervals)
    n_det = len(m.detections)
    tp, fn, fp_out = m.tp, m.fn, len(m.false_alarms)
    precision = _safe_div(tp, n_det)
    recall = _safe_div(tp, k_events)
    if k_events == 0:
        f1 = math.nan
    elif tp == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    delays = [d for d in m.delays if d is not None]
    mtd = float(np.mean(delays)) if delays else math.nan
    median_delay = float(np.median(delays)) if delays else math.nan
    t_stable = max(m.t_stable, 0)
    mtfa = t_stable / fp_out if fp_out else math.inf
    mtfa_capped = min(mtfa, float(t_stable))
    mdr = 1.0 - recall if k_events else math.nan
    onset_ref = DelayReference(protocol.delay_reference) is DelayReference.ONSET
    if onset_ref and delays and mtd > 0:
        mtr = mtfa / mtd * (1.0 - mdr)
        mtr_capped = mtfa_capped / mtd * (1.0 - mdr)
    else:
        mtr = mtr_capped = math.nan
    return {
        "n_events": float(k_events),
        "n_detections": float(n_det),
        "tp": float(tp),
        "fn": float(fn),
        "fp_out": float(fp_out),
        "fp_red": float(len(m.redundant)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mdr": mdr,
        "mtd": mtd,
        "median_delay": median_delay,
        "mtfa": mtfa,
        "mtfa_capped": mtfa_capped,
        "mtr": mtr,
        "mtr_capped": mtr_capped,
        "far": 1000.0 * fp_out / t_stable if t_stable else math.nan,
        "t_stable": float(t_stable),
    }


#: Метрики, возвращаемые :func:`detection_metrics`.
METRIC_NAMES = tuple(
    detection_metrics((), GroundTruth((), 10), EvaluationSpec(warmup=0, acceptance_window=1))
)

#: Направление «лучше» для метрик, по которым строятся ранги.
HIGHER_IS_BETTER: dict[str, bool] = {
    "precision": True,
    "recall": True,
    "f1": True,
    "mdr": False,
    "mtd": False,
    "median_delay": False,
    "mtfa": True,
    "mtfa_capped": True,
    "mtr": True,
    "mtr_capped": True,
    "far": False,
    "fp_out": False,
    "accuracy": True,
    "delta_accuracy": True,
}


@dataclass(frozen=True)
class MetricFunction:
    """Одна метрика в форме протокола :class:`~scendrift.interfaces.Metric`."""

    name: str

    def __call__(
        self, detections: Sequence[int], ground_truth: GroundTruth, protocol: EvaluationSpec
    ) -> float:
        return detection_metrics(detections, ground_truth, protocol)[self.name]


#: Метрики по имени.
METRICS: dict[str, MetricFunction] = {name: MetricFunction(name) for name in METRIC_NAMES}

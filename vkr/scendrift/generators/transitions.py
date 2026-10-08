"""Функции перехода p(t) и временная разметка потока.

Ответственный: Будаев К. В.

Для каждого объекта t временная разметка задаёт тройку
(src_t, dst_t, p_t): переход идёт из состояния цепочки концептов src в
состояние dst, а p_t ∈ [0, 1] — значение функции перехода. Вне интервалов
перехода p_t ∈ {0, 1}, и объект целиком принадлежит одному концепту.

Функции перехода на полуинтервале [onset, end) длины ℓ (центр — середина
объекта, отсюда сдвиг 0,5):

* линейная: p(t) = (t − onset + 0,5)/ℓ;
* сигмоидная (как ConceptDriftStream в MOA): σ(t) = 1/(1 + exp(−s(t − τ)/ℓ)),
  s = 2·ln((1 − ε)/ε), ε = 0,01. Она отнормирована так, что p = 0 в onset
  и p = 1 в end, поэтому переход целиком лежит в интервале ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scendrift.scenario.enums import TransitionShape
from scendrift.scenario.schema import DriftEvent, ScenarioSpec

__all__ = ["SIGMOID_EPS", "transition", "Timeline", "timeline"]

#: Доля «хвостов» сигмоиды, отрезаемых нормировкой.
SIGMOID_EPS = 0.01


def transition(t: np.ndarray, event: DriftEvent) -> np.ndarray:
    """Значения функции перехода события в моменты t (вне интервала — 0 или 1)."""
    t = np.asarray(t, dtype=float)
    if event.width == 0:
        return (t >= event.position).astype(float)
    onset, width = event.onset, event.width
    local = (t - onset + 0.5) / width
    if event.shape == TransitionShape.LINEAR:
        p = local
    else:
        steep = 2.0 * math.log((1.0 - SIGMOID_EPS) / SIGMOID_EPS)

        def sig(u: np.ndarray | float) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-steep * (np.asarray(u) - 0.5)))

        lo, hi = sig(0.0), sig(1.0)
        p = (sig(local) - lo) / (hi - lo)
    p = np.where(t < onset, 0.0, np.where(t >= event.end, 1.0, p))
    return np.clip(p, 0.0, 1.0)


@dataclass(frozen=True)
class Timeline:
    """Временная разметка потока.

    Attributes:
        src: номер исходного состояния цепочки концептов для каждого объекта.
        dst: номер конечного состояния (равен src вне переходов).
        p: значение функции перехода.
        event: номер события (с нуля), к которому относится объект; −1 — до первого.
    """

    src: np.ndarray
    dst: np.ndarray
    p: np.ndarray
    event: np.ndarray


def timeline(spec: ScenarioSpec) -> Timeline:
    """Строит временную разметку по событиям сценария.

    Состояние цепочки с номером k возникает после события k − 1 (нумерация
    событий с нуля): до первого события действует состояние 0, после
    завершения события e — состояние e + 1.
    """
    n = spec.stream.n_samples
    t = np.arange(n)
    src = np.zeros(n, dtype=np.int32)
    dst = np.zeros(n, dtype=np.int32)
    p = np.zeros(n, dtype=float)
    event = np.full(n, -1, dtype=np.int32)
    for e, ev in enumerate(spec.events):
        start = ev.onset
        stop = spec.events[e + 1].onset if e + 1 < len(spec.events) else n
        idx = t[start:stop]
        src[start:stop] = e
        dst[start:stop] = e + 1
        p[start:stop] = transition(idx, ev)
        event[start:stop] = e
    return Timeline(src=src, dst=dst, p=p, event=event)

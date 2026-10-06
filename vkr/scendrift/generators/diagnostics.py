"""Диагностика генератора: оценка фактических параметров дрейфа по данным (эксперимент Э1).

Ответственный: Будаев К. В.

Ground truth содержит величину, вычисленную по модели. Функции модуля
оценивают ту же величину независимо, **по порождённым данным**, и
сравнивают с заданной. Так проверяется, что генератор реализует
заявленную модель.

Оценки для семейства ``hyperplane_gauss`` (стабильные участки до и после
события):

* real — доля объектов участка «после», на которых правила разметки
  g_{k−1} и g_k расходятся (для поворота нормали это TV, утверждение 1);
* virtual — сдвиг средних проекций на направление v, пересчитанный в
  TV = 2Φ(δ̂/2) − 1;
* prior — |ȳ_после − ȳ_до| по чистым меткам.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
from scipy.special import ndtr

from scendrift.interfaces import StreamData
from scendrift.scenario.enums import DriftForm, DriftKind
from scendrift.scenario.schema import ScenarioSpec
from scendrift.scenario.semantics import (
    angle_for_rotation,
    build_chain,
    real_severity_from_angle,
)

__all__ = [
    "stable_before",
    "stable_after",
    "empirical_magnitude",
    "transition_curve",
    "digest",
]


def stable_before(spec: ScenarioSpec, k: int) -> slice:
    """Стабильный участок перед событием k (с нуля): [end_{k−1}, onset_k)."""
    start = spec.events[k - 1].end if k > 0 else 0
    return slice(start, spec.events[k].onset)


def stable_after(spec: ScenarioSpec, k: int) -> slice:
    """Стабильный участок после события k: [end_k, onset_{k+1})."""
    stop = spec.events[k + 1].onset if k + 1 < spec.n_events else spec.stream.n_samples
    return slice(spec.events[k].end, stop)


def empirical_magnitude(spec: ScenarioSpec, data: StreamData, k: int) -> tuple[float, float]:
    """Оценка величины события k по данным и её стандартная ошибка.

    Поддерживаются обычные события семейства ``hyperplane_gauss``.

    Returns:
        Пара (оценка, стандартная ошибка).
    """
    if spec.stream.family != "hyperplane_gauss" or spec.events[k].returns_to is not None:
        raise ValueError("оценка по данным реализована для обычных событий hyperplane_gauss")
    chain = build_chain(spec)
    before, after = stable_before(spec, k), stable_after(spec, k)
    kind = spec.events[k].kind
    if kind is DriftKind.REAL:
        X = data.X[after]
        flags = chain.states[k].label(X) != chain.states[k + 1].label(X)
        est = float(flags.mean())
        return est, math.sqrt(max(est * (1 - est), 1e-12) / len(flags))
    if kind is DriftKind.VIRTUAL:
        v = chain.geometry[k].direction
        a, b = data.X[before] @ v, data.X[after] @ v
        delta = float(b.mean() - a.mean())
        se_delta = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        est = float(2.0 * ndtr(abs(delta) / 2.0) - 1.0)
        dens = math.exp(-(delta**2) / 8.0) / math.sqrt(2.0 * math.pi)  # производная TV по δ
        return est, dens * se_delta
    ya, yb = data.y_clean[before], data.y_clean[after]
    est = float(abs(yb.mean() - ya.mean()))
    se = math.sqrt(ya.var() / len(ya) + yb.var() / len(yb))
    return est, se


def transition_curve(
    spec: ScenarioSpec, data: StreamData, k: int, window: int = 500
) -> pd.DataFrame:
    """Кривая перехода реального события k: эмпирическая и теоретическая.

    Эмпирическая кривая — скользящая доля объектов, у которых чистая метка
    расходится с правилом старого концепта g_{k−1}. Теоретическая кривая:

    * sudden и gradual — p(t)·m (смесь старого и нового концептов);
    * incremental — m(θ(t)), где θ(t) — угол при повороте на ψ·p(t).

    Эмпирическая кривая сглажена скользящим окном, поэтому сравнивать её
    нужно с теоретической кривой, сглаженной тем же окном (столбец
    ``theory_smoothed``). Иначе ступенька внезапного дрейфа даёт
    расхождение порядка m/2, которое вызвано только сглаживанием.

    Returns:
        Таблица со столбцами t, empirical, theory, theory_smoothed.
    """
    event = spec.events[k]
    if event.kind is not DriftKind.REAL or spec.stream.family != "hyperplane_gauss":
        raise ValueError("кривая перехода строится для реального дрейфа hyperplane_gauss")
    chain = build_chain(spec)
    old = chain.states[k]
    lo = max(0, event.onset - 2 * window)
    hi = min(spec.stream.n_samples, event.end + 2 * window)
    t = np.arange(lo, hi)
    mismatch = (old.label(data.X[lo:hi]) != data.y_clean[lo:hi]).astype(float)
    empirical = pd.Series(mismatch).rolling(window, center=True, min_periods=window // 2).mean()
    p = data.progress[lo:hi].astype(float) if data.progress is not None else np.zeros(hi - lo)
    m = float(event.magnitude)  # type: ignore[arg-type]
    if event.form is DriftForm.INCREMENTAL:
        geom = chain.geometry[k]
        mask = np.zeros_like(old.w)
        mask[geom.affected] = 1.0
        q = float((old.w * mask) @ (old.w * mask))
        theory = np.array(
            [
                real_severity_from_angle(
                    angle_for_rotation(geom.rotation * pi, q), old.pi0, old.prior
                )
                for pi in p
            ]
        )
    else:
        theory = p * m
    smoothed = pd.Series(theory).rolling(window, center=True, min_periods=window // 2).mean()
    return pd.DataFrame(
        {
            "t": t,
            "empirical": empirical.to_numpy(),
            "theory": theory,
            "theory_smoothed": smoothed.to_numpy(),
        }
    )


def digest(data: StreamData) -> str:
    """SHA-256 от признаков и меток потока (для проверки воспроизводимости)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(data.X).tobytes())
    h.update(np.ascontiguousarray(data.y).tobytes())
    return h.hexdigest()[:16]

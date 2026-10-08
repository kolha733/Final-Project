"""Построение ground truth — истинной разметки дрейфа.

Ответственный: Будаев К. В.

Ground truth строится по сценарию, а не по данным. Это возможно, потому что
позиции и интервалы событий заданы в сценарии, а фактическая величина
вычисляется семейством генератора (аналитически или на эталонной выборке).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from scendrift.interfaces import DriftInterval, GroundTruth
from scendrift.scenario.schema import ScenarioSpec

__all__ = ["build_ground_truth"]


def build_ground_truth(spec: ScenarioSpec, realized: Sequence[Mapping[str, float]]) -> GroundTruth:
    """Истинная разметка: интервалы событий с заданной и фактической величиной.

    Args:
        spec: сценарий.
        realized: фактические величины событий (по одному словарю на событие).

    Raises:
        ValueError: если число записей ``realized`` не совпадает с числом событий.
    """
    if len(realized) != spec.n_events:
        raise ValueError("нужна фактическая величина для каждого события")
    intervals = []
    for k, (event, real) in enumerate(zip(spec.events, realized, strict=True)):
        intervals.append(
            DriftInterval(
                index=k,
                onset=event.onset,
                center=event.position,
                end=event.end,
                form=event.form.value,
                kind="recurring" if event.returns_to is not None else event.kind.value,  # type: ignore[union-attr]
                magnitude=event.magnitude,
                realized={key: float(value) for key, value in real.items()},
                returns_to=event.returns_to,
            )
        )
    return GroundTruth(intervals=tuple(intervals), n_samples=spec.stream.n_samples)

"""Таблица результатов бенчмарка: метрики каждого прогона и описание сценария.

Ответственный: Гарифзянов Т. Р.

Строка таблицы — один прогон (сценарий, seed, детектор). К метрикам
обнаружения (:func:`~scendrift.evaluation.metrics.detection_metrics`)
добавляются:

* accuracy — prequential accuracy на t ≥ W;
* delta_accuracy — разность accuracy с базовой линией NoDrift на той же
  реализации потока (если она есть в прогонах);
* oracle_gap — разность accuracy базовой линии Oracle и детектора;
* параметры сценария и класс таксономии — для анализа по группам (Э3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from scendrift.evaluation.metrics import detection_metrics
from scendrift.generators.base import get_generator
from scendrift.generators.ground_truth import build_ground_truth
from scendrift.interfaces import DetectionRun
from scendrift.scenario import io
from scendrift.scenario.schema import ScenarioSpec
from scendrift.scenario.taxonomy import classify, signature

__all__ = ["scenario_descriptor", "runs_to_frame"]


def scenario_descriptor(spec: ScenarioSpec) -> dict[str, object]:
    """Параметры сценария и класс таксономии в плоском виде."""
    regular = [e for e in spec.events if e.returns_to is None]
    first = regular[0] if regular else None
    cls = classify(spec)
    return {
        "family": spec.stream.family,
        "n": spec.stream.n_samples,
        "d": spec.stream.n_features,
        "pi0": spec.stream.minority_share,
        "eta": spec.stream.label_noise,
        "K": spec.n_events,
        "kind": first.kind.value if first is not None and first.kind else None,
        "form": spec.events[0].form.value if spec.events else None,
        "width": spec.events[0].width if spec.events else None,
        "magnitude": first.magnitude if first is not None else None,
        "alpha": first.affected_share if first is not None else None,
        "recurring": cls.recurring,
        "severity": cls.severity,
        "speed": cls.speed,
        "signature": signature(spec),
    }


def runs_to_frame(
    runs: Sequence[DetectionRun], specs: Sequence[ScenarioSpec] | Mapping[str, ScenarioSpec]
) -> pd.DataFrame:
    """Таблица результатов: прогоны + метрики + описание сценариев.

    Ground truth не зависит от seed реализации, поэтому строится один раз на
    сценарий — без порождения данных, по фактическим величинам событий.
    """
    by_id = dict(specs) if isinstance(specs, Mapping) else {io.scenario_id(s): s for s in specs}
    truth = {
        sid: build_ground_truth(spec, get_generator(spec.stream.family).realized(spec))
        for sid, spec in by_id.items()
    }
    described = {sid: scenario_descriptor(spec) for sid, spec in by_id.items()}
    rows = []
    for run in runs:
        spec = by_id[run.scenario_id]
        metrics = detection_metrics(run.detections, truth[run.scenario_id], spec.evaluation)
        rows.append(
            {
                "scenario_id": run.scenario_id,
                "seed": run.seed,
                "detector": run.detector,
                "accuracy": run.accuracy,
                "runtime_s": run.runtime_s,
                "n_resets": run.extra.get("n_resets"),
                **metrics,
                **described[run.scenario_id],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    key = ["scenario_id", "seed"]
    for baseline, column, sign in (("NoDrift", "delta_accuracy", 1), ("Oracle", "oracle_gap", -1)):
        base = frame.loc[frame["detector"] == baseline, [*key, "accuracy"]]
        if base.empty:
            continue
        merged = frame[key].merge(base.rename(columns={"accuracy": "_base"}), on=key, how="left")
        frame[column] = sign * (frame["accuracy"].to_numpy() - merged["_base"].to_numpy())
    return frame

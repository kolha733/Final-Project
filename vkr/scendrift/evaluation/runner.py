"""Раннер: prequential-прогон детекторов по сценариям, повторы, кэш, параллелизм.

Ответственный: Гарифзянов Т. Р.

Один прогон — тройка (сценарий, seed реализации, детектор):

1. поток порождается генератором (один раз на пару «сценарий, seed»,
   общий для всех детекторов);
2. базовая модель h работает по схеме test-then-train с первого объекта;
3. начиная с объекта W (разогрев) детектор получает индикатор ошибки e_t;
4. при срабатывании в момент t детектор начинается заново (``reset``), а
   при политике a = reset сбрасывается и модель: новая модель учится с
   объекта t + 1. Повторный запуск детектора после срабатывания одинаков
   для всех детекторов и не зависит от того, как каждый из них
   обрабатывает срабатывание внутри себя;
5. prequential accuracy считается на объектах t ≥ W.

Ошибки модели вычисляются блоками (:class:`~scendrift.evaluation.models.PrequentialGaussianNB`);
после сброса блок пересчитывается с нового начала.

Повторы: сценарий прогоняется с seed реализации s, s + 1, …, s + R − 1,
где s — seed сценария. Кэш: результат прогона сохраняется в JSON под
ключом — хэшем (идентификатор сценария с seed, имя детектора с
параметрами, базовая модель, версия формата кэша). Повторный запуск
бенчмарка берёт готовые прогоны из кэша.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed

from scendrift.evaluation.models import make_model
from scendrift.evaluation.protocol import AdaptationPolicy, EvaluationSpec
from scendrift.generators import generate
from scendrift.interfaces import DetectionRun, DriftDetector, StreamData
from scendrift.scenario import io
from scendrift.scenario.schema import ScenarioSpec

__all__ = [
    "CACHE_VERSION",
    "BLOCK",
    "run_stream",
    "replicate_seeds",
    "RunCache",
    "run_scenario",
    "run_benchmark",
    "with_protocol",
]

#: Версия формата и семантики прогона; при изменении раннера кэш становится недействительным.
CACHE_VERSION = 1

#: Размер блока при вычислении ошибок базовой модели.
BLOCK = 2_048


def run_stream(
    data: StreamData, protocol: EvaluationSpec, detector: DriftDetector
) -> dict[str, Any]:
    """Прогон одного детектора по готовому потоку.

    Returns:
        Словарь: detections (кортеж моментов), accuracy, n_resets, runtime_s.
    """
    start_time = time.perf_counter()
    model = make_model(protocol.base_model, data.X, data.y)
    n, warmup = len(data.y), protocol.warmup
    det = detector.clone()
    if getattr(det, "needs_ground_truth", False):
        det = det.bind(data.ground_truth, warmup)  # type: ignore[attr-defined]
    # Приведение к перечислению: model_copy(update=...) может оставить строку.
    reset_model = AdaptationPolicy(protocol.on_detection) is AdaptationPolicy.RESET
    detections: list[int] = []
    correct, n_resets = 0, 0
    state, pos = model.empty_state(), 0
    while pos < n:
        stop = min(n, pos + BLOCK)
        errors, new_state = model.block(pos, stop, state)
        restart = None
        for i, err in enumerate(errors.tolist()):
            t = pos + i
            if t < warmup:
                continue
            correct += not err
            if det.update(float(err)):
                detections.append(t)
                det.reset()
                if reset_model:
                    restart = t + 1
                    break
        if restart is None:
            state, pos = new_state, stop
        else:
            state, pos, n_resets = model.empty_state(), restart, n_resets + 1
    evaluated = n - warmup
    return {
        "detections": tuple(detections),
        "accuracy": correct / evaluated if evaluated > 0 else float("nan"),
        "n_resets": n_resets,
        "runtime_s": time.perf_counter() - start_time,
    }


def replicate_seeds(spec: ScenarioSpec, repeats: int) -> list[int]:
    """Seed реализации для R повторов: s, s + 1, …, s + R − 1."""
    return [spec.seed + r for r in range(int(repeats))]


def _key(spec: ScenarioSpec, seed: int, detector_name: str) -> str:
    payload = [
        io.scenario_id(spec.with_seed(seed)),
        detector_name,
        spec.evaluation.base_model,
        CACHE_VERSION,
    ]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()[:20]


class RunCache:
    """Кэш прогонов на диске: один JSON-файл на прогон."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> DetectionRun | None:
        path = self._path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        data["detections"] = tuple(data["detections"])
        return DetectionRun(**data)

    def put(self, key: str, run: DetectionRun) -> None:
        payload = asdict(run)
        payload["extra"] = dict(run.extra)
        self._path(key).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def run_scenario(
    spec: ScenarioSpec,
    seed: int,
    detectors: Sequence[DriftDetector],
    cache: RunCache | None = None,
) -> list[DetectionRun]:
    """Прогоны всех детекторов на одной реализации сценария (поток порождается один раз)."""
    runs: list[DetectionRun | None] = []
    missing = []
    for i, det in enumerate(detectors):
        cached = cache.get(_key(spec, seed, det.name)) if cache is not None else None
        runs.append(cached)
        if cached is None:
            missing.append(i)
    if missing:
        data = generate(spec, seed)
        sid = io.scenario_id(spec)
        for i in missing:
            det = detectors[i]
            res = run_stream(data, spec.evaluation, det)
            run = DetectionRun(
                scenario_id=sid,
                seed=int(seed),
                detector=det.name,
                detections=res["detections"],
                n_samples=spec.stream.n_samples,
                accuracy=res["accuracy"],
                runtime_s=res["runtime_s"],
                extra={"n_resets": res["n_resets"], "base_model": spec.evaluation.base_model},
            )
            if cache is not None:
                cache.put(_key(spec, seed, det.name), run)
            runs[i] = run
    return [r for r in runs if r is not None]


def run_benchmark(
    specs: Sequence[ScenarioSpec],
    detectors: Sequence[DriftDetector],
    *,
    repeats: int = 1,
    n_jobs: int = 1,
    cache_dir: str | Path | None = None,
) -> list[DetectionRun]:
    """Прогон всех детекторов по всем сценариям с R повторами.

    Args:
        specs: сценарии (например, набор из :mod:`scendrift.formation.suite`).
        detectors: детекторы и базовые линии.
        repeats: число повторов R (seed реализации s, …, s + R − 1).
        n_jobs: число параллельных процессов (joblib); 1 — без параллелизма.
        cache_dir: каталог кэша (None — без кэша).

    Returns:
        Прогоны в порядке: сценарий → seed → детектор.
    """
    names = [d.name for d in detectors]
    if len(set(names)) != len(names):
        raise ValueError("имена детекторов должны быть уникальны")
    tasks = [(spec, seed) for spec in specs for seed in replicate_seeds(spec, repeats)]

    def task(spec: ScenarioSpec, seed: int) -> list[DetectionRun]:
        cache = RunCache(cache_dir) if cache_dir is not None else None
        return run_scenario(spec, seed, detectors, cache)

    if n_jobs == 1:
        results = [task(spec, seed) for spec, seed in tasks]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(task)(spec, seed) for spec, seed in tasks)
    return [run for chunk in results for run in chunk]


def with_protocol(spec: ScenarioSpec, **updates: object) -> ScenarioSpec:
    """Копия сценария с изменённым протоколом оценки (с проверкой значений).

    ``model_copy(update=...)`` в pydantic значения не проверяет; здесь
    протокол строится заново через ``model_validate``, поэтому, например,
    строка ``"none"`` становится значением перечисления.
    """
    protocol = EvaluationSpec.model_validate({**spec.evaluation.model_dump(), **updates})
    return spec.model_copy(update={"evaluation": protocol})

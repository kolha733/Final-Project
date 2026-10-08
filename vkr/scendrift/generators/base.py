"""Общий конвейер генерации потока и реестр генераторов.

Ответственный: Будаев К. В.

Генератор семейства реализует две операции:

* ``_sample`` — порождение признаков и чистых меток по временной разметке;
* ``realized`` — фактическая величина каждого события по группам параметров.

Остальное общее для всех семейств:

1. проверка допустимости сценария;
2. построение временной разметки;
3. шум меток;
4. построение ground truth;
5. упаковка результата в :class:`~scendrift.interfaces.StreamData`.

Случайность разделена на независимые подпотоки ГСЧ: выборка объектов и
шум меток используют разные подпотоки seed реализации. Поэтому при
изменении уровня шума признаки X не меняются, и сценарии, отличающиеся
только шумом, сравниваются на одних и тех же объектах.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np

from scendrift.generators.ground_truth import build_ground_truth
from scendrift.generators.transitions import Timeline, timeline
from scendrift.interfaces import StreamData
from scendrift.scenario.io import scenario_id
from scendrift.scenario.schema import ScenarioSpec
from scendrift.scenario.validation import ensure_valid

__all__ = [
    "FamilyGenerator",
    "register_generator",
    "get_generator",
    "generate",
    "rng_stream",
]

_SAMPLE_TAG = 3_000_017
_NOISE_TAG = 3_000_019


def rng_stream(seed: int, tag: int, *extra: int) -> np.random.Generator:
    """Независимый подпоток ГСЧ, однозначно заданный seed и меткой."""
    return np.random.default_rng([int(seed), tag, *map(int, extra)])


class FamilyGenerator(ABC):
    """Базовый класс генератора семейства ``family``."""

    family: str = ""

    def generate(self, spec: ScenarioSpec, seed: int | None = None) -> StreamData:
        """Порождает поток по сценарию.

        Args:
            spec: допустимый сценарий этого семейства.
            seed: seed реализации; по умолчанию ``spec.seed``.

        Raises:
            ScenarioValidationError: если сценарий недопустим.
            ValueError: если семейство сценария не совпадает с генератором.
        """
        if spec.stream.family != self.family:
            raise ValueError(f"генератор {self.family!r} получил сценарий {spec.stream.family!r}")
        ensure_valid(spec)
        seed = spec.seed if seed is None else int(seed)
        tl = timeline(spec)
        X, y_clean, concept = self._sample(spec, tl, rng_stream(seed, _SAMPLE_TAG))
        flips = rng_stream(seed, _NOISE_TAG).random(len(y_clean)) < spec.stream.label_noise
        y = (y_clean ^ flips).astype(np.int8)
        gt = build_ground_truth(spec, self.realized(spec))
        return StreamData(
            X=X,
            y=y,
            ground_truth=gt,
            concept=concept.astype(np.int32),
            y_clean=y_clean.astype(np.int8),
            scenario_id=scenario_id(spec.with_seed(seed)),
            seed=seed,
            progress=tl.p.astype(np.float32),
            feature_names=self.feature_names(spec),
        )

    @abstractmethod
    def _sample(
        self, spec: ScenarioSpec, tl: Timeline, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Признаки X (n × d), чистые метки (n) и номер порождающего состояния (n)."""

    @abstractmethod
    def realized(self, spec: ScenarioSpec) -> list[Mapping[str, float]]:
        """Фактическая величина каждого события по компонентам real/virtual/prior."""

    def feature_names(self, spec: ScenarioSpec) -> tuple[str, ...]:
        """Имена признаков (по умолчанию x0, x1, …)."""
        return tuple(f"x{j}" for j in range(spec.stream.n_features))


_GENERATORS: dict[str, FamilyGenerator] = {}


def register_generator(generator: FamilyGenerator) -> FamilyGenerator:
    """Регистрирует генератор семейства."""
    _GENERATORS[generator.family] = generator
    return generator


def get_generator(family: str) -> FamilyGenerator:
    """Генератор семейства.

    Raises:
        KeyError: если генератор не зарегистрирован.
    """
    if family not in _GENERATORS:
        raise KeyError(f"нет генератора для семейства {family!r}")
    return _GENERATORS[family]


def generate(spec: ScenarioSpec, seed: int | None = None) -> StreamData:
    """Порождает поток по сценарию любого зарегистрированного семейства."""
    return get_generator(spec.stream.family).generate(spec, seed)

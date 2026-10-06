"""Обёртки классических генераторов river: SEA, Agrawal, STAGGER, Sine, Mixed.

Ответственный: Будаев К. В.

Классический сценарий — смена функции классификации генератора (варианта
концепта) в заданные моменты. Такие потоки исторически составляют
«ручные» бенчмарки [SEA, Agrawal, STAGGER и др.], и на них в эксперименте
Э6 сравниваются ранжирования детекторов.

Семейства некалиброваны: величину смены концепта нельзя задать, её можно
только измерить. Для всех пяти генераторов признаки X при одном seed не
зависят от функции классификации (проверено), поэтому величина
TV(P_a(X, y), P_b(X, y)) = P(y_a ≠ y_b) точно оценивается на общей
эталонной выборке из N_REF объектов. Стандартная ошибка оценки не
превышает 0,5/√N_REF ≈ 0,0035.

Параметры семейства (``family_params``):

* ``variants`` — номера вариантов для концептов c₀, c₁, … (по одному на
  исходный концепт и на каждое обычное событие). По умолчанию варианты
  перебираются по кругу: 0, 1, 2, …

Ограничения семейства (C8):

* размерность задаётся генератором;
* доля класса не управляется (π₀ остаётся по умолчанию);
* α = 1;
* только вид real и формы sudden и gradual.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from river.datasets import synth

from scendrift.generators.base import FamilyGenerator, register_generator
from scendrift.generators.transitions import Timeline
from scendrift.scenario.enums import DriftForm, DriftKind
from scendrift.scenario.families import FamilyInfo, register_family
from scendrift.scenario.report import Violation
from scendrift.scenario.schema import ScenarioSpec

__all__ = [
    "CLASSIC",
    "ClassicSpec",
    "ClassicGenerator",
    "N_REF",
    "concept_variants",
    "switch_magnitude",
]

#: Размер эталонной выборки для оценки величины смены концепта.
N_REF = 20_000

_REF_TAG = 4_000_037
_CONCEPT_TAG = 4_000_039


@dataclass(frozen=True)
class ClassicSpec:
    """Описание классического генератора river.

    Attributes:
        factory: конструктор генератора river.
        param: имя параметра, задающего вариант концепта.
        n_variants: число вариантов концепта.
        n_features: размерность.
        kwargs: прочие параметры (шум и балансировка классов отключены:
            шум меток добавляет фреймворк, а балансировка меняет P(X)).
        source: ключ первоисточника генератора в списке литературы.
    """

    factory: Callable[..., Any]
    param: str
    n_variants: int
    n_features: int
    kwargs: Mapping[str, Any]
    source: str


CLASSIC: dict[str, ClassicSpec] = {
    "river:SEA": ClassicSpec(synth.SEA, "variant", 4, 3, {"noise": 0.0}, "sea2001"),
    "river:Agrawal": ClassicSpec(
        synth.Agrawal,
        "classification_function",
        10,
        9,
        {"balance_classes": False, "perturbation": 0.0},
        "agrawal1993",
    ),
    "river:STAGGER": ClassicSpec(
        synth.STAGGER, "classification_function", 3, 3, {"balance_classes": False}, "stagger1986"
    ),
    "river:Sine": ClassicSpec(
        synth.Sine,
        "classification_function",
        4,
        2,
        {"balance_classes": False, "has_noise": False},
        "ddm2004",
    ),
    "river:Mixed": ClassicSpec(
        synth.Mixed, "classification_function", 2, 4, {"balance_classes": False}, "ddm2004"
    ),
}


def _river_seed(*parts: int) -> int:
    """Целочисленный seed для river из иерархии seed."""
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def _draw(family: str, variant: int, seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Порождает n объектов варианта ``variant`` генератора семейства."""
    cs = CLASSIC[family]
    gen = cs.factory(**{cs.param: variant, "seed": seed, **cs.kwargs})
    X = np.empty((n, cs.n_features))
    y = np.empty(n, dtype=np.int8)
    for i, (x, label) in enumerate(itertools.islice(gen, n)):
        X[i] = list(x.values())
        y[i] = int(label)
    return X, y


@lru_cache(maxsize=64)
def _reference(family: str, variant: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return _draw(family, variant, seed, N_REF)


def concept_variants(spec: ScenarioSpec) -> list[int]:
    """Вариант концепта для каждого состояния цепочки c₀ … c_K.

    Обычное событие создаёт новый концепт и берёт следующий вариант из
    ``variants``. Повторяющееся событие возвращает вариант концепта ρ.
    """
    cs = CLASSIC[spec.stream.family]
    n_regular = sum(e.returns_to is None for e in spec.events)
    default = [i % cs.n_variants for i in range(n_regular + 1)]
    pool = list(spec.stream.family_params.get("variants", default))
    states = [pool[0]]
    used = 1
    for event in spec.events:
        if event.returns_to is None:
            states.append(pool[used] if used < len(pool) else -1)
            used += 1
        else:
            states.append(states[event.returns_to])
    return states


def switch_magnitude(family: str, a: int, b: int, concept_seed: int = 0) -> tuple[float, float]:
    """TV смены варианта a → b и изменение доли класса |ΔP(y)| на эталонной выборке."""
    seed = _river_seed(concept_seed, _REF_TAG)
    Xa, ya = _reference(family, a, seed)
    Xb, yb = _reference(family, b, seed)
    if not np.array_equal(Xa, Xb):  # pragma: no cover - защита от изменения river
        raise RuntimeError(f"{family}: X зависит от варианта, оценка TV недопустима")
    return float(np.mean(ya != yb)), float(abs(yb.mean() - ya.mean()))


def _check(spec: ScenarioSpec) -> list[Violation]:
    cs = CLASSIC[spec.stream.family]
    out: list[Violation] = []
    if spec.stream.n_features != cs.n_features:
        out.append(
            Violation(
                "C8",
                f"размерность задаётся генератором: d = {cs.n_features}",
                "stream.n_features",
                suggestion=cs.n_features,
            )
        )
    if spec.stream.minority_share != 0.5:
        out.append(
            Violation(
                "C8", "доля класса задаётся генератором, π₀ не управляется", "stream.minority_share"
            )
        )
    for k, event in enumerate(spec.events):
        if event.returns_to is None and event.affected_share != 1.0:
            out.append(
                Violation(
                    "C8", "доля затронутых признаков не управляется", f"events[{k}].affected_share"
                )
            )
    n_regular = sum(e.returns_to is None for e in spec.events)
    variants = spec.stream.family_params.get("variants")
    bad = variants is not None and (
        not isinstance(variants, list)
        or any(not isinstance(v, int) or not 0 <= v < cs.n_variants for v in variants)
        or len(variants) != n_regular + 1
    )
    if bad:
        out.append(
            Violation(
                "C8",
                f"variants: нужен список из {n_regular + 1} номеров в [0, {cs.n_variants})",
                "stream.family_params",
            )
        )
        return out
    states = concept_variants(spec)
    for k, event in enumerate(spec.events):
        if states[k + 1] == states[k]:
            code = "C7" if event.returns_to is not None else "C8"
            out.append(
                Violation(
                    code,
                    f"событие не меняет концепт: вариант {states[k]} → {states[k + 1]}",
                    f"events[{k}]",
                )
            )
    return out


class ClassicGenerator(FamilyGenerator):
    """Генератор классического семейства river."""

    def __init__(self, family: str) -> None:
        self.family = family

    def feature_names(self, spec: ScenarioSpec) -> tuple[str, ...]:
        cs = CLASSIC[self.family]
        gen = cs.factory(**{cs.param: 0, "seed": 0, **cs.kwargs})
        x, _ = next(iter(gen))
        return tuple(str(k) for k in x)

    def realized(self, spec: ScenarioSpec) -> list[Mapping[str, float]]:
        states = concept_variants(spec)
        out = []
        for k in range(spec.n_events):
            tv, _ = switch_magnitude(
                self.family, states[k], states[k + 1], spec.stream.concept_seed
            )
            out.append(
                {DriftKind.REAL.value: tv, DriftKind.VIRTUAL.value: 0.0, DriftKind.PRIOR.value: 0.0}
            )
        return out

    def _sample(
        self, spec: ScenarioSpec, tl: Timeline, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = spec.stream.n_samples
        states = concept_variants(spec)
        take_new = rng.random(n) < tl.p
        state = np.where(take_new, tl.dst, tl.src).astype(np.int32)
        X = np.empty((n, CLASSIC[self.family].n_features))
        y = np.empty(n, dtype=np.int8)
        seed_root = int(rng.integers(2**31))
        for s in np.unique(state):
            idx = np.flatnonzero(state == s)
            Xs, ys = _draw(
                self.family, states[s], _river_seed(seed_root, _CONCEPT_TAG, s), len(idx)
            )
            X[idx], y[idx] = Xs, ys
        return X, y, state


for _name in CLASSIC:
    register_family(
        FamilyInfo(
            name=_name,
            description=f"Генератор river {_name.split(':')[1]}: смена варианта концепта",
            kinds=frozenset({DriftKind.REAL}),
            forms=frozenset({DriftForm.SUDDEN, DriftForm.GRADUAL}),
            supports_recurring=True,
            calibrated=False,
            check=_check,
            params=frozenset({"variants"}),
        )
    )
    register_generator(ClassicGenerator(_name))

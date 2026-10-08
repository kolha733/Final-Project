"""Шкала сложности λ ∈ [0, 1] и каталог профилей easy / medium / hard.

Ответственный: Будаев К. В.

Один параметр λ управляет сразу несколькими факторами сложности. Каждый
фактор j интерполируется между значением «легко» (λ = 0) и «трудно»
(λ = 1) на своей шкале g_j (линейной или логарифмической):

    ξ_j(λ) = g_j⁻¹((1 − λ)·g_j(ξ_j^easy) + λ·g_j(ξ_j^hard)).

Поэтому каждый фактор — монотонная функция λ. Остальные параметры
сценария берутся из базового шаблона и от λ не зависят.

Факторы шкалы по умолчанию (план работы, разд. 2.2): с ростом λ
уменьшаются величина дрейфа m и доля затронутых признаков α, растут
ширина перехода ℓ, шум меток η и дисбаланс (уменьшается π₀).

То, что эти факторы действительно усложняют обнаружение, — **гипотеза**.
Она проверяется в эксперименте Э4: F1 детекторов должна убывать с ростом λ
(ρ Спирмена < 0), а вклад каждого фактора оценивается отдельно.

Профили каталога easy, medium и hard — это точки шкалы λ = 0; 0,5; 1.
Они регистрируются как шаблоны ``catalog/easy`` и т. д., поэтому
наследование (``extends: catalog/medium``) и шкала λ согласованы
по построению.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from scendrift.formation.mapping import PARAM_PATHS, point_to_dict, point_to_spec
from scendrift.formation.templates import register_template
from scendrift.scenario.schema import ScenarioSpec

__all__ = [
    "Factor",
    "DifficultyScale",
    "BASE_TEMPLATE",
    "DIFFICULTY",
    "PROFILES",
]


@dataclass(frozen=True)
class Factor:
    """Фактор сложности: параметр, значения при λ = 0 и λ = 1, шкала интерполяции."""

    name: str
    easy: float
    hard: float
    scale: Literal["linear", "log"] = "linear"
    integer: bool = False

    def __post_init__(self) -> None:
        if self.easy == self.hard:
            raise ValueError(f"{self.name}: значения easy и hard совпадают")
        if self.scale == "log" and min(self.easy, self.hard) <= 0:
            raise ValueError(f"{self.name}: для логарифмической шкалы значения должны быть > 0")

    def at(self, lam: float) -> float | int:
        """Значение фактора при сложности λ ∈ [0, 1]."""
        if not 0.0 <= lam <= 1.0:
            raise ValueError("λ должна лежать в [0, 1]")
        if self.scale == "log":
            value = math.exp((1 - lam) * math.log(self.easy) + lam * math.log(self.hard))
        else:
            value = (1 - lam) * self.easy + lam * self.hard
        if lam == 0.0:
            value = self.easy
        elif lam == 1.0:
            value = self.hard
        return int(round(value)) if self.integer else float(value)

    @property
    def direction(self) -> str:
        """Направление изменения при росте λ."""
        return "↑" if self.hard > self.easy else "↓"


class DifficultyScale:
    """Шкала сложности: базовый шаблон и факторы."""

    def __init__(self, base: Mapping[str, Any], factors: Sequence[Factor]) -> None:
        names = [f.name for f in factors]
        if len(set(names)) != len(names):
            raise ValueError("факторы повторяются")
        for name in names:
            if name not in PARAM_PATHS and "." not in name:
                raise KeyError(f"фактор {name!r} не отображается в поле сценария")
        self.base = dict(base)
        self.factors = tuple(factors)

    def values(self, lam: float) -> dict[str, float | int]:
        """Значения факторов при сложности λ."""
        return {f.name: f.at(lam) for f in self.factors}

    def template(self, lam: float) -> dict[str, Any]:
        """Описание сценария (компактная форма) при сложности λ."""
        return point_to_dict(self.values(lam), self.base)

    def scenario(
        self, lam: float, *, concept_seed: int = 0, seed: int = 0, name: str | None = None
    ) -> ScenarioSpec:
        """Сценарий при сложности λ."""
        return point_to_spec(
            self.values(lam),
            self.base,
            concept_seed=concept_seed,
            seed=seed,
            name=name or f"difficulty-{lam:.3f}",
            meta={"difficulty": float(lam)},
        )

    def table(self, lams: Sequence[float]) -> list[dict[str, Any]]:
        """Значения факторов на сетке λ (для таблиц)."""
        return [{"λ": lam, **self.values(lam)} for lam in lams]

    def describe(self) -> list[dict[str, str]]:
        """Описание факторов: имя, easy, hard, шкала, направление."""
        return [
            {
                "фактор": f.name,
                "λ = 0": f"{f.easy:g}".replace(".", ","),
                "λ = 1": f"{f.hard:g}".replace(".", ","),
                "шкала": "лог." if f.scale == "log" else "лин.",
                "с ростом λ": f.direction,
            }
            for f in self.factors
        ]


#: Базовый шаблон каталога: всё, что не зависит от сложности.
BASE_TEMPLATE: dict[str, Any] = {
    "description": "Базовый шаблон каталога: hyperplane_gauss, три реальных дрейфа",
    # d = 20: при α = 0,2 затрагиваются 4 признака, и величина m = 0,05 достижима
    # (C10) у всех трёх событий; это проверяется в п. 4.6.5 на сетке λ и многих seed.
    "stream": {"family": "hyperplane_gauss", "n_samples": 20_000, "n_features": 20},
    "drifts": {
        "schedule": {"mode": "uniform", "count": 3},
        "defaults": {"form": "gradual", "kind": "real", "shape": "linear"},
    },
    "evaluation": {"warmup": 1_000, "acceptance_window": 2_000},
}

#: Шкала сложности по умолчанию.
DIFFICULTY = DifficultyScale(
    BASE_TEMPLATE,
    [
        Factor("magnitude", 0.4, 0.05, "log"),
        Factor("affected_share", 1.0, 0.2),
        Factor("width", 200, 4_000, "log", integer=True),
        Factor("label_noise", 0.0, 0.2),
        Factor("minority_share", 0.5, 0.1),
    ],
)

#: Профили каталога и соответствующие им значения λ.
PROFILES: dict[str, float] = {"easy": 0.0, "medium": 0.5, "hard": 1.0}

register_template("base", BASE_TEMPLATE)
for _name, _lam in PROFILES.items():
    _profile = point_to_dict(DIFFICULTY.values(_lam), {})
    _profile["description"] = f"Профиль {_name}: шкала сложности при λ = {_lam:g}"
    register_template(_name, {"extends": "catalog/base", **_profile})

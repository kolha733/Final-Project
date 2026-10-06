"""Таксономия сценариев: отнесение сценария к классам по осям классификации.

Ответственный: Будаев К. В.

По таксономии группируются результаты бенчмарка: «на каких классах
сценариев детектор ошибается». Пороги — договорённость, зафиксированная в
константах. При изменении порогов меняются только границы классов,
сами сценарии остаются прежними.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from scendrift.scenario.enums import DriftKind
from scendrift.scenario.schema import ScenarioSpec

__all__ = [
    "SEVERITY_BINS",
    "SPEED_RATIO",
    "BALANCED_SHARE",
    "ScenarioClass",
    "classify",
    "signature",
]

#: Границы классов величины: low < 0,1 ≤ medium < 0,3 ≤ high.
SEVERITY_BINS = (0.1, 0.3)

#: Переход считается медленным, если w ≥ SPEED_RATIO · Δ.
SPEED_RATIO = 0.5

#: Поток сбалансирован, если π ≥ BALANCED_SHARE.
BALANCED_SHARE = 0.4


@dataclass(frozen=True)
class ScenarioClass:
    """Класс сценария по осям таксономии.

    Attributes:
        n_drifts: число событий K.
        forms: формы переходов (отсортированы).
        kinds: виды обычных событий (отсортированы).
        recurring: есть ли возврат к прежнему концепту.
        severity: класс наибольшей заданной величины: none/low/medium/high.
        speed: abrupt (все ℓ = 0), fast или slow (по наибольшей ширине).
        scope: global (α = 1 у всех событий real/virtual), partial или n/a (их нет).
        balance: balanced или imbalanced.
        noise: clean (η = 0) или noisy.
    """

    n_drifts: int
    forms: tuple[str, ...]
    kinds: tuple[str, ...]
    recurring: bool
    severity: str
    speed: str
    scope: str
    balance: str
    noise: str

    def as_dict(self) -> dict[str, object]:
        """Плоский словарь для таблиц pandas."""
        return asdict(self)


def _severity_class(m: float | None) -> str:
    if m is None:
        return "none"
    low, high = SEVERITY_BINS
    return "low" if m < low else ("medium" if m < high else "high")


def _scope(regular: list) -> str:
    """Охват: α учитывается только у real и virtual (у prior он не используется)."""
    spatial = [e for e in regular if e.kind in (DriftKind.REAL, DriftKind.VIRTUAL)]
    if not spatial:
        return "n/a"
    return "global" if all(e.affected_share == 1.0 for e in spatial) else "partial"


def classify(spec: ScenarioSpec) -> ScenarioClass:
    """Относит сценарий к классам таксономии."""
    events = spec.events
    magnitudes = [e.magnitude for e in events if e.magnitude is not None]
    widths = [e.width for e in events]
    if not widths or max(widths) == 0:
        speed = "abrupt"
    elif max(widths) < SPEED_RATIO * spec.evaluation.acceptance_window:
        speed = "fast"
    else:
        speed = "slow"
    regular = [e for e in events if e.kind is not None]
    return ScenarioClass(
        n_drifts=len(events),
        forms=tuple(sorted({e.form.value for e in events})),
        kinds=tuple(sorted({e.kind.value for e in regular})),  # type: ignore[union-attr]
        recurring=any(e.returns_to is not None for e in events),
        severity=_severity_class(max(magnitudes) if magnitudes else None),
        speed=speed,
        scope=_scope(regular),
        balance="balanced" if spec.stream.minority_share >= BALANCED_SHARE else "imbalanced",
        noise="clean" if spec.stream.label_noise == 0.0 else "noisy",
    )


def signature(spec: ScenarioSpec) -> str:
    """Краткая строковая сигнатура класса, например ``real/gradual/K=3/high/slow``."""
    c = classify(spec)
    parts = [
        "+".join(c.kinds) or "none",
        "+".join(c.forms) or "none",
        f"K={c.n_drifts}",
        c.severity,
        c.speed,
        c.scope,
        c.balance,
        c.noise,
    ]
    if c.recurring:
        parts.insert(3, "recurring")
    return "/".join(parts)

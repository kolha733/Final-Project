"""Отчёт о проверке сценария: нарушения ограничений валидности.

Ответственный: Будаев К. В.

Каталог ограничений :data:`CONSTRAINTS` — это единый справочник. Из него
строятся таблица ограничений в ноутбуке и в пояснительной записке, поэтому
формулировки в коде и в тексте не расходятся.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["CONSTRAINTS", "Violation", "ValidationReport", "ScenarioValidationError"]

#: Каталог ограничений валидности: код → (область проверки, формулировка).
#: События нумеруются с единицы: e_1, …, e_K; концепт c_k возникает после e_k.
CONSTRAINTS: dict[str, tuple[str, str]] = {
    "C1": ("событие", "φ = sudden ⇔ ℓ = 0; при φ ∈ {gradual, incremental} ℓ ≥ 2; onset ≥ 0"),
    "C2": (
        "семейство",
        "обычное событие: m ∈ (0, 1] задана для калиброванного семейства и не задана "
        "для некалиброванного (там m оценивается по данным)",
    ),
    "C3": ("событие", "для повторяющегося события (ρ задано) κ и m не задаются"),
    "C4": ("сценарий", "onset₁ ≥ W: дрейф не начинается во время разогрева"),
    "C5": ("сценарий", "end_K + Δ ≤ n: после последнего перехода есть окно допуска"),
    "C6": ("сценарий", "onset_{k+1} ≥ end_k + Δ: следы событий [onset, end + Δ) не пересекаются"),
    "C7": (
        "сценарий",
        "ρ_k ≤ k − 1 (концепт уже существует), и возврат меняет распределение: c_{ρ_k} ≠ c_{k−1}",
    ),
    "C8": (
        "семейство",
        "вид κ, форма φ, возврат и ключи family_params поддерживаются семейством f",
    ),
    "C9": ("семейство", "число затронутых признаков ⌊α·d⌉ ≥ 2 (для κ ∈ {real, virtual})"),
    "C10": ("семейство", "real: m ≤ m_max(q, π₀, π_k) — величина достижима поворотом внутри A"),
    "C11": ("семейство", "virtual: m ≤ 0,99 (TV < 1 при конечном сдвиге)"),
    "C12": ("семейство", "prior: новая доля P(y=1) лежит в [0,01; 0,99]"),
    "C13": ("семейство", "семейство f зарегистрировано в реестре"),
    "C14": ("сценарий", "W + Δ ≤ n: протокол оценки помещается в поток"),
    "W1": (
        "предупреждение",
        "real: m·(1 − 2η) < 0,02 — дрейф почти не наблюдаем на фоне шума меток",
    ),
    "W2": (
        "предупреждение",
        "virtual: P(y|X) не меняется, детекторы по ошибке могут его не видеть",
    ),
    "W3": (
        "предупреждение",
        "real при π_k ≠ π₀: вместе с P(y|X) меняется и P(X) (перевзвешивание классов)",
    ),
    "W4": (
        "предупреждение",
        "реальные данные: событие real или virtual меняет и P(y) более чем на 0,02",
    ),
}


@dataclass(frozen=True)
class Violation:
    """Нарушение ограничения валидности.

    Attributes:
        code: код ограничения из :data:`CONSTRAINTS`.
        message: пояснение на русском языке.
        path: путь к полю сценария, например ``events[2].magnitude``.
        level: ``error`` делает сценарий недопустимым, ``warning`` нет.
        suggestion: допустимое значение для исправления, если оно известно.
    """

    code: str
    message: str
    path: str = ""
    level: Literal["error", "warning"] = "error"
    suggestion: Any = None


@dataclass
class ValidationReport:
    """Результат проверки сценария."""

    violations: list[Violation] = field(default_factory=list)

    @property
    def errors(self) -> list[Violation]:
        """Нарушения уровня ``error``."""
        return [v for v in self.violations if v.level == "error"]

    @property
    def warnings(self) -> list[Violation]:
        """Нарушения уровня ``warning``."""
        return [v for v in self.violations if v.level == "warning"]

    @property
    def ok(self) -> bool:
        """Сценарий допустим, если нет ни одной ошибки."""
        return not self.errors

    def codes(self) -> set[str]:
        """Множество кодов нарушенных ограничений."""
        return {v.code for v in self.violations}

    def raise_if_invalid(self) -> None:
        """Выбрасывает :class:`ScenarioValidationError` при наличии ошибок."""
        if not self.ok:
            raise ScenarioValidationError(self)

    def __str__(self) -> str:
        if not self.violations:
            return "Сценарий допустим, нарушений нет."
        lines = [f"[{v.level}] {v.code} {v.path}: {v.message}" for v in self.violations]
        return "\n".join(lines)


class ScenarioValidationError(ValueError):
    """Сценарий нарушает ограничения валидности."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("Недопустимый сценарий:\n" + str(report))

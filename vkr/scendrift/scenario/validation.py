"""Проверка ограничений валидности сценария и автоматическое исправление.

Ответственный: Будаев К. В.

Локальные ограничения (C1–C3) проверяет pydantic при создании модели.
Модуль добавляет:

* ограничения временной разметки (C4–C6) и ссылок возврата (C7), общие для
  всех семейств;
* поддержку вида и формы дрейфа семейством (C8, C13);
* ограничения семейства (для ``hyperplane_gauss``: C9–C12);
* предупреждения W1, W2.

Функция :func:`repair` нужна модулю формирования сценариев (этап 3): она
переносит события так, чтобы выполнялись C4–C6, и применяет предложенные
валидатором допустимые значения (C9–C12).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from scendrift.scenario.enums import DriftKind
from scendrift.scenario.families import get_family
from scendrift.scenario.report import ScenarioValidationError, ValidationReport, Violation
from scendrift.scenario.schema import DriftEvent, ScenarioSpec, layout_uniform

__all__ = ["check", "ensure_valid", "is_valid", "repair", "RepairResult", "concept_ids"]

#: Порог наблюдаемости W1 для эффективной величины m·(1 − 2η).
OBSERVABILITY_THRESHOLD = 0.02

_EVENT_PATH = re.compile(r"^events\[(\d+)\]\.(\w+)$")


def concept_ids(spec: ScenarioSpec) -> list[int | None]:
    """Идентификаторы концептов c₀ … c_K без построения геометрии.

    Обычное событие создаёт новый концепт, а повторяющееся берёт
    идентификатор концепта ρ. Для недопустимой ссылки возвращается None.
    """
    ids: list[int | None] = [0]
    next_id = 1
    for k, event in enumerate(spec.events):
        if event.returns_to is None:
            ids.append(next_id)
            next_id += 1
        elif event.returns_to <= k:
            ids.append(ids[event.returns_to])
        else:
            ids.append(None)
    return ids


def _check_timeline(spec: ScenarioSpec) -> list[Violation]:
    """C4–C7: разогрев, хвост, непересечение следов, ссылки возврата."""
    out: list[Violation] = []
    ev, n = spec.events, spec.stream.n_samples
    delta, warmup = spec.evaluation.acceptance_window, spec.evaluation.warmup
    if not ev:
        return out
    if ev[0].onset < warmup:
        out.append(
            Violation(
                "C4",
                f"первый переход начинается в {ev[0].onset} < W = {warmup}",
                "events[0].position",
            )
        )
    if ev[-1].end + delta > n:
        out.append(
            Violation(
                "C5",
                f"end_K + Δ = {ev[-1].end + delta} > n = {n}: нет окна допуска",
                f"events[{len(ev) - 1}].position",
            )
        )
    for k in range(len(ev) - 1):
        if ev[k + 1].onset < ev[k].end + delta:
            out.append(
                Violation(
                    "C6",
                    f"событие {k + 1} начинается в {ev[k + 1].onset}, раньше "
                    f"end_{k} + Δ = {ev[k].end + delta}",
                    f"events[{k + 1}].position",
                )
            )
    ids = concept_ids(spec)
    for k, event in enumerate(ev):
        if event.returns_to is None:
            continue
        if ids[k + 1] is None or ids[k + 1] == ids[k]:
            out.append(
                Violation(
                    "C7",
                    f"возврат к концепту {event.returns_to} невозможен: он ещё не "
                    "существует или совпадает с текущим",
                    f"events[{k}].returns_to",
                )
            )
    return out


def _check_support(spec: ScenarioSpec) -> list[Violation]:
    """C8, C13: семейство зарегистрировано и поддерживает события."""
    fam = get_family(spec.stream.family)
    if fam is None:
        return [
            Violation(
                "C13",
                f"семейство {spec.stream.family!r} не зарегистрировано",
                "stream.family",
            )
        ]
    out: list[Violation] = []
    for k, event in enumerate(spec.events):
        path = f"events[{k}]"
        if event.returns_to is not None and not fam.supports_recurring:
            out.append(Violation("C8", "семейство не поддерживает возврат", f"{path}.returns_to"))
        if event.kind is not None and event.kind not in fam.kinds:
            out.append(Violation("C8", f"вид {event.kind} не поддерживается", f"{path}.kind"))
        if event.form not in fam.forms:
            out.append(Violation("C8", f"форма {event.form} не поддерживается", f"{path}.form"))
    return out


def _warnings(spec: ScenarioSpec) -> list[Violation]:
    """W1, W2: сценарий допустим, но результаты стоит интерпретировать осторожно."""
    out: list[Violation] = []
    eta = spec.stream.label_noise
    virtual = []
    for k, event in enumerate(spec.events):
        if event.magnitude is not None and event.kind is DriftKind.REAL:
            effective = event.magnitude * (1.0 - 2.0 * eta)
            if effective < OBSERVABILITY_THRESHOLD:
                out.append(
                    Violation(
                        "W1",
                        f"эффективная величина m·(1 − 2η) = {effective:.4f} "
                        f"< {OBSERVABILITY_THRESHOLD}",
                        f"events[{k}].magnitude",
                        level="warning",
                    )
                )
        if event.kind is DriftKind.VIRTUAL:
            virtual.append(k)
    if virtual:
        out.append(
            Violation(
                "W2",
                f"события {virtual} — виртуальный дрейф: P(y|X) не меняется, "
                "детекторы по потоку ошибок не обязаны его обнаруживать",
                "events",
                level="warning",
            )
        )
    return out


def check(spec: ScenarioSpec) -> ValidationReport:
    """Проверяет все ограничения валидности сценария.

    Args:
        spec: сценарий, уже прошедший локальную проверку pydantic.

    Returns:
        Отчёт со списком нарушений без повторов (по коду и пути).
    """
    violations = _check_timeline(spec)
    support = _check_support(spec)
    violations += support
    fam = get_family(spec.stream.family)
    if fam is not None and fam.check is not None and not support:
        violations += fam.check(spec)
    violations += _warnings(spec)
    seen: set[tuple[str, str]] = set()
    unique = []
    for v in violations:
        key = (v.code, v.path)
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return ValidationReport(unique)


def is_valid(spec: ScenarioSpec) -> bool:
    """Сценарий удовлетворяет всем ограничениям (предупреждения допустимы)."""
    return check(spec).ok


def ensure_valid(spec: ScenarioSpec) -> ScenarioSpec:
    """Возвращает сценарий без изменений или выбрасывает ScenarioValidationError."""
    check(spec).raise_if_invalid()
    return spec


@dataclass
class RepairResult:
    """Результат исправления: допустимый сценарий и журнал действий."""

    spec: ScenarioSpec
    actions: list[str] = field(default_factory=list)
    report: ValidationReport = field(default_factory=ValidationReport)


def _relayout(spec: ScenarioSpec, actions: list[str]) -> ScenarioSpec:
    """Переносит события по равномерному расписанию; при нехватке места отбрасывает хвост."""
    events = list(spec.events)
    n, delta = spec.stream.n_samples, spec.evaluation.acceptance_window
    warmup = spec.evaluation.warmup
    while events and sum(e.width + delta for e in events) > n - warmup:
        events.pop()
        actions.append(f"C4–C6: событие {len(events)} отброшено — не хватает длины потока")
    centers = layout_uniform([e.width for e in events], warmup, n, delta)
    moved = [e.model_copy(update={"position": c}) for e, c in zip(events, centers, strict=True)]
    actions.append(f"C4–C6: события перенесены на позиции {centers}")
    return spec.model_copy(update={"events": tuple(moved)})


def repair(spec: ScenarioSpec, *, max_iter: int | None = None) -> RepairResult:
    """Делает сценарий допустимым минимальными изменениями.

    Порядок действий:

    1. При нарушении C4–C6 события переносятся по равномерному расписанию,
       а если их следы не помещаются в поток, последние события отбрасываются.
    2. Затем по одному событию, от первого к последнему, применяются
       допустимые значения, предложенные валидатором (C9–C12). Действовать
       приходится последовательно, потому что каждое событие меняет
       состояние, от которого зависит допустимость следующих.

    Raises:
        ScenarioValidationError: если сценарий невозможно исправить
            (например, C7, C8 или C13).
    """
    actions: list[str] = []
    report = check(spec)
    if report.codes() & {"C4", "C5", "C6"}:
        spec = _relayout(spec, actions)
        report = check(spec)
    limit = max_iter if max_iter is not None else 2 * len(spec.events) + 2
    for _ in range(limit):
        fixable: dict[int, list[Violation]] = {}
        for v in report.errors:
            match = _EVENT_PATH.match(v.path)
            if match and v.suggestion is not None and float(v.suggestion) > 0.0:
                fixable.setdefault(int(match.group(1)), []).append(v)
        if not fixable:
            break
        first = min(fixable)
        events = list(spec.events)
        update: dict[str, float] = {}
        for v in fixable[first]:
            fname = _EVENT_PATH.match(v.path).group(2)  # type: ignore[union-attr]
            update[fname] = float(v.suggestion)
            actions.append(f"{v.code}: events[{first}].{fname} → {float(v.suggestion):.6g}")
        events[first] = DriftEvent.model_validate({**events[first].model_dump(), **update})
        spec = spec.model_copy(update={"events": tuple(events)})
        report = check(spec)
    if not report.ok:
        raise ScenarioValidationError(report)
    return RepairResult(spec, actions, report)

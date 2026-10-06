"""Реестр семейств генераторов потоков и их возможностей.

Ответственный: Будаев К. В.

Семейство объявляет, какие виды и формы дрейфа оно поддерживает, и даёт
функцию проверки ограничений, специфичных для семейства. На этапе 1
регистрируется собственное калиброванное семейство ``hyperplane_gauss``.
На этапе 2 добавляются обёртки классических генераторов river
(``river:*``) и внедрение дрейфа в реальные данные (``real:*``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from scendrift.scenario.enums import DriftForm, DriftKind
from scendrift.scenario.report import Violation
from scendrift.scenario.schema import ScenarioSpec

__all__ = ["FamilyInfo", "register_family", "get_family", "registered_families"]

FamilyCheck = Callable[[ScenarioSpec], list[Violation]]


@dataclass(frozen=True)
class FamilyInfo:
    """Описание семейства генераторов.

    Attributes:
        name: идентификатор семейства (значение ``stream.family``).
        description: краткое описание.
        kinds: поддерживаемые виды дрейфа.
        forms: поддерживаемые формы перехода.
        supports_recurring: поддерживается ли возврат к прежнему концепту.
        calibrated: величина дрейфа имеет точную семантику единой шкалы.
        check: проверка ограничений семейства (возвращает нарушения).
    """

    name: str
    description: str
    kinds: frozenset[DriftKind]
    forms: frozenset[DriftForm]
    supports_recurring: bool
    calibrated: bool
    check: FamilyCheck | None = None


_REGISTRY: dict[str, FamilyInfo] = {}


def register_family(info: FamilyInfo, *, overwrite: bool = False) -> None:
    """Регистрирует семейство генераторов.

    Raises:
        KeyError: если семейство уже зарегистрировано и ``overwrite`` ложно.
    """
    if info.name in _REGISTRY and not overwrite:
        raise KeyError(f"семейство {info.name!r} уже зарегистрировано")
    _REGISTRY[info.name] = info


def get_family(name: str) -> FamilyInfo | None:
    """Возвращает описание семейства или None, если оно не зарегистрировано."""
    return _REGISTRY.get(name)


def registered_families() -> list[str]:
    """Имена зарегистрированных семейств в порядке регистрации."""
    return list(_REGISTRY)


def _check_hyperplane(spec: ScenarioSpec) -> list[Violation]:
    from scendrift.scenario.semantics import build_chain

    return build_chain(spec).violations


register_family(
    FamilyInfo(
        name="hyperplane_gauss",
        description=(
            "Калиброванное семейство: гиперплоскость над гауссовыми признаками, "
            "три вида дрейфа с точной семантикой величины"
        ),
        kinds=frozenset(DriftKind),
        forms=frozenset(DriftForm),
        supports_recurring=True,
        calibrated=True,
        check=_check_hyperplane,
    )
)

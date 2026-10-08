"""Автоматическое формирование и параметризация сценариев.

Ответственный: Будаев К. В.

Модули:

* :mod:`~scendrift.formation.samplers` — планы эксперимента (сетка, случайная
  выборка, латинский гиперкуб, последовательности Соболя) и метрики равномерности;
* :mod:`~scendrift.formation.mapping` — отображение точки пространства
  параметров в сценарий и способы учёта ограничений (reject, repair, adapt);
* :mod:`~scendrift.formation.templates` — шаблоны и наследование ``extends``;
* :mod:`~scendrift.formation.compose` — композиция сценария из блоков;
* :mod:`~scendrift.formation.difficulty` — шкала сложности λ и каталог профилей;
* :mod:`~scendrift.formation.suite` — наборы сценариев, манифест, версии.

Функция композиции вызывается как ``scendrift.formation.compose.compose``:
имя ``compose`` на уровне пакета занято одноимённым модулем.
"""

from scendrift.formation.compose import Drift, Recur, Stable
from scendrift.formation.difficulty import DIFFICULTY, DifficultyScale, Factor
from scendrift.formation.mapping import point_to_spec, realize, spec_to_point, summarize
from scendrift.formation.samplers import design_metrics, make_sampler
from scendrift.formation.suite import Suite, SuiteConfig, build_suite, load_suite
from scendrift.formation.templates import catalog_names, deep_merge, resolve

__all__ = [
    "Stable",
    "Drift",
    "Recur",
    "DIFFICULTY",
    "DifficultyScale",
    "Factor",
    "point_to_spec",
    "realize",
    "spec_to_point",
    "summarize",
    "design_metrics",
    "make_sampler",
    "Suite",
    "SuiteConfig",
    "build_suite",
    "load_suite",
    "catalog_names",
    "deep_merge",
    "resolve",
]

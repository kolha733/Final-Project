"""Генераторы потоков с контролируемым дрейфом и ground truth.

Ответственный: Будаев К. В.

Импорт пакета регистрирует семейства генераторов:

* ``hyperplane_gauss`` — калиброванное семейство (:mod:`.hyperplane`);
* ``river:*`` — классические генераторы river (:mod:`.classic`);
* ``real:*`` — внедрение дрейфа в реальные наборы данных (:mod:`.real`).

Главная точка входа — :func:`generate`.
"""

from scendrift.generators.base import FamilyGenerator, generate, get_generator, register_generator
from scendrift.generators.classic import ClassicGenerator
from scendrift.generators.hyperplane import HyperplaneGenerator
from scendrift.generators.real import RealGenerator
from scendrift.generators.transitions import Timeline, timeline, transition

__all__ = [
    "ClassicGenerator",
    "FamilyGenerator",
    "HyperplaneGenerator",
    "RealGenerator",
    "Timeline",
    "generate",
    "get_generator",
    "register_generator",
    "timeline",
    "transition",
]

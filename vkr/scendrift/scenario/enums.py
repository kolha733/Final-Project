"""Перечисления формальной модели сценария дрейфа.

Ответственный: Будаев К. В.

Повторяющийся дрейф (recurring) здесь сознательно не является формой
перехода. Возврат к ранее встречавшемуся концепту может быть как внезапным,
так и постепенным, поэтому он задаётся отдельно, полем ``returns_to``
события дрейфа (см. :class:`scendrift.scenario.schema.DriftEvent`).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["DriftForm", "DriftKind", "TransitionShape"]


class DriftForm(StrEnum):
    """Форма дрейфа во времени: как новый концепт замещает старый.

    * ``SUDDEN`` — внезапный дрейф (sudden/abrupt): мгновенная смена концепта, ширина 0.
    * ``GRADUAL`` — постепенный дрейф (gradual): на интервале перехода объекты
      порождаются то старым, то новым концептом, причём вероятность нового растёт.
    * ``INCREMENTAL`` — инкрементальный дрейф (incremental): параметры концепта
      непрерывно меняются, проходя через промежуточные концепты.
    """

    SUDDEN = "sudden"
    GRADUAL = "gradual"
    INCREMENTAL = "incremental"


class DriftKind(StrEnum):
    """Вид дрейфа: какая компонента совместного распределения P(X, y) меняется.

    * ``REAL`` — реальный дрейф (real drift): меняется P(y|X), а P(X) и P(y)
      сохраняются.
    * ``VIRTUAL`` — виртуальный дрейф (virtual drift, covariate shift): меняется
      P(X), а P(y|X) и P(y) сохраняются.
    * ``PRIOR`` — дрейф априорных вероятностей (prior/label shift): меняется P(y),
      а P(X|y) сохраняется.
    """

    REAL = "real"
    VIRTUAL = "virtual"
    PRIOR = "prior"


class TransitionShape(StrEnum):
    """Профиль функции перехода p(t) на интервале [onset, end).

    * ``LINEAR`` — линейный рост от 0 до 1.
    * ``SIGMOID`` — сигмоида как в MOA (ConceptDriftStream), отнормированная
      так, что p(onset) = 0 и p(end) = 1 ровно на границах интервала.
    """

    LINEAR = "linear"
    SIGMOID = "sigmoid"

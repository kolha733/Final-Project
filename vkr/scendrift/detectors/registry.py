"""Адаптеры детекторов дрейфа river и базовые линии.

Ответственный: Гарифзянов Т. Р.

Все детекторы приводятся к интерфейсу :class:`~scendrift.interfaces.DriftDetector`:
``update(value) -> bool``, ``reset()``, ``clone()``. Детектор получает
индикатор ошибки базовой модели e_t ∈ {0, 1}.

Детекторы river 0.26.1 (заимствованы):

* ADWIN [Bifet, Gavaldà, 2007] — адаптивное окно, двусторонний;
* KSWIN [Raab et al., 2020] — критерий Колмогорова–Смирнова в окне, двусторонний;
* Page–Hinkley [Page, 1954] — кумулятивная сумма; ``mode="both"`` (по
  умолчанию в river) и ``mode="up"`` (только рост ошибки);
* DDM [Gama et al., 2004], EDDM [Baena-García et al., 2006], HDDM_A и
  HDDM_W [Frías-Blanco et al., 2015], FHDDM [Pesaranghader, Viktor, 2016] —
  детекторы по бинарному потоку ошибок; в используемой конфигурации
  реагируют только на рост ошибки.

FHDDM получает индикатор **верного** ответа 1 − e_t. В river 0.26.1
описание входа FHDDM («1 — ошибка») противоречит реализации: реализация
сигналит, когда среднее входа падает ниже своего максимума. Это
формулировка исходной статьи, где в окне хранятся индикаторы верного
ответа. При подаче ошибок FHDDM реагировал бы на улучшение модели, а не
на дрейф; проверка — в п. 4.7.

Базовые линии (собственная реализация):

* ``NoDrift`` — никогда не срабатывает: accuracy без адаптации;
* ``Periodic`` — срабатывает каждые ``period`` объектов после сброса.
  Проверяет «иллюзию прогресса» [Bifet, 2017]: при period ≤ Δ в каждое окно
  допуска попадает срабатывание, и recall равен 1 при любом дрейфе;
* ``Oracle`` — срабатывает в центрах τ_k истинных событий (при внезапном
  дрейфе — в момент дрейфа). Это опорная точка для accuracy: сброс модели
  тогда, когда новый концепт становится преобладающим.

Имя детектора с нестандартными параметрами включает их, например
``ADWIN(delta=0.01)``: по имени кэшируются результаты.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from river import drift
from river.drift import binary

from scendrift.interfaces import DriftDetector, GroundTruth

__all__ = [
    "DetectorInfo",
    "RiverDetector",
    "NoDrift",
    "Periodic",
    "Oracle",
    "DETECTORS",
    "BASELINES",
    "make_detector",
    "detector_table",
]


@dataclass(frozen=True)
class DetectorInfo:
    """Описание детектора river в реестре.

    Attributes:
        name: имя в отчётах.
        factory: класс river.
        params: параметры по умолчанию (переопределяют умолчания river).
        binary_input: детектор ожидает bool (иначе число).
        two_sided: реагирует и на снижение ошибки.
        source: ключ источника в списке литературы.
        invert: подавать индикатор верного ответа 1 − e вместо ошибки e.
    """

    name: str
    factory: Callable[..., Any]
    params: Mapping[str, Any] = field(default_factory=dict)
    binary_input: bool = False
    two_sided: bool = False
    source: str = ""
    invert: bool = False


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else repr(value)


class RiverDetector:
    """Адаптер детектора river к интерфейсу ``DriftDetector``."""

    def __init__(self, info: DetectorInfo, **overrides: Any) -> None:
        self.info = info
        self.overrides = dict(overrides)
        self.params = {**info.params, **overrides}
        self._inner = info.factory(**self.params)

    @property
    def name(self) -> str:
        if not self.overrides:
            return self.info.name
        args = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(self.overrides.items()))
        return f"{self.info.name}({args})"

    def update(self, value: float) -> bool:
        if self.info.invert:
            value = 1.0 - float(value)
        self._inner.update(bool(value) if self.info.binary_input else float(value))
        return bool(self._inner.drift_detected)

    def reset(self) -> None:
        self._inner = self.info.factory(**self.params)

    def clone(self) -> RiverDetector:
        return RiverDetector(self.info, **self.overrides)

    def __repr__(self) -> str:
        return f"RiverDetector({self.name})"


class NoDrift:
    """Базовая линия: никогда не срабатывает."""

    name = "NoDrift"

    def update(self, value: float) -> bool:
        return False

    def reset(self) -> None:
        return None

    def clone(self) -> NoDrift:
        return NoDrift()


class Periodic:
    """Базовая линия: срабатывает каждые ``period`` объектов после последнего сброса."""

    def __init__(self, period: int = 2_000) -> None:
        if period < 1:
            raise ValueError("period ≥ 1")
        self.period = int(period)
        self._count = 0

    @property
    def name(self) -> str:
        return f"Periodic({self.period})"

    def update(self, value: float) -> bool:
        self._count += 1
        return self._count >= self.period

    def reset(self) -> None:
        self._count = 0

    def clone(self) -> Periodic:
        return Periodic(self.period)


class Oracle:
    """Базовая линия: срабатывает в центрах истинных событий.

    Детектору нужно знать время, поэтому раннер вызывает :meth:`bind` с
    ground truth и номером первого объекта, поданного детектору (W).
    Сброс после срабатывания не сбрасывает часы.
    """

    name = "Oracle"
    needs_ground_truth = True

    def __init__(self) -> None:
        self._times: frozenset[int] = frozenset()
        self._t = 0

    def bind(self, ground_truth: GroundTruth, start: int) -> Oracle:
        """Экземпляр, срабатывающий в моменты τ_k (счёт времени с ``start``)."""
        bound = Oracle()
        bound._times = frozenset(iv.center for iv in ground_truth.intervals)
        bound._t = start
        return bound

    def update(self, value: float) -> bool:
        hit = self._t in self._times
        self._t += 1
        return hit

    def reset(self) -> None:
        return None

    def clone(self) -> Oracle:
        return Oracle()


#: Детекторы river в реестре.
DETECTORS: dict[str, DetectorInfo] = {
    info.name: info
    for info in [
        DetectorInfo("ADWIN", drift.ADWIN, two_sided=True, source="adwin2007"),
        DetectorInfo("KSWIN", drift.KSWIN, {"seed": 0}, two_sided=True, source="kswin2020"),
        DetectorInfo("PageHinkley", drift.PageHinkley, two_sided=True, source="page1954"),
        DetectorInfo("PageHinkley-up", drift.PageHinkley, {"mode": "up"}, source="page1954"),
        DetectorInfo("DDM", binary.DDM, binary_input=True, source="ddm2004"),
        DetectorInfo("EDDM", binary.EDDM, binary_input=True, source="eddm2006"),
        DetectorInfo("HDDM_A", binary.HDDMA, binary_input=True, source="hddm2015"),
        DetectorInfo("HDDM_W", binary.HDDMW, binary_input=True, source="hddm2015"),
        DetectorInfo("FHDDM", binary.FHDDM, binary_input=True, source="fhddm2016", invert=True),
    ]
}

#: Базовые линии.
BASELINES: dict[str, Callable[..., DriftDetector]] = {
    "NoDrift": NoDrift,
    "Periodic": Periodic,
    "Oracle": Oracle,
}


def make_detector(name: str, **params: Any) -> DriftDetector:
    """Детектор или базовая линия по имени из реестра.

    Raises:
        KeyError: если имя неизвестно.
    """
    if name in DETECTORS:
        return RiverDetector(DETECTORS[name], **params)
    if name in BASELINES:
        return BASELINES[name](**params)
    known = ", ".join([*DETECTORS, *BASELINES])
    raise KeyError(f"неизвестный детектор {name!r}; доступны: {known}")


def detector_table(names: Sequence[str] | None = None) -> list[dict[str, str]]:
    """Сводка реестра для таблиц ноутбука и записки."""
    rows = []
    for name in names or DETECTORS:
        info = DETECTORS[name]
        params = ", ".join(f"{k}={_fmt(v)}" for k, v in info.params.items()) or "по умолчанию"
        rows.append(
            {
                "детектор": name,
                "класс river": f"{info.factory.__module__.split('.')[-1]}.{info.factory.__name__}",
                "параметры": params,
                "вход": ("1 − e (bool)" if info.invert else "e (bool)")
                if info.binary_input
                else "e (число)",
                "направление": "оба" if info.two_sided else "рост ошибки",
            }
        )
    return rows

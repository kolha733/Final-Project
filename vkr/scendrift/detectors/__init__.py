"""Адаптеры детекторов дрейфа river и базовые линии.

Ответственный: Гарифзянов Т. Р.
"""

from scendrift.detectors.registry import (
    BASELINES,
    DETECTORS,
    NoDrift,
    Oracle,
    Periodic,
    RiverDetector,
    detector_table,
    make_detector,
)

__all__ = [
    "BASELINES",
    "DETECTORS",
    "NoDrift",
    "Oracle",
    "Periodic",
    "RiverDetector",
    "detector_table",
    "make_detector",
]

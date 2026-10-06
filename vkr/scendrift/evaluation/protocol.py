"""Протокол оценки Ω — составная часть формальной модели сценария.

Ответственный: Гарифзянов Т. Р.

Сценарий S = ⟨B, E, Ω, s⟩ включает протокол оценки Ω = (h, a, W, Δ, r):

* h — базовая (онлайн) модель, ошибки которой отслеживает детектор;
* a — политика адаптации при срабатывании детектора;
* W — длина разогрева (warm-up), в течение которого дрейф не допускается;
* Δ — окно допуска: срабатывание засчитывается как обнаружение события,
  если оно произошло на полуинтервале [onset, end + Δ);
* r — от какой точки отсчитывается задержка обнаружения: начало или центр перехода.

Правила сопоставления срабатываний с событиями и определения метрик
(TP/FP/FN, MTD, MDR, MTFA, MTR) формализованы в разделе 2.6 ноутбука,
реализация вынесена в модуль ``scendrift.evaluation.metrics`` (этап 4).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["AdaptationPolicy", "DelayReference", "EvaluationSpec"]


class AdaptationPolicy(StrEnum):
    """Реакция на срабатывание детектора.

    * ``RESET`` — базовая модель сбрасывается и обучается заново
      (стандартная схема detect-and-reset).
    * ``NONE`` — модель продолжает обучаться, срабатывания только фиксируются.
    """

    RESET = "reset"
    NONE = "none"


class DelayReference(StrEnum):
    """Точка отсчёта задержки обнаружения."""

    ONSET = "onset"
    CENTER = "center"


class EvaluationSpec(BaseModel):
    """Протокол оценки Ω (часть сценария, учитывается в его идентификаторе)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_model: str = Field(
        "gaussian_nb",
        description="h — идентификатор базовой онлайн-модели в реестре моделей.",
    )
    on_detection: AdaptationPolicy = Field(
        AdaptationPolicy.RESET, description="a — политика адаптации при срабатывании."
    )
    warmup: int = Field(
        1_000, ge=0, description="W — длина разогрева: дрейф не начинается раньше W."
    )
    acceptance_window: int = Field(
        2_000,
        ge=1,
        description="Δ — окно допуска: обнаружение засчитывается на [onset, end + Δ).",
    )
    delay_reference: DelayReference = Field(
        DelayReference.ONSET, description="r — точка отсчёта задержки."
    )

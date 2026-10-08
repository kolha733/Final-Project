"""Оценка детекторов: протокол, базовая модель, раннер, метрики, статистика.

Ответственный: Гарифзянов Т. Р.

Модули ``models``, ``runner``, ``metrics``, ``results`` и ``stats``
импортируются явно: протокол оценки нужен уже схеме сценария (п. 4.1),
а остальные модули зависят от генераторов (п. 4.3–4.5).
"""

from scendrift.evaluation.protocol import AdaptationPolicy, DelayReference, EvaluationSpec

__all__ = ["AdaptationPolicy", "DelayReference", "EvaluationSpec"]

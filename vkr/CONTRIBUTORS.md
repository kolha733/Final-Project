# Распределение вклада исполнителей

Работа коллективная. У каждого модуля в docstring указан ответственный; в ноутбуке разделы помечены [К], [Т] или [К+Т]; на схемах архитектуры блоки окрашены по исполнителю.

| Модуль / раздел | Ответственный | Этап |
|---|---|---|
| `scendrift/scenario/enums.py`, `schema.py`, `report.py` | Будаев К. В. | 1 |
| `scendrift/scenario/semantics.py` (калибровка величины, цепочка концептов) | Будаев К. В. | 1 |
| `scendrift/scenario/validation.py`, `families.py` (ограничения C1–C14, предупреждения W1–W4, repair) | Будаев К. В. | 1–2 |
| `scendrift/scenario/space.py`, `taxonomy.py`, `io.py` | Будаев К. В. | 1 |
| `scendrift/evaluation/protocol.py` (протокол оценки Ω) | Гарифзянов Т. Р. | 1 |
| `scendrift/reporting/style.py` (единый стиль графиков) | Гарифзянов Т. Р. | 1 |
| `scendrift/interfaces.py` (контракты модулей) | Будаев К. В. (данные), Гарифзянов Т. Р. (оценка) | 1 |
| `tools/diagrams.py`, `tools/build_notebook.py`, `tools/gost.py` | совместно | 1 |
| Ноутбук: разд. 1.1–1.2, 2.1–2.5 | Будаев К. В. | 1 |
| Ноутбук: разд. 1.3–1.4 (обзор детекторов и аналогов), 2.6 | Гарифзянов Т. Р. | 1 |
| Ноутбук: разд. 0, 1.5, 3, 6, 7 | совместно | 1 |
| `scendrift/generators/transitions.py`, `base.py`, `ground_truth.py` (функции перехода, конвейер генерации, ground truth) | Будаев К. В. | 2 |
| `scendrift/generators/hyperplane.py` (калиброванное семейство) | Будаев К. В. | 2 |
| `scendrift/generators/classic.py` (обёртка генераторов river, вычисление TV смены концепта) | Будаев К. В. | 2 |
| `scendrift/generators/real.py` (внедрение дрейфа в реальные данные с точной калибровкой) | Будаев К. В. | 2 |
| `scendrift/generators/diagnostics.py`, `tests/test_generators.py` | Будаев К. В. | 2 |
| Ноутбук: разд. 4.1–4.5, 5.1 (Э1) | Будаев К. В. | 1–2 |
| `scendrift/formation/samplers.py` (планы эксперимента, метрики равномерности) | Будаев К. В. | 3 |
| `scendrift/formation/mapping.py` (параметризация; способы учёта ограничений, в т. ч. условные домены) | Будаев К. В. | 3 |
| `scendrift/formation/templates.py`, `compose.py` (наследование, композиция из блоков) | Будаев К. В. | 3 |
| `scendrift/formation/difficulty.py`, `suite.py` (шкала λ и каталог; наборы, манифест, версии) | Будаев К. В. | 3 |
| `tests/test_formation.py`, `configs/suites/` | Будаев К. В. | 3 |
| Ноутбук: разд. 2.7, 4.6 | Будаев К. В. | 3 |
| `scendrift/detectors/registry.py` (адаптеры river, базовые линии NoDrift, Periodic, Oracle; проверка направления и входа FHDDM) | Гарифзянов Т. Р. | 4 |
| `scendrift/evaluation/models.py` (векторная prequential-модель, эквивалентная river GaussianNB) | Гарифзянов Т. Р. | 4 |
| `scendrift/evaluation/runner.py` (раннер, повторы, кэш, параллелизм) | Гарифзянов Т. Р. | 4 |
| `scendrift/evaluation/metrics.py`, `results.py`, `stats.py` (метрики п. 2.6, таблица результатов, статистика по Demšar) | Гарифзянов Т. Р. | 4 |
| `scendrift/reporting/plots.py`, `report.py` (CD-диаграмма, растр срабатываний, отчёты) | Гарифзянов Т. Р. | 4 |
| `tests/test_evaluation.py`; ноутбук: разд. 4.7–4.11 | Гарифзянов Т. Р. | 4 |
| Эксперимент Э1 | Будаев К. В. | 2 |
| Эксперимент Э7 (п. 5.7) | Будаев К. В. | 5 |
| Эксперименты Э2, Э3 (с Э9), Э5, Э6, Э8 (п. 5.2, 5.3, 5.5, 5.6, 5.8); `scendrift/evaluation/analysis.py`, `tests/test_analysis.py` | Гарифзянов Т. Р. | 5 |
| Эксперимент Э4 (п. 5.4), обсуждение и угрозы валидности (п. 5.9) | совместно | 5 |

Заимствованные компоненты (не являются вкладом авторов): детекторы, алгоритм базовой модели (гауссовский наивный Байес river — собственная только его векторная реализация), классические генераторы (SEA, Agrawal, STAGGER, Sine, Mixed) и наборы данных Bananas и Phishing — river 0.26.1; генераторы LHS и последовательностей Соболя, центрированное расхождение, критерий Колмогорова–Смирнова, статистические тесты, ранговые корреляции, T-функция Оуэна — SciPy; валидация схемы — pydantic; дерево решений и кросс-валидация — scikit-learn.

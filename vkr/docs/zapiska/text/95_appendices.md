# ПРИЛОЖЕНИЕ А | (справочное) | Каталог ограничений допустимости сценария

Каталог ограничений (C1–C14) и предупреждений (W1–W4) строится из кода фреймворка (модуль scendrift.scenario.report) при сборке пояснительной записки. Ограничения уровня «событие» проверяются при создании объекта, остальные — валидатором с учётом возможностей семейства генератора (подраздел 2.5).

Table: Ограничения допустимости и предупреждения {#tbl:constraints}

<!-- table:constraints -->

# ПРИЛОЖЕНИЕ Б | (справочное) | Примеры описания сценариев и конфигурации набора

Пример Б.1 — сценарий в канонической форме: один внезапный реальный дрейф величины 0,25 в середине потока.

```yaml
schema_version: "1.0"
name: sudden_real_single
stream:
  family: hyperplane_gauss
  n_samples: 20000
  n_features: 10
  minority_share: 0.5
  label_noise: 0.0
  concept_seed: 0
events:
  - {position: 10000, form: sudden, kind: real, magnitude: 0.25, affected_share: 1.0}
evaluation:
  base_model: gaussian_nb
  on_detection: reset
  warmup: 1000
  acceptance_window: 2000
seed: 1
```

Пример Б.2 — сценарий в компактной форме: инкрементальные дрейфы со случайным расписанием и частичным охватом признаков (величина ограничена сверху ограничением C10).

```yaml
schema_version: "1.0"
name: incremental_random_partial
stream: {n_samples: 40000, n_features: 15, minority_share: 0.5, concept_seed: 5}
drifts:
  schedule: {mode: random, count: 4, min_gap: 500}
  defaults: {form: incremental, kind: real, width: 3000, magnitude: 0.1,
             affected_share: 0.3}
  overrides:
    - {index: 3, shape: sigmoid}
seed: 3
```

Пример Б.3 — наследование профиля каталога с переопределением величины дрейфа.

```yaml
extends: catalog/medium
drifts: {defaults: {magnitude: 0.3}}
```

Пример Б.4 — конфигурация набора сценариев основного бенчмарка Э2 (режим FAST).

```yaml
name: e2_fast
version: 1.0.0
description: Основной бенчмарк Э2 (FAST) — план Соболя по 11 параметрам
template:
  extends: catalog/base
design: {method: sobol, n: 64, seed: 2027}
constraints: adapt
```

Пример Б.5 — формирование набора, сохранение с манифестом и проверка целостности на языке Python.

```python
import yaml
from scendrift.formation import SuiteConfig, build_suite, load_suite

config = SuiteConfig.model_validate(
    yaml.safe_load(open("configs/suites/e2_fast.yaml", encoding="utf-8")))
suite = build_suite(config)            # план Соболя -> сценарии -> условные домены
suite.save("results/suites/e2_fast")   # manifest.yaml и scenarios/*.yaml
assert load_suite("results/suites/e2_fast").suite_id == suite.suite_id
```

# ПРИЛОЖЕНИЕ В | (обязательное) | Распределение работ между исполнителями

Работа выполнена коллективом из двух исполнителей: Будаев Константин Владимирович отвечал за сценарии и данные, Гарифзянов Тимур Русланович — за оценку и анализ. Ответственный исполнитель указан также в строке документации каждого модуля фреймворка и в заголовке каждого раздела исполняемого ноутбука. Распределение работ приведено в таблице @tbl:roles.

Table: Распределение работ между исполнителями {#tbl:roles}

| Часть работы | Будаев К. В. | Гарифзянов Т. Р. | Совместно |
|------------------|------------------------------|------------------------------|------------------------|
| Введение | — | — | целиком |
| Глава 1 | 1.1, 1.2: потоки, дрейф, таксономия | 1.3, 1.4: детекторы, средства оценки, метрики | 1.5, 1.6 |
| Глава 2 | 2.1–2.5, 2.7: сценарий, шкала величины, пространство, ограничения, формирование наборов | 2.6: протокол оценки и метрики | 2.8 |
| Глава 3 | 3.3–3.5: формат сценария, генерация, формирование наборов | 3.6: детекторы, раннер, метрики, статистика, отчёты | 3.1, 3.2, 3.7, 3.8 |
| Глава 4 | 4.2 (Э1), 4.3, 4.9 (Э7) | 4.4 (Э2), 4.5 (Э3), 4.7 (Э5), 4.8 (Э6), 4.10 (Э8) | 4.1, 4.6 (Э4), 4.11, 4.12 |
| Заключение | — | — | целиком |
| Модули фреймворка | scenario, generators, formation | detectors, evaluation, reporting | interfaces, сборка ноутбука и записки |

# ПРИЛОЖЕНИЕ Г | (справочное) | Структура программного комплекса и воспроизведение результатов

Программный комплекс размещён в репозитории и включает:

- VKR_Budaev_Garifzyanov.ipynb — исполняемый ноутбук, основной артефакт работы: текст и работающий фреймворк, выполненные сверху вниз;
- scendrift/ — пакет фреймворка (подпакеты scenario, formation, generators, detectors, evaluation, reporting);
- notebook/ — исходные тексты разделов ноутбука и список литературы references.yaml;
- configs/ — примеры сценариев, конфигурации наборов и JSON Schema сценария;
- tests/ — модульные тесты pytest;
- tools/ — сборка ноутбука, схемы архитектуры, оформление списка литературы по ГОСТ Р 7.0.100–2018;
- results/ — результаты экспериментов в формате CSV и сохранённые наборы сценариев;
- docs/zapiska/ — исходные тексты и сборщик настоящей пояснительной записки.

Результаты воспроизводятся командами (из каталога vkr):

```bash
python -m pip install -r requirements.txt
python -m pytest                                  # модульные тесты
python tools/build_notebook.py --execute          # ноутбук и все эксперименты (FAST)
SCENDRIFT_MODE=FULL python tools/build_notebook.py --execute   # полный объём
python docs/zapiska/build_docx.py                 # пояснительная записка (.docx и .pdf)
```

Сборщик записки подставляет в текст числа и таблицы из файлов results/*.csv, нумерует рисунки, таблицы, формулы и источники, оформляет документ по ГОСТ 7.32–2017 [@gost732] и список источников по ГОСТ Р 7.0.100–2018 [@gost70100]; все параметры оформления заданы в файле docs/zapiska/format_config.py.

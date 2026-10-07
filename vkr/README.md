# scendrift — бенчмарки детекторов концептуального дрейфа на параметризованных сценариях

Выпускная квалификационная работа (коллективная), Финансовый университет при Правительстве РФ, факультет информационных технологий и анализа больших данных, кафедра искусственного интеллекта.

**Тема:** «Разработка фреймворка генерации бенчмарков систем выявления концептуального дрейфа с автоматизированным формированием и параметризацией сценариев».

**Исполнители:** Будаев К. В., Гарифзянов Т. Р. (группа ПМ23-5).
**Научный руководитель:** ассистент Окунева Э. А.

## Состав

| Путь | Назначение |
|---|---|
| `VKR_Budaev_Garifzyanov.ipynb` | основной артефакт: текст работы и работающий фреймворк (собирается из `notebook/`, выполнен сверху вниз) |
| `scendrift/` | пакет фреймворка (источник истины для кода) |
| `notebook/*.md` | тексты разделов ноутбука; `references.yaml` — единый список литературы |
| `configs/examples/` | примеры сценариев (YAML) |
| `configs/suites/` | конфигурации наборов сценариев |
| `configs/schema/scenario.schema.json` | JSON Schema сценария (генерируется из pydantic) |
| `tests/` | тесты pytest |
| `tools/` | сборка ноутбука, схемы архитектуры, оформление литературы по ГОСТ |
| `docs/figures/` | рисунки для ноутбука, записки и презентации (`docs/figures/zapiska/` — копии без общего заголовка) |
| `docs/zapiska/` | пояснительная записка: текст (`text/*.md`), числа из результатов (`facts.py`), параметры оформления по ГОСТ 7.32 (`format_config.py`), сборщик (`build_docx.py`) и готовые `.docx` и `.pdf` |
| `PLAN.md`, `CONTRIBUTORS.md` | план работ и распределение вклада |

## Быстрый старт

```bash
cd vkr
python -m pip install -r requirements.txt
python -m pytest                              # тесты
python tools/build_notebook.py --execute      # собрать и выполнить ноутбук
python tools/build_notebook.py --check        # сверить код в ноутбуке с модулями
python docs/zapiska/build_docx.py             # собрать пояснительную записку (.docx и .pdf; нужен LibreOffice)
```

Пример: загрузить сценарий, проверить его и вывести фактическую величину дрейфа.

```python
from scendrift.scenario import io, semantics
from scendrift.scenario.validation import check

spec = io.load("configs/examples/03_recurring_mixed.yaml")
print(io.scenario_id(spec), check(spec))
for g in semantics.build_chain(spec).geometry:
    print(g.index, g.kind, g.realized)
```

Порождение потока по сценарию (любое зарегистрированное семейство: `hyperplane_gauss`, `river:*`, `real:*`):

```python
from scendrift.generators import generate

data = generate(spec, seed=1)           # X, y, y_clean, concept, progress
print(data.X.shape, data.ground_truth.to_records()[0])
```

Автоматическое формирование набора сценариев по конфигурации и проверка целостности:

```python
import yaml
from scendrift.formation import SuiteConfig, build_suite, load_suite

config = SuiteConfig.model_validate(yaml.safe_load(open("configs/suites/e2_fast.yaml")))
suite = build_suite(config)                 # план Соболя → сценарии → условные домены
suite.save("results/suites/e2_fast")        # manifest.yaml + scenarios/*.yaml
assert load_suite("results/suites/e2_fast").suite_id == suite.suite_id
```

Сценарий может наследовать профиль каталога или собираться из блоков:

```yaml
extends: catalog/medium                     # easy / medium / hard — точки шкалы сложности λ
drifts: {defaults: {magnitude: 0.3}}
```

Оценка детекторов на наборе сценариев, статистика и отчёт:

```python
from scendrift.detectors import DETECTORS, make_detector
from scendrift.evaluation.runner import run_benchmark
from scendrift.evaluation.results import runs_to_frame
from scendrift.evaluation.stats import average_ranks, block_matrix, friedman

detectors = [make_detector(n) for n in DETECTORS] + [make_detector("NoDrift"), make_detector("Oracle")]
runs = run_benchmark(suite.scenarios, detectors, repeats=3, n_jobs=4, cache_dir="results/cache")
frame = runs_to_frame(runs, suite.scenarios)          # метрики п. 2.6 + параметры сценариев
matrix, _ = block_matrix(frame[frame.detector.isin(DETECTORS)], "f1")
print(average_ranks(matrix), friedman(matrix))
```

## Как править

- **Код** правится в `scendrift/*.py`.
- **Текст** — в `notebook/*.md`.
- После правок ноутбук пересобирается командой `python tools/build_notebook.py --execute`. Правки, сделанные прямо в `.ipynb`, при пересборке теряются.
- Ссылки на литературу в тексте пишутся ключами `[@key]` из `notebook/references.yaml`. Сборщик нумерует их по порядку первого упоминания.

## Эксперименты

Глава 5 ноутбука содержит эксперименты Э1–Э8. Сырые результаты лежат в `results/*.csv`, рисунки — в `docs/figures/`, проверка утверждений выводов — в `results/chapter5_claims.csv`. Прогоны кэшируются в `results/cache/` (не хранится в git); первый запуск в режиме FAST на 4 ядрах занимает около 15 минут, повторный — несколько минут.

## Режимы запуска

- `SCENDRIFT_MODE=FAST` (по умолчанию) — сокращённые эксперименты, минуты на ноутбуке.
- `SCENDRIFT_MODE=FULL` — полный эксперимент для сервера или Colab Pro: больше seed и сценариев в Э1, полный объём Э2–Э8 (этап 5).

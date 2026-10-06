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
| `configs/schema/scenario.schema.json` | JSON Schema сценария (генерируется из pydantic) |
| `tests/` | тесты pytest |
| `tools/` | сборка ноутбука, схемы архитектуры, оформление литературы по ГОСТ |
| `docs/figures/` | рисунки для ноутбука, записки и презентации |
| `PLAN.md`, `CONTRIBUTORS.md` | план работ и распределение вклада |

## Быстрый старт

```bash
cd vkr
python -m pip install -r requirements.txt
python -m pytest                              # тесты
python tools/build_notebook.py --execute      # собрать и выполнить ноутбук
python tools/build_notebook.py --check        # сверить код в ноутбуке с модулями
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

## Как править

- **Код** правится в `scendrift/*.py`.
- **Текст** — в `notebook/*.md`.
- После правок ноутбук пересобирается командой `python tools/build_notebook.py --execute`. Правки, сделанные прямо в `.ipynb`, при пересборке теряются.
- Ссылки на литературу в тексте пишутся ключами `[@key]` из `notebook/references.yaml`. Сборщик нумерует их по порядку первого упоминания.

## Режимы запуска

- `SCENDRIFT_MODE=FAST` (по умолчанию) — сокращённые эксперименты, минуты на ноутбуке.
- `SCENDRIFT_MODE=FULL` — полный эксперимент для сервера или Colab Pro (этап 5).

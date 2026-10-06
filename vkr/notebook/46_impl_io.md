### 4.6. Сериализация и идентификация сценариев [К]

Сценарий сохраняется в YAML или JSON. Его идентификатор — первые 12 шестнадцатеричных знаков SHA-256 от канонического JSON смысловых полей ⟨B, E, Ω, s⟩. Описательные поля (имя, теги, метаданные происхождения) в хэш не входят. Отсюда два свойства:
1. переименование сценария не меняет его идентификатор;
2. любое смысловое изменение меняет.

Дополнительно вычисляется `stream_id` — хэш только того, что влияет на данные (без протокола оценки). Он нужен для кэширования потоков. Версионирование наборов сценариев (манифесты) добавляется на этапе 3.
```writefile scendrift/scenario/io.py
```
<!-- cell -->
Загрузим все примеры из п. 3.3, проверим их и выведем идентификаторы. Затем проверим сохранение и загрузку без потерь (round-trip) и выгрузим JSON Schema.
```python
import json

from scendrift.scenario import io as sio

rows = []
for path in sorted(Path("configs/examples").glob("*.yaml")):
    spec_ex = sio.load(path)
    report = check(spec_ex)
    assert report.ok, report
    assert sio.loads_yaml(sio.dumps_yaml(spec_ex)) == spec_ex  # round-trip без потерь
    rows.append({"файл": path.name, "id": sio.scenario_id(spec_ex),
                 "stream_id": sio.stream_id(spec_ex), "K": spec_ex.n_events,
                 "предупреждения": ", ".join(sorted(report.codes())) or "—",
                 "таксономия": signature(spec_ex)})
display(pd.DataFrame(rows))

renamed = demo.model_copy(update={"name": "другое имя", "tags": ("x",)})
assert sio.scenario_id(renamed) == sio.scenario_id(demo)
assert sio.scenario_id(demo.with_seed(43)) != sio.scenario_id(demo)

schema_path = sio.export_json_schema("configs/schema/scenario.schema.json")
schema = json.loads(schema_path.read_text(encoding="utf-8"))
print(f"JSON Schema: {schema_path}, определений: {len(schema['$defs'])}")
print(json.dumps(schema["$defs"]["DriftEvent"]["properties"]["magnitude"], ensure_ascii=False,
                 indent=2))
```
<!-- cell -->
### 4.7–4.11. Детекторы, раннер, метрики, статистика, отчёты [Т]

> **Ожидает реализации (этап 4).** Протокол оценки Ω (п. 4.1.2) и интерфейсы `DriftDetector`, `Metric`, `DetectionRun` (п. 3.4) уже зафиксированы. Формальные определения метрик даны в п. 2.6.

## 4. Реализация

Каждый модуль приводится полностью в ячейке `%%writefile`, которая создаёт файл пакета. Перед ячейкой объясняется назначение модуля, после неё идёт демонстрация. Оформление кода соответствует PEP 8 (проверяется `ruff`), у функций есть аннотации типов и docstrings.

### 4.1. Формальная модель сценария: схема, семантика, проверка [К]

#### 4.1.1. Перечисления

Формы перехода, виды дрейфа и профили функции перехода. Повторяющийся дрейф — не форма, а отдельное поле события (см. п. 2.2).
```writefile scendrift/scenario/enums.py
```
<!-- cell -->
#### 4.1.2. Протокол оценки Ω [Т]

Протокол оценки входит в сценарий и в его идентификатор: от выбора $W$ и $\Delta$ зависят результаты. На данные протокол не влияет.
```writefile scendrift/evaluation/__init__.py
```
```writefile scendrift/evaluation/protocol.py
```
<!-- cell -->
#### 4.1.3. Схема сценария

Pydantic-модели `StreamSpec` (B), `DriftEvent` ($e_k$) и `ScenarioSpec` (S), а также компактная форма `DriftPlan` с функцией развёртывания `expand_plan`. Модели неизменяемы (`frozen=True`) и запрещают неизвестные поля (`extra="forbid"`): опечатка в YAML-файле даёт ошибку, а не молча игнорируется.
```writefile scendrift/scenario/schema.py
```
<!-- cell -->
#### 4.1.4. Отчёт о проверке и каталог ограничений

`CONSTRAINTS` — единый справочник ограничений: из него строятся таблицы в ноутбуке и в пояснительной записке.
```writefile scendrift/scenario/report.py
```
```writefile scendrift/scenario/__init__.py
```
<!-- cell -->
#### 4.1.5. Семантика величины дрейфа и цепочка концептов

Модуль реализует утверждения 1–6 из п. 2.3:
- $\Phi_2$ через T-функцию Оуэна (`scipy.special.owens_t`), точно до машинной точности;
- калибровку угла поворота методом Брента;
- сдвиг заданной величины TV (утверждение 5);
- правило prior-дрейфа.

Функция `build_chain` вычисляет цепочку концептов $c_0 \to c_1 \to \dots \to c_K$ и разложение каждого перехода по группам параметров, не порождая данных. Попутно она:
- отклоняет возврат, не меняющий распределение (C7);
- предупреждает о реальном дрейфе при $\pi_k \ne \pi_0$ (W3).
```writefile scendrift/scenario/semantics.py
```
<!-- cell -->
#### 4.1.6. Реестр семейств генераторов

Семейство объявляет поддерживаемые виды и формы дрейфа и свою функцию проверки. На этапе 2 сюда добавятся обёртки классических генераторов river и внедрение дрейфа в реальные данные.
```writefile scendrift/scenario/families.py
```
<!-- cell -->
#### 4.1.7. Валидатор и исправление сценариев

`check` проверяет ограничения C4–C13 и выдаёт предупреждения W1–W2. `repair` делает сценарий допустимым минимальными изменениями.

Исправление идёт последовательно, от первого события к последнему, потому что каждое событие меняет состояние, от которого зависит допустимость следующих. Пример: после prior-дрейфа $0{,}5 \to 0{,}99$ следующий prior-дрейф величины 0,5 возможен только вниз.
```writefile scendrift/scenario/validation.py
```
<!-- cell -->
#### 4.1.8. Пространство параметров и таксономия

Домены (непрерывный, целочисленный, категориальный, в том числе с логарифмической шкалой) умеют отображать единичный куб $[0, 1)^D$ в значения параметров. Это общая основа для всех планов эксперимента этапа 3.
```writefile scendrift/scenario/space.py
```
```writefile scendrift/scenario/taxonomy.py
```
<!-- cell -->
#### 4.1.9. Демонстрация: сценарий, проверка, цепочка концептов

Построим сценарий из примера 3 (п. 3.3) программно, в компактной форме. В нём три события:
1. постепенный реальный дрейф;
2. виртуальный дрейф;
3. возврат к исходному концепту.

Развернём сценарий в каноническую форму и проверим.
```python
from scendrift.evaluation.protocol import EvaluationSpec
from scendrift.scenario import DriftPlan, ScenarioSpec, StreamSpec, expand_plan
from scendrift.scenario import semantics as sem
from scendrift.scenario.taxonomy import signature
from scendrift.scenario.validation import check, repair

stream = StreamSpec(n_samples=20_000, n_features=10, minority_share=0.3, label_noise=0.05,
                    concept_seed=7)
plan = DriftPlan.model_validate({
    "schedule": {"mode": "uniform", "count": 3},
    "defaults": {"form": "gradual", "kind": "real", "width": 1000, "magnitude": 0.2,
                 "affected_share": 0.5},
    "overrides": [{"index": 1, "kind": "virtual", "magnitude": 0.4},
                  {"index": 2, "returns_to": 0}],
})
evaluation = EvaluationSpec()
demo = ScenarioSpec(name="recurring_mixed", stream=stream,
                    events=expand_plan(plan, stream, evaluation), evaluation=evaluation, seed=42)

events_table = pd.DataFrame([
    {"k": k, "onset": e.onset, "τ": e.position, "end": e.end, "форма": e.form.value,
     "вид": e.kind.value if e.kind else "возврат", "m": e.magnitude,
     "α": e.affected_share, "ρ": e.returns_to}
    for k, e in enumerate(demo.events)
])
display(events_table)
print("Таксономия:", signature(demo))
print(check(demo))
```
<!-- cell -->
Сценарий допустим. Предупреждение W2 напоминает, что второе событие (в таблицах $k = 1$, нумерация с нуля) — виртуальный дрейф: детекторы, отслеживающие ошибку классификатора, не обязаны его замечать.

Цепочка концептов показывает фактическую величину каждого перехода по трём компонентам единой шкалы:
- у обычных событий ненулевой будет только «своя» компонента, равная заданной величине;
- возврат к $c_0$ отменяет и поворот, и сдвиг, поэтому у него две ненулевые компоненты.
```python
chain = sem.build_chain(demo)
realized = pd.DataFrame([
    {"событие": k, "вид": g.kind.value if g.kind else f"возврат к c{g.returns_to}",
     "|A|": len(g.affected), "ψ, рад": round(g.rotation, 4), "δ": round(g.shift, 4),
     **{f"m_{name}": round(value, 6) for name, value in g.realized.items()}}
    for k, g in enumerate(chain.geometry)
])
display(realized)
assert abs(chain.geometry[0].realized["real"] - 0.2) < 1e-9
assert abs(chain.geometry[1].realized["virtual"] - 0.4) < 1e-9
assert chain.states[3].concept_id == chain.states[0].concept_id
```
<!-- cell -->
#### 4.1.10. Точность калибровки на случайных сценариях

Проверим калибровку массово на 2000 случайных событиях:
- вид, доля класса, доля затронутых признаков и величина берутся в пределах допустимой области;
- в половине случаев событию предшествует prior-дрейф, так что калибровка проверяется и при $\pi_k \ne \pi_0$.

Затем сравним фактическую величину из цепочки концептов с заданной.
```python
calib_rng = np.random.default_rng(GLOBAL_SEED)
errors = []
for i in range(2000):
    d_i = int(calib_rng.integers(4, 40))
    kind = str(calib_rng.choice(["real", "virtual", "prior"]))
    pi0 = float(calib_rng.uniform(0.05, 0.5))
    alpha = float(calib_rng.uniform(2.5 / d_i, 1.0))
    events = []
    prior = pi0
    if calib_rng.random() < 0.5:  # предшествующий prior-дрейф: π_k ≠ π₀
        shift = float(calib_rng.uniform(0.05, 0.9 - pi0))
        events.append({"position": 5_000, "kind": "prior", "magnitude": shift})
        prior = pi0 + shift
    if kind == "real":
        k_aff = sem.affected_count(alpha, d_i)
        upper = sem.max_real_severity(k_aff / d_i, pi0, prior)
    elif kind == "virtual":
        upper = sem.MAX_VIRTUAL_MAGNITUDE
    else:
        upper = max(1 - 0.01 - prior, prior - 0.01)
    m = float(calib_rng.uniform(0.01, 1.0) * upper)
    events.append({"position": 12_000, "kind": kind, "magnitude": m, "affected_share": alpha})
    spec_i = ScenarioSpec(
        stream=StreamSpec(n_features=d_i, minority_share=pi0, concept_seed=i),
        events=tuple(events),
    )
    g = sem.build_chain(spec_i).geometry[-1]
    errors.append({"вид": kind, "после prior": len(events) == 2,
                   "ошибка": abs(g.realized[kind] - m),
                   "посторонние компоненты": sum(v for key, v in g.realized.items() if key != kind)})
calib = pd.DataFrame(errors)
summary = calib.groupby(["вид", "после prior"]).agg(
    событий=("ошибка", "size"), max_ошибка=("ошибка", "max"),
    max_посторонние=("посторонние компоненты", "max"))
display(summary)
assert calib["ошибка"].max() < 1e-8 and calib["посторонние компоненты"].max() < 1e-8
```
<!-- cell -->
Заданная величина воспроизводится с ошибкой порядка машинной точности (~$10^{-14}$; порог проверки $10^{-8}$), а компоненты других групп параметров равны нулю. Значит, каждое событие меняет только свою группу параметров, и его величина равна TV совместных распределений (утверждение 1).

Это свойство самой модели: тесты сверяют его с методом Монте-Карло, в том числе для реального дрейфа после prior-дрейфа. На уровне порождённых данных его проверит эксперимент Э1 (этап 2).
<!-- cell -->
#### 4.1.11. Демонстрация исправления недопустимого сценария

Сценарий нарушает три ограничения:
- C4 — первое событие внутри разогрева;
- C6 — события перекрываются;
- C10 — величина реального дрейфа недостижима при доле затронутых признаков $\alpha = 0{,}2$.

`repair` переносит события и заменяет величину на наибольшую достижимую.
```python
bad = ScenarioSpec(
    stream=StreamSpec(n_samples=15_000, n_features=10),
    events=(
        {"position": 500, "kind": "real", "magnitude": 0.5, "affected_share": 0.2},
        {"position": 2_000, "form": "gradual", "width": 800, "kind": "prior", "magnitude": 0.3},
    ),
)
print("До исправления:\n" + str(check(bad)))
fixed = repair(bad)
print("\nДействия:", *fixed.actions, sep="\n  ")
print("\nПосле исправления:", check(fixed.spec))
display(pd.DataFrame([{"k": k, "onset": e.onset, "end": e.end, "вид": e.kind.value,
                       "m": e.magnitude} for k, e in enumerate(fixed.spec.events)]))
```
<!-- cell -->
#### 4.1.12. Каталог ограничений и пространство параметров из кода

Таблицы п. 2.4–2.5 вставляются в текст из кода при сборке ноутбука, поэтому расхождение исключено. Ячейка ниже выводит их напрямую и проверяет состав каталога ограничений.
```python
from scendrift.scenario.report import CONSTRAINTS
from scendrift.scenario.space import DEFAULT_SPACE

constraints_table = pd.DataFrame(
    [(code, level, text) for code, (level, text) in CONSTRAINTS.items()],
    columns=["код", "уровень", "ограничение"],
)
display(constraints_table)
assert list(CONSTRAINTS) == [f"C{i}" for i in range(1, 15)] + ["W1", "W2", "W3"]
space_table = pd.DataFrame(DEFAULT_SPACE.table())
display(space_table[["параметр", "обозначение", "часть", "домен", "по умолчанию", "активен, если"]])
print("Варьируемые параметры:", DEFAULT_SPACE.free_names())
```

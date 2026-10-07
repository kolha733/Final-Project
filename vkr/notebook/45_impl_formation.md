### 4.6. Формирование, параметризация и версионирование наборов сценариев [К]

Раздел реализует метод из п. 2.7 — центральный результат работы. Набор сценариев строится автоматически и воспроизводимо:

    конфигурация → пространство Ξ → план U → точки ξ → сценарии S(ξ) → учёт ограничений → набор + манифест.

Модули пакета `scendrift.formation`:

| Модуль | Назначение | Пункт |
|---|---|---|
| `samplers` | планы эксперимента и метрики равномерности | 4.6.1 |
| `templates` | шаблоны и наследование `extends` | 4.6.2 |
| `mapping` | точка → сценарий; учёт ограничений reject / repair / adapt | 4.6.3 |
| `compose` | композиция сценария из блоков | 4.6.4 |
| `difficulty` | шкала сложности λ и каталог easy / medium / hard | 4.6.5 |
| `suite` | наборы, манифест, проверка целостности, версии | 4.6.6 |

Заимствованы только генераторы LHS и последовательностей Соболя и расчёт расхождения (`scipy.stats.qmc` [@scipy2020]). Обратное отображение доменов `to_unit` добавлено в модуль `space` (п. 4.1.8).
<!-- cell -->
#### 4.6.1. Планы эксперимента

Все планы получают одно пространство и один бюджет $N$ и возвращают точки единичного куба. Сетка при бюджете $N$ берёт наибольшее число уровней, при котором узлов не больше $N$.
```writefile scendrift/formation/samplers.py
```
<!-- cell -->
Сравним планы из 128 точек в пространстве по умолчанию (11 варьируемых параметров). На рисунке — проекция на две оси: величину $m$ и долю класса $\pi_0$.
```python
from scendrift.formation.samplers import design_metrics, grid_levels, make_sampler

N_DEMO = 128
PLAN_NAMES = {"grid": "сетка", "random": "случайная выборка", "lhs": "латинский гиперкуб",
              "sobol": "Соболь"}
print("Уровни сетки при бюджете 128:", grid_levels(DEFAULT_SPACE, N_DEMO))

free = DEFAULT_SPACE.free_names()
ix, iy = free.index("magnitude"), free.index("minority_share")
fig, axes = plt.subplots(1, 4, figsize=(12, 3.9), sharex=True, sharey=True)
for ax, method in zip(axes, PLAN_NAMES):
    U_demo = make_sampler(method).design(DEFAULT_SPACE, N_DEMO, GLOBAL_SEED)
    ax.scatter(U_demo[:, ix], U_demo[:, iy], s=14, color=CATEGORICAL[0], alpha=0.8)
    ax.set(title=f"{PLAN_NAMES[method]}\n{len(U_demo)} точек", xlabel="u(m)", xlim=(0, 1),
           ylim=(0, 1))
    ax.set_aspect("equal")
axes[0].set_ylabel("u(π₀)")
axes[0].annotate("все узлы сетки\nв одной точке", xy=(0.5, 0.5), xytext=(0.08, 0.8), fontsize=9,
                 color=TEXT_SECONDARY, arrowprops={"arrowstyle": "->", "color": TEXT_SECONDARY})
decimal_comma(*axes)
fig.suptitle("Рис. 4.1. Проекция плана из 128 точек на две из 11 осей", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save_figure(fig, "docs/figures/fig_4_1_plans.png")
plt.show()

plan_rows = []
for method in PLAN_NAMES:
    for s in range(5):
        U_demo = make_sampler(method).design(DEFAULT_SPACE, N_DEMO, GLOBAL_SEED + s)
        plan_rows.append({"план": PLAN_NAMES[method], "точек": len(U_demo),
                          **design_metrics(U_demo)})
plan_table = (pd.DataFrame(plan_rows).groupby("план", sort=False).mean()
              .rename(columns={"cd": "CD-расхождение", "strata": "доля слоёв"}))
plan_table.to_csv("results/formation_plans.csv")
display(plan_table.round(4))
```
<!-- cell -->
Сетка из 18 узлов перебирает только категориальные параметры: все числовые параметры стоят в середине доменов. Это иллюстрирует оценку из п. 2.7. Среди остальных планов у последовательности Соболя наименьшее расхождение, а LHS и Соболь занимают все слои каждой оси. Средние по пяти seed.
<!-- cell -->
#### 4.6.2. Шаблоны и наследование

Шаблон — неполное описание сценария. Потомок ссылается на родителя полем `extends` (встроенный каталог `catalog/<имя>` или файл) и уточняет его. Правила слияния приведены в п. 2.7.
```writefile scendrift/formation/templates.py
```
<!-- cell -->
Демонстрация правил слияния: `null` удаляет ключ, словари сливаются, наследование по файлам работает в несколько уровней, цикл обнаруживается.
```python
import tempfile

import yaml

from scendrift.formation.templates import TemplateError, deep_merge, resolve

parent = {"stream": {"n_samples": 20_000, "n_features": 10, "minority_share": 0.3},
          "drifts": {"schedule": {"count": 3}, "defaults": {"kind": "real", "magnitude": 0.2}}}
child = {"stream": {"minority_share": None, "label_noise": 0.1},
         "drifts": {"defaults": {"magnitude": 0.1}}}
print(yaml.safe_dump(deep_merge(parent, child), allow_unicode=True, sort_keys=False))

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "imbalanced.yaml").write_text(yaml.safe_dump(parent), encoding="utf-8")
    (tmp / "noisy.yaml").write_text("extends: imbalanced\nstream: {label_noise: 0.1}\n",
                                    encoding="utf-8")
    (tmp / "scenario.yaml").write_text("extends: noisy\ndrifts: {schedule: {count: 2}}\n",
                                       encoding="utf-8")
    spec_tpl = sio.load(tmp / "scenario.yaml")
    print("три уровня наследования:", spec_tpl.stream.minority_share, spec_tpl.stream.label_noise,
          spec_tpl.n_events, check(spec_tpl).ok)
    (tmp / "a.yaml").write_text("extends: b\n", encoding="utf-8")
    (tmp / "b.yaml").write_text("extends: a\n", encoding="utf-8")
    try:
        resolve({"extends": "a"}, base_dir=tmp)
    except TemplateError as exc:
        print("цикл обнаружен:", str(exc).split(": ", 1)[1])
```
<!-- cell -->
#### 4.6.3. Параметризация и учёт ограничений

Точка пространства записывается в поля шаблона, после чего компактная форма разворачивается в каноническую. Если точка недопустима, применяется один из трёх способов учёта ограничений (п. 2.7):
- `reject` — отклонение;
- `repair` — проекция на допустимую область;
- `adapt` — условные домены.

Сценарий $i$ получает структурный seed и seed реализации, выведенные из пары (seed набора, $i$).
```writefile scendrift/formation/mapping.py
```
<!-- cell -->
Возьмём недопустимую точку: реальный дрейф величины 0,45 при $\alpha = 0{,}2$ и $\pi_0 = 0{,}1$. Поворот внутри двух признаков из десяти даёт не больше $m_{\max} \approx 0{,}12$ (C10). Посмотрим, что делает каждый способ.
```python
from scendrift.formation.mapping import POLICIES, point_to_spec, realize, spec_to_point, summarize

bad_point = {**DEFAULT_SPACE.defaults(), "kind": "real", "magnitude": 0.45, "affected_share": 0.2,
             "minority_share": 0.1, "n_drifts": 2, "form": "gradual", "width": 1_500}
print("исходная точка:", check(point_to_spec(bad_point, concept_seed=0, seed=0)).codes())
u_bad = np.array([DEFAULT_SPACE.to_unit(bad_point)])
policy_rows = []
for policy in POLICIES:
    (cand,) = realize(DEFAULT_SPACE, u_bad, policy=policy, seed=0)
    got = spec_to_point(cand.spec, DEFAULT_SPACE) if cand.spec else {}
    policy_rows.append({"способ": policy, "статус": cand.status, "m": got.get("magnitude"),
                        "α": got.get("affected_share"),
                        "журнал": "; ".join(cand.actions) or "—"})
display(pd.DataFrame(policy_rows))
```
<!-- cell -->
Проекция ставит величину ровно на границу $m_{\max}$, то есть точка попадает в атом на границе $\partial F$. Условные домены отображают ту же координату плана в суженный домен $[0{,}02;\ m_{\max}]$: точка с $u(m)$ около 0,9 получает значение около 90 % допустимого диапазона, а не его край.

Сводка по плану Соболя из 128 точек:
```python
U_sobol = make_sampler("sobol").design(DEFAULT_SPACE, 128, GLOBAL_SEED)
summary_rows = []
for policy in POLICIES:
    s_policy = summarize(realize(DEFAULT_SPACE, U_sobol, policy=policy, seed=GLOBAL_SEED))
    summary_rows.append({"способ": policy, **{k: s_policy[k] for k in (
        "accepted", "valid", "repaired", "adapted", "rejected")},
        "нарушения исходных точек": s_policy["initial_codes"]})
display(pd.DataFrame(summary_rows))
```
<!-- cell -->
Нарушение C7 в исходных точках каскадное: если первое событие недопустимо (например, по C9), концепт не меняется, и последующий возврат «ничего не меняет». После исправления первого события C7 исчезает.

При способе `adapt` статус «adapted» получают и некоторые изначально допустимые точки. Это не ошибка: условный домен — другое отображение $\Psi$, и он применяется ко всем точкам одинаково, а не только к нарушителям. Например, при $d = 4$ доля признаков отображается в $[0{,}5;\ 1]$, а не в $[0{,}2;\ 1]$.
<!-- cell -->
#### 4.6.4. Композиция сценария из блоков

Сценарий описывается как временная шкала из блоков `Stable(L)`, `Drift(...)`, `Recur(to=k)`. Позиции событий и длина потока вычисляются автоматически. Тот же сценарий можно записать декларативно, ключом `blocks` в YAML.
```writefile scendrift/formation/compose.py
```
<!-- cell -->
```python
from scendrift.formation.compose import Drift, Recur, Stable, block_dicts, compose
from scendrift.generators import timeline
from scendrift.scenario.validation import concept_ids

blocks = [Stable(4_000), Drift(kind="real", magnitude=0.3, form="gradual", width=1_500),
          Stable(4_000), Drift(kind="virtual", magnitude=0.3),
          Stable(4_000), Recur(to=0, form="gradual", width=1_000), Stable(4_000)]
composed = compose(blocks, stream={"n_features": 10}, name="composed-demo")
print(f"n = {composed.stream.n_samples}, события:",
      [(e.onset, e.end, e.kind.value if e.kind else f"возврат к {e.returns_to}")
       for e in composed.events], "|", check(composed), "|", signature(composed))
assert sio.from_dict({"blocks": block_dicts(blocks), "stream": {"n_features": 10}}).events == \
    composed.events  # декларативная форма даёт тот же сценарий

tl = timeline(composed)
ids = np.array(concept_ids(composed), dtype=float)
level = ids[tl.src] * (1 - tl.p) + ids[tl.dst] * tl.p
fig, ax = plt.subplots(figsize=(11, 2.8))
for k, ev in enumerate(composed.events):
    window = (ev.onset, ev.end + composed.evaluation.acceptance_window)
    ax.axvspan(*window, color=CATEGORICAL[1], alpha=0.12, linewidth=0,
               label="окно допуска Λₖ" if k == 0 else None)
ax.axvspan(0, composed.evaluation.warmup, color=TEXT_SECONDARY, alpha=0.12, linewidth=0,
           label="разогрев W")
ax.plot(np.arange(composed.stream.n_samples), level, color=CATEGORICAL[0], linewidth=2,
        label="действующий концепт")
ax.set(xlabel="номер объекта t", ylabel="номер концепта", yticks=[0, 1, 2],
       title="Stable · Drift(real, gradual) · Stable · Drift(virtual) · Stable · Recur(0) · Stable")
ax.legend(loc="upper right", fontsize=9)
decimal_comma(ax)
fig.suptitle("Рис. 4.2. Сценарий, составленный из блоков", x=0.01, ha="left", fontsize=12,
             fontweight="bold")
fig.tight_layout()
save_figure(fig, "docs/figures/fig_4_2_composition.png")
plt.show()
```
<!-- cell -->
#### 4.6.5. Шкала сложности λ и каталог профилей

Факторы шкалы по умолчанию: с ростом $\lambda$ уменьшаются величина $m$ и доля затронутых признаков $\alpha$, растут ширина перехода $\ell$, шум $\eta$ и дисбаланс (уменьшается $\pi_0$). Профили каталога easy, medium и hard — точки шкалы при $\lambda = 0;\ 0{,}5;\ 1$.

В базовом шаблоне $d = 20$. При $d = 10$ и $\alpha = 0{,}2$ затрагиваются только два признака. После первых поворотов случайное подмножество из двух признаков может нести малую массу нормали, и у второго и третьего события величина $m = 0{,}05$ становится недостижимой (C10). При $d = 20$ затрагиваются четыре признака, и подмножество почти всегда содержит «нетронутый» признак с массой $1/d = 0{,}05$, которой достаточно. Ячейка ниже проверяет допустимость шкалы на сетке $\lambda$ и многих структурных seed — для $d = 20$ и для сравнения для $d = 10$.
```writefile scendrift/formation/difficulty.py
```
<!-- cell -->
```python
from scendrift.formation.difficulty import DIFFICULTY, PROFILES, DifficultyScale
from scendrift.formation.templates import catalog_names

display(pd.DataFrame(DIFFICULTY.describe()))
display(pd.DataFrame(DIFFICULTY.table([0, 0.25, 0.5, 0.75, 1])).round(4))

# Допустимость на всей шкале: сетка λ × структурные seed.
N_CS = {"FAST": 200, "FULL": 2000}[MODE]
lam_grid = np.linspace(0, 1, 21)
scale_failures = [(float(lam), cs) for lam in lam_grid for cs in range(N_CS)
                  if not check(DIFFICULTY.scenario(float(lam), concept_seed=cs)).ok]
print(f"d = 20: проверено {len(lam_grid) * N_CS} сценариев шкалы, недопустимых: "
      f"{len(scale_failures)}")
scale_d10 = DifficultyScale(deep_merge(DIFFICULTY.base, {"stream": {"n_features": 10}}),
                            DIFFICULTY.factors)
fail_d10 = [(float(lam), cs) for lam in lam_grid for cs in range(N_CS)
            if not check(scale_d10.scenario(float(lam), concept_seed=cs)).ok]
print(f"d = 10 (для сравнения): недопустимых {len(fail_d10)}, при λ ∈ "
      f"{sorted({lam for lam, _ in fail_d10})}")

print("Каталог:", catalog_names())
medium = sio.from_dict({"extends": "catalog/medium", "drifts": {"defaults": {"magnitude": 0.3}}})
print("catalog/medium с переопределением m = 0,3:", signature(medium), "|", check(medium))
```
<!-- cell -->
#### 4.6.6. Наборы сценариев: манифест, целостность, версии

Набор формируется по декларативной конфигурации. Манифест хранит версию схемы, версию пакета, идентификатор пространства параметров, конфигурацию, сводку формирования и для каждого сценария — идентификатор, файл и класс таксономии. При загрузке идентификатор каждого сценария пересчитывается по файлу. Поэтому изменение файла вручную обнаруживается.
```writefile scendrift/formation/suite.py
```
```writefile scendrift/formation/__init__.py
```
<!-- cell -->
Конфигурации наборов для основного бенчмарка Э2:
```writefile configs/suites/e2_fast.yaml
```
```writefile configs/suites/e2_full.yaml
```
<!-- cell -->
Сформируем набор Э2 текущего режима, сохраним его, загрузим с проверкой целостности и посмотрим состав.
```python
from scendrift.formation.suite import SuiteConfig, SuiteIntegrityError, build_suite, load_suite

E2_CONFIG = Path("configs/suites") / {"FAST": "e2_fast.yaml", "FULL": "e2_full.yaml"}[MODE]
e2_config = SuiteConfig.model_validate(yaml.safe_load(E2_CONFIG.read_text(encoding="utf-8")))
e2_suite = build_suite(e2_config)
manifest_path = e2_suite.save(Path("results/suites") / e2_config.name)
e2_loaded = load_suite(manifest_path.parent)
assert e2_loaded.suite_id == e2_suite.suite_id
print(f"{e2_suite.name} v{e2_suite.version}: {len(e2_suite)} сценариев, {e2_suite.suite_id}, "
      f"пространство {e2_suite.space_id}, пакет {e2_suite.framework_version}")
print("сводка формирования:", {k: e2_suite.formation[k] for k in (
    "design_points", "accepted", "valid", "adapted", "rejected", "duplicates")})
e2_table = pd.DataFrame(e2_suite.table())
display(e2_table.drop(columns=["signature"]).head(8).round(3))
print("классов таксономии (вид × форма × величина × скорость × баланс × возврат):",
      e2_table["signature"].str.split("/").map(lambda s: tuple(s[i] for i in (0, 1, 3, 4, 6))
                                               ).nunique())
```
<!-- cell -->
Проверка целостности и расширяемость набора. Изменим один файл сценария в копии набора — загрузка должна отказать. Затем сравним наборы из 64 и 128 точек для плана Соболя и для LHS при одном seed.
```python
import shutil

with tempfile.TemporaryDirectory() as tmp:
    copy_dir = Path(tmp) / "suite"
    shutil.copytree(manifest_path.parent, copy_dir)
    victim = copy_dir / "scenarios" / f"{e2_suite.scenarios[0].name}.yaml"
    spec_v = sio.load(victim)
    sio.save(spec_v.model_copy(update={"seed": spec_v.seed + 1}), victim)
    try:
        load_suite(copy_dir)
        print("изменение НЕ обнаружено")
    except SuiteIntegrityError as exc:
        print("изменение обнаружено:", str(exc).splitlines()[1])

ext_rows = []
for method in ["sobol", "lhs"]:
    cfg = {"name": f"ext-{method}", "template": {"extends": "catalog/base"},
           "design": {"method": method, "n": 64, "seed": GLOBAL_SEED}}
    small = build_suite(cfg)
    large = build_suite({**cfg, "design": {**cfg["design"], "n": 128}})
    diff = small.diff(large)
    ext_rows.append({"план": PLAN_NAMES[method], "общих сценариев": diff["common"],
                     "добавлено": len(diff["added"]), "удалено": len(diff["removed"])})
display(pd.DataFrame(ext_rows))
```
<!-- cell -->
#### 4.6.7. Сравнение способов учёта ограничений и планов

**(а) Как способ учёта ограничений меняет распределение сценариев.** Это свойство способа, а не плана. Поэтому оно измеряется методом Монте-Карло на большой случайной выборке $N_{MC}$ точек. Эталон — независимая случайная выборка того же размера, из которой оставлены только допустимые точки: она равномерна на $F$.

Для каждого способа измеряются:
- доля принятых точек;
- доля сценариев с величиной **на границе достижимости**: любое увеличение $m$ нарушает C10–C12 (`magnitude_on_boundary`);
- статистика Колмогорова–Смирнова по каждому параметру (в координатах единичного куба) против эталона, равномерного на $F$, и против исходного плана.

Центрированное расхождение итогового набора здесь не подходит: оно измеряет равномерность по всему кубу, а допустимый набор обязан лежать в $F$.
```python
from scipy.stats import ks_2samp

from scendrift.formation.mapping import magnitude_on_boundary

FORM_EXP = {"FAST": {"mc": 6000, "n": 128, "seeds": 5},
            "FULL": {"mc": 20000, "n": 256, "seeds": 20}}[MODE]


def realized_unit(cands: list) -> np.ndarray:
    """Фактические параметры принятых сценариев в координатах единичного куба."""
    return np.array([DEFAULT_SPACE.to_unit(spec_to_point(c.spec, DEFAULT_SPACE), c.unit)
                     for c in cands if c.spec is not None])


start = time.perf_counter()
U_ref = make_sampler("random").design(DEFAULT_SPACE, FORM_EXP["mc"], GLOBAL_SEED + 100)
reference = realized_unit(realize(DEFAULT_SPACE, U_ref, policy="reject", seed=GLOBAL_SEED + 100))
U_mc = make_sampler("random").design(DEFAULT_SPACE, FORM_EXP["mc"], GLOBAL_SEED + 200)
plan_unit = np.array([DEFAULT_SPACE.to_unit(DEFAULT_SPACE.from_unit(u), u) for u in U_mc])

mc_rows, ks_rows = [], []
for policy in POLICIES:
    cands = realize(DEFAULT_SPACE, U_mc, policy=policy, seed=GLOBAL_SEED + 200)
    X_pol = realized_unit(cands)
    kinds = pd.Series([spec_to_point(c.spec, DEFAULT_SPACE)["kind"]
                       for c in cands if c.spec is not None])
    mc_rows.append({"способ": policy, "доля принятых": len(X_pol) / len(U_mc),
                    "доля на границе": np.mean([magnitude_on_boundary(c.spec)
                                                for c in cands if c.spec is not None]),
                    **{f"доля {k}": float((kinds == k).mean()) for k in ("real", "virtual", "prior")}})
    for j, name in enumerate(free):
        ks_rows.append({"способ": policy, "параметр": name,
                        "КС: равномерное на F": ks_2samp(X_pol[:, j], reference[:, j]).statistic,
                        "КС: исходный план": ks_2samp(X_pol[:, j], plan_unit[:, j]).statistic})
ks_table = pd.DataFrame(ks_rows)
ks_table.to_csv("results/formation_ks.csv", index=False)
ks_noise = 1.358 * np.sqrt(2 / FORM_EXP["mc"])  # критическое значение КС при α = 0,05 (n ≈ m)
mc_table = pd.DataFrame(mc_rows).set_index("способ")
for col in ["КС: равномерное на F", "КС: исходный план"]:
    worst = ks_table.loc[ks_table.groupby("способ", sort=False)[col].idxmax()]
    mc_table[f"max {col}"] = worst.set_index("способ")[col]
    mc_table[f"где ({col.split(': ')[1]})"] = worst.set_index("способ")["параметр"]
print(f"N_MC = {FORM_EXP['mc']}, эталон: {len(reference)} точек, "
      f"порог шума КС ≈ {ks_noise:.3f}, время {time.perf_counter() - start:.1f} с")
mc_table.assign(**{"порог КС": ks_noise, "N_MC": FORM_EXP["mc"]}).to_csv(
    "results/formation_policies.csv")
display(mc_table.round(3))
```
<!-- cell -->
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
y_pos = np.arange(len(free))
for ax, col in zip(axes, ["КС: равномерное на F", "КС: исходный план"]):
    for k, (color, policy) in enumerate(zip(CATEGORICAL, POLICIES)):
        part = ks_table[ks_table["способ"] == policy].set_index("параметр").reindex(free)
        ax.barh(y_pos + (k - 1) * 0.27, part[col], height=0.27, color=color, label=policy)
    ax.axvline(ks_noise, color=TEXT_SECONDARY, linestyle="--", linewidth=1,
               label="порог шума (α = 0,05)")
    ax.set(xlabel="статистика Колмогорова–Смирнова", title=f"против: {col.split(': ')[1]}")
decimal_comma(*axes)
axes[0].set_yticks(y_pos, [f"{DEFAULT_SPACE[name].symbol} — {name}" for name in free])
axes[0].invert_yaxis()
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9.5, frameon=False)
fig.suptitle("Рис. 4.3. Как способ учёта ограничений меняет распределение параметров",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0.07, 1, 1))
save_figure(fig, "docs/figures/fig_4_3_policies.png")
plt.show()
```
<!-- cell -->
**(б) Планы при равном бюджете.** Для каждого сочетания «план × способ» формируется набор из $N$ точек; для случайных планов — по нескольким seed. Измеряются доля принятых точек, доля сценариев на границе и число различных классов таксономии по осям «вид × форма × величина × скорость × баланс × возврат». Сетка детерминирована и даёт 18 узлов.
```python
from scendrift.scenario.taxonomy import classify


def taxonomy_cell(spec: ScenarioSpec) -> tuple:
    c = classify(spec)
    return (c.kinds, c.forms, c.severity, c.speed, c.balance, c.recurring)


def evaluate_plan(method: str, U: np.ndarray, seed: int) -> list[dict]:
    rows_eval = []
    for policy in POLICIES:
        acc = [c for c in realize(DEFAULT_SPACE, U, policy=policy, seed=seed) if c.spec]
        rows_eval.append({
            "план": PLAN_NAMES[method], "способ": policy, "seed": seed,
            "доля принятых": len(acc) / len(U),
            "доля на границе": np.mean([magnitude_on_boundary(c.spec) for c in acc]),
            "классов": len({taxonomy_cell(c.spec) for c in acc}),
        })
    return rows_eval


start = time.perf_counter()
cmp_rows = evaluate_plan("grid", make_sampler("grid").design(DEFAULT_SPACE, FORM_EXP["n"], 0), 0)
for method in ["random", "lhs", "sobol"]:
    for s in range(FORM_EXP["seeds"]):
        seed_s = GLOBAL_SEED + s
        cmp_rows += evaluate_plan(method, make_sampler(method).design(
            DEFAULT_SPACE, FORM_EXP["n"], seed_s), seed_s)
comparison = pd.DataFrame(cmp_rows)
comparison.to_csv("results/formation_comparison.csv", index=False)
print(f"N = {FORM_EXP['n']}, seed: {FORM_EXP['seeds']}, время {time.perf_counter() - start:.1f} с")
comparison_summary = comparison.groupby(["план", "способ"], sort=False)[
    ["доля принятых", "доля на границе", "классов"]].agg(["mean", "std"])
display(comparison_summary.round(3))
```
<!-- cell -->
#### 4.6.8. Проверка утверждений и выводы по п. 4.6

Как и в Э1, каждое утверждение выводов проверяется по результатам ячеек выше.
```python
by_plan = comparison.groupby(["план", "способ"])["классов"].mean()
claims_46 = [
    ("Соболь: наименьшее CD-расхождение плана", plan_table["CD-расхождение"].idxmin() == "Соболь"),
    ("LHS и Соболь занимают все слои каждой оси",
     bool((plan_table.loc[["латинский гиперкуб", "Соболь"], "доля слоёв"] == 1.0).all())),
    ("сетка при бюджете 128 даёт 18 узлов", int(plan_table.loc["сетка", "точек"]) == 18),
    ("шкала λ допустима при d = 20 и недопустима местами при d = 10",
     not scale_failures and bool(fail_d10)),
    ("reject: распределение не отличается от равномерного на F (КС ≤ порога)",
     bool(mc_table.loc["reject", "max КС: равномерное на F"] <= ks_noise)),
    ("reject: распределение отличается от плана (КС > порога)",
     bool(mc_table.loc["reject", "max КС: исходный план"] > ks_noise)),
    ("repair: часть сценариев на границе достижимости",
     bool(mc_table.loc["repair", "доля на границе"] > 0.01)),
    ("adapt: нет сценариев на границе, принято ≥ 99 %",
     mc_table.loc["adapt", "доля на границе"] == 0 and mc_table.loc["adapt", "доля принятых"] >= 0.99),
    ("adapt: классов таксономии больше, чем у reject, для каждого плана",
     all(by_plan[(p, "adapt")] > by_plan[(p, "reject")] for p in ["случайная выборка",
                                                                  "латинский гиперкуб", "Соболь"])),
    ("adapt: Соболь покрывает больше классов, чем случайная выборка",
     by_plan[("Соболь", "adapt")] > by_plan[("случайная выборка", "adapt")]),
    ("расширение Соболя 64 → 128 сохраняет все 64 сценария; у LHS — ни одного",
     ext_rows[0]["общих сценариев"] == 64 and ext_rows[1]["общих сценариев"] == 0),
]
display(pd.DataFrame(claims_46, columns=["утверждение", "выполнено"]))
```
<!-- cell -->
**Выводы по п. 4.6.** Числа — из прогона FAST.

1. **План эксперимента.** При бюджете 128 у последовательности Соболя наименьшее центрированное расхождение (0,021 против 0,040 у LHS и 0,068 у случайной выборки) и наибольшее минимальное расстояние между точками. LHS и Соболь стратифицируют каждую ось. Сетка в 11-мерном пространстве вырождается в 18 узлов, перебирающих только категориальные параметры. Поэтому основной план — последовательность Соболя.
2. **Учёт ограничений: универсально лучшего способа нет.** В пространстве по умолчанию недопустима примерно пятая часть точек: отклонение принимает 79 % точек Монте-Карло.
   - **Отклонение** сохраняет равномерность на допустимой области: отличие от эталона на уровне шума (КС 0,022 при пороге 0,025). Но оно теряет пятую часть бюджета и смещает баланс плана: чаще всего отклоняется реальный дрейф (C10), его доля падает с 32 до 24 %, а доля prior растёт с 34 до 42 %.
   - **Проекция** сохраняет весь бюджет, но около 9 % сценариев оказываются на границе достижимости: их величина — максимально возможная при их геометрии. Это атом в распределении, то есть смещение набора к «крайним» сценариям.
   - **Условные домены** принимают 99,9 % точек, не создают атомов и сохраняют баланс плана по видам дрейфа. Цена — наибольшее отклонение по доле признаков $\alpha$ (КС 0,107 против плана): при малом $d$ условный домен перемасштабирует $\alpha$ у всех точек, а не только у недопустимых.

   По умолчанию наборы формируются способом `adapt`. Статус каждого сценария (valid / adapted) сохраняется в манифесте, поэтому из набора можно выделить подмножество исходно допустимых точек.
3. **Покрытие.** При одном бюджете условные домены и проекция покрывают больше классов таксономии, чем отклонение (у Соболя 81 против 69). Соболь покрывает больше классов, чем LHS и случайная выборка, и с меньшим разбросом между seed.
4. **Расширяемость.** Набор Соболя из 64 сценариев — начальная часть набора из 128: при увеличении бюджета все уже прогнанные сценарии и их результаты сохраняются. У LHS общих сценариев нет. Поэтому набор Э2 в режиме FULL (256) включает набор FAST (64).
5. **Шкала сложности** допустима на всей сетке $\lambda$ при $d = 20$. При $d = 10$ при больших $\lambda$ появляются недопустимые сценарии, поэтому в базовом шаблоне $d = 20$. Что шкала действительно отражает сложность обнаружения, проверяется в Э4.
6. **Целостность.** Изменение файла сценария в сохранённом наборе обнаруживается при загрузке: идентификатор, пересчитанный по файлу, не совпадает с манифестом.

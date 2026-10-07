### 4.7. Адаптеры детекторов и базовые линии [Т]

Детекторы river 0.26.1 [@river2021] приводятся к интерфейсу `DriftDetector` (п. 3.4): `update(e) → bool`, `reset()`, `clone()`. Детектор получает индикатор ошибки базовой модели $e_t$. Сравниваются:
- девять детекторов river: ADWIN [@adwin2007], KSWIN [@kswin2020], Page–Hinkley [@page1954] в двустороннем и одностороннем вариантах, DDM [@ddm2004], EDDM [@eddm2006], HDDM_A и HDDM_W [@hddm2015], FHDDM [@fhddm2016];
- три базовые линии (собственная реализация): NoDrift, Periodic и Oracle.

Детекторы заимствованы; собственная часть — единый интерфейс, реестр с параметрами в имени и базовые линии.
```writefile scendrift/detectors/registry.py
```
```writefile scendrift/detectors/__init__.py
```
<!-- cell -->
```python
from scendrift.detectors import DETECTORS, detector_table, make_detector

display(pd.DataFrame(detector_table()))
```
<!-- cell -->
**Проверка направления реакции.** Направление в реестре проверяется на синтетических потоках ошибок. Первые 3000 объектов идут с частотой ошибок $a$, следующие 1000 — с частотой $b$. Считается доля потоков, в которых детектор сработал среди последних 1000 объектов.

Срабатывание после изменения может оказаться и фоновой ложной тревогой. Поэтому для каждого случая есть контрольный поток без изменения: частота $a$ на всех 4000 объектах. Детектор реагирует на изменение, если доля после изменения заметно больше контрольной. Seed — 40 (у медленного KSWIN — 10).

Отдельно проверяется FHDDM при подаче ошибок $e_t$, как сказано в описании входа в river. Реализация river сигналит, когда среднее входа падает ниже своего максимума: это формулировка исходной статьи, где в окне лежат индикаторы верного ответа. Поэтому при подаче $e_t$ FHDDM должен реагировать на **снижение** ошибки, а при подаче $1 - e_t$ — на рост.
```python
from river.drift import binary as river_binary
from scendrift.detectors import RiverDetector
from scendrift.detectors.registry import DetectorInfo


def share_after(det_factory, a: float, b: float, seeds: int) -> float:
    """Доля потоков a → b, в которых детектор сработал на объектах 3000–3999."""
    hits = 0
    for s in range(seeds):
        rng = np.random.default_rng(s)
        stream = np.r_[rng.random(3_000) < a, rng.random(1_000) < b].astype(float)
        det = det_factory()
        hits += any(det.update(v) and t >= 3_000 for t, v in enumerate(stream.tolist()))
    return hits / seeds


def direction_row(label: str, registry: str, factory, seeds: int) -> dict:
    return {"детектор": label, "в реестре": registry,
            "рост 0,1 → 0,35": share_after(factory, 0.1, 0.35, seeds),
            "контроль 0,1": share_after(factory, 0.1, 0.1, seeds),
            "падение 0,35 → 0,1": share_after(factory, 0.35, 0.1, seeds),
            "контроль 0,35": share_after(factory, 0.35, 0.35, seeds)}


fhddm_raw = DetectorInfo("FHDDM (вход e)", river_binary.FHDDM, binary_input=True)
direction = pd.DataFrame(
    [direction_row(name, "оба" if DETECTORS[name].two_sided else "рост ошибки",
                   lambda name=name: make_detector(name), 10 if name == "KSWIN" else 40)
     for name in DETECTORS]
    + [direction_row("FHDDM, вход e (как в описании river)", "—",
                     lambda: RiverDetector(fhddm_raw), 40)])
direction.to_csv("results/detector_direction.csv", index=False)
display(direction)
```
<!-- cell -->
Результаты проверки:
- **Односторонние детекторы** после падения ошибки срабатывают не чаще, чем на контрольном потоке без изменения. **ADWIN и двусторонний Page–Hinkley** срабатывают после падения во всех 40 потоках.
- **FHDDM с входом $e_t$** срабатывает после падения ошибки в 97,5 % потоков, а после роста — лишь в 7,5 %. Поэтому в реестре он получает $1 - e_t$.
- **Контрольные столбцы показывают фоновые ложные тревоги.** У EDDM при частоте ошибок 0,1 они возникают в 72,5 % потоков на отрезке из 1000 объектов, у HDDM_W при частоте 0,35 — в 45 %.
- **KSWIN** с параметрами по умолчанию обнаружил рост ошибки лишь в 40 % потоков. Он сравнивает 30 последних значений с 30 случайными из окна в 100 и требует, чтобы статистика КС была больше 0,1. Это свойство детектора, а не ошибка адаптера.
<!-- cell -->
### 4.8. Базовая модель и раннер [Т]

Базовая модель $h$ повторяет математику `river.naive_bayes.GaussianNB`, но вычисляет ошибки векторно, блоками, по кумулятивным достаточным статистикам. Цикл Python остаётся только у детектора.

Раннер выполняет протокол $\Omega$ (п. 2.6):
- модель работает по схеме test-then-train с первого объекта;
- детектор получает $e_t$ начиная с $t = W$;
- после срабатывания детектор начинается заново, а при $a = \text{reset}$ сбрасывается и модель;
- повторы идут по seed $s, s+1, \dots$;
- результаты кэшируются по идентификатору сценария с seed и имени детектора;
- реализации сценариев обрабатываются параллельно (joblib).
```writefile scendrift/evaluation/models.py
```
```writefile scendrift/evaluation/runner.py
```
```writefile scendrift/evaluation/__init__.py
```
<!-- cell -->
**Эквивалентность с river.** Сравним ошибки векторной модели и `GaussianNB` из river на потоках трёх групп семейств со сбросами модели в произвольные моменты. Сравнение побитовое.
```python
from river import naive_bayes

from scendrift.evaluation.models import PrequentialGaussianNB


def river_nb_errors(X: np.ndarray, y: np.ndarray, resets: set[int]) -> np.ndarray:
    out, model = [], naive_bayes.GaussianNB()
    for t in range(len(y)):
        if t in resets:
            model = naive_bayes.GaussianNB()
        x = dict(enumerate(X[t].tolist()))
        out.append(model.predict_one(x) != int(y[t]))
        model.learn_one(x, int(y[t]))
    return np.array(out)


def vector_nb_errors(X: np.ndarray, y: np.ndarray, resets: set[int]) -> np.ndarray:
    nb = PrequentialGaussianNB(X, y)
    points = sorted({0, *resets, len(y)})
    return np.concatenate([nb.errors(a, b) for a, b in zip(points[:-1], points[1:])])


nb_rows = []
for family, spec_nb in demo_specs.items():
    data_nb = generate(spec_nb)
    resets = {1, 2, 3_000, 7_777, 12_345}
    start = time.perf_counter()
    ref = river_nb_errors(data_nb.X, data_nb.y, resets)
    t_river = time.perf_counter() - start
    start = time.perf_counter()
    ours = vector_nb_errors(data_nb.X, data_nb.y, resets)
    t_ours = time.perf_counter() - start
    nb_rows.append({"семейство": family, "объектов": len(ref),
                    "расхождений": int((ref != ours).sum()), "доля ошибок": ref.mean(),
                    "river, с": t_river, "векторно, с": t_ours, "ускорение": t_river / t_ours})
nb_table = pd.DataFrame(nb_rows)
display(nb_table.round(4))
```
<!-- cell -->
**Прогон одного сценария.** Все детекторы и базовые линии на одной реализации профиля `catalog/medium` (три постепенных реальных дрейфа, дисбаланс, шум 0,1). Верхняя панель — скользящая частота ошибок модели без адаптации, нижняя — срабатывания относительно ground truth.
```python
from scendrift.evaluation.runner import run_benchmark, run_scenario
from scendrift.reporting.plots import cd_diagram, detection_raster

ALL_DETECTORS = [make_detector(name) for name in DETECTORS] + [
    make_detector("NoDrift"), make_detector("Periodic", period=2_000), make_detector("Oracle")]
medium_spec = sio.from_dict({"extends": "catalog/medium", "name": "medium"})
medium_data = generate(medium_spec)
medium_runs = run_scenario(medium_spec, medium_spec.seed, ALL_DETECTORS)

fig, (ax_err, ax_det) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 3]})
rolling = pd.Series(PrequentialGaussianNB(medium_data.X, medium_data.y).errors()).rolling(500).mean()
ax_err.plot(rolling.index, rolling, color=CATEGORICAL[0], linewidth=1.5)
ax_err.set(ylabel="частота ошибок", title="модель без адаптации (окно 500)")
detection_raster(medium_data.ground_truth, medium_spec.evaluation,
                 {r.detector: r.detections for r in medium_runs}, ax=ax_det)
ax_det.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=6, fontsize=8.5, frameon=False)
decimal_comma(ax_err)
fig.suptitle("Рис. 4.4. Срабатывания детекторов на одной реализации профиля medium",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
save_figure(fig, "docs/figures/fig_4_4_detections.png")
plt.show()
```
<!-- cell -->
**Переходный процесс после сброса.** После сброса новая модель сначала ошибается часто, затем частота ошибок быстро падает. Двусторонний детектор может принять это падение за дрейф и снова сбросить модель. Сравним число ложных тревог при политике $a = \text{reset}$ и $a = \text{none}$ (модель не сбрасывается) на трёх профилях каталога.
```python
from scendrift.evaluation.results import runs_to_frame

POLICY_DETECTORS = [make_detector(n) for n in ["ADWIN", "PageHinkley", "PageHinkley-up", "HDDM_A"]]
policy_specs, policy_rows = [], []
for profile in ["easy", "medium", "hard"]:
    base_spec = sio.from_dict({"extends": f"catalog/{profile}", "name": profile})
    for policy in ["reset", "none"]:
        protocol_p = EvaluationSpec.model_validate(
            {**base_spec.evaluation.model_dump(), "on_detection": policy})
        spec_p = base_spec.model_copy(update={"evaluation": protocol_p})
        policy_specs.append(spec_p)
policy_runs = run_benchmark(policy_specs, POLICY_DETECTORS, repeats=3, n_jobs=4)
policy_frame = runs_to_frame(policy_runs, policy_specs)
policy_frame["политика"] = policy_frame["scenario_id"].map(
    {sio.scenario_id(s): s.evaluation.on_detection.value for s in policy_specs})
policy_table = policy_frame.pivot_table(index="detector", columns="политика",
                                        values=["fp_out", "precision", "recall"], aggfunc="mean")
policy_frame.groupby(["detector", "политика"])[["fp_out", "precision", "recall"]].mean().to_csv(
    "results/protocol_policy.csv")
display(policy_table.round(2))
```
<!-- cell -->
### 4.9. Метрики [Т]

Модуль реализует правило сопоставления и определения п. 2.6 без изменений, включая соглашения для вырожденных случаев. Модуль результатов собирает таблицу «прогон → метрики + параметры сценария». В ней же считаются разность accuracy с NoDrift на той же реализации потока и разрыв до Oracle.
```writefile scendrift/evaluation/metrics.py
```
```writefile scendrift/evaluation/results.py
```
<!-- cell -->
**Расчёт вручную.** Три события, $W = 500$, $\Delta = 500$, задержка от начала перехода. Значения, вычисленные вручную, сравниваются с модулем.
```python
from scendrift.evaluation.metrics import detection_metrics, match_detections
from scendrift.interfaces import DriftInterval, GroundTruth

gt_example = GroundTruth((DriftInterval(0, 1_000, 1_000, 1_000, "sudden", "real", 0.2),
                          DriftInterval(1, 3_000, 3_100, 3_200, "gradual", "real", 0.2),
                          DriftInterval(2, 6_000, 6_000, 6_000, "sudden", "real", 0.2)), 8_000)
protocol_example = EvaluationSpec(warmup=500, acceptance_window=500)
detections_example = [100, 700, 1_049, 1_200, 3_050, 4_000, 4_100]
match_example = match_detections(detections_example, gt_example, protocol_example)
print("окна Λ:", [iv.acceptance_window(500) for iv in gt_example.intervals])
print("обнаружения:", match_example.hits, "задержки:", match_example.delays)
print("избыточные:", match_example.redundant, "ложные тревоги:", match_example.false_alarms,
      "T_stable:", match_example.t_stable)

t_stable = 7_500 - (500 + 700 + 500)
by_hand = {"precision": 2 / 6, "recall": 2 / 3, "f1": 2 * (2 / 6) * (2 / 3) / (2 / 6 + 2 / 3),
           "mtd": (50 + 51) / 2, "mtfa": t_stable / 3,
           "mtr": t_stable / 3 / 50.5 * (2 / 3), "far": 1_000 * 3 / t_stable}
computed = detection_metrics(detections_example, gt_example, protocol_example)
metric_check = pd.DataFrame({"вручную": by_hand,
                             "модуль": {k: computed[k] for k in by_hand}})
display(metric_check.round(4))
assert np.allclose(metric_check["вручную"], metric_check["модуль"])
```
<!-- cell -->
Срабатывание 100 приходится на разогрев и не учитывается. Срабатывание 700 лежит вне окон и считается ложной тревогой. Срабатывание 1049 — первое в окне $[1000, 1500)$, это обнаружение с задержкой $1049 - 1000 + 1 = 50$; следующее, 1200, избыточное. Третье событие пропущено.
<!-- cell -->
### 4.10. Статистическое сравнение [Т]

Сравнение $k$ методов на $N$ сценариях выполняется по Demšar [@demsar2006]:
- средние ранги;
- критерий Фридмана [@friedman1937] с поправкой Имана–Давенпорта;
- пост-хок Неменьи [@nemenyi1963] с критической разностью $\mathrm{CD} = q_\alpha\sqrt{k(k+1)/(6N)}$;
- попарный критерий Уилкоксона [@wilcoxon1945] с поправкой Холма [@holm1979].

Значение метрики в блоке усредняется по повторам, чтобы повторы не считались независимыми наборами данных. Квантили $q_\alpha$ вычисляются по распределению стьюдентизированного размаха (`scipy.stats.studentized_range`), а не берутся из таблицы. Ниже они сверяются с таблицей Demšar.
```writefile scendrift/evaluation/stats.py
```
<!-- cell -->
```python
from scendrift.evaluation.stats import (average_ranks, block_matrix, friedman, nemenyi_cd,
                                        wilcoxon_holm)

demsar_q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
            10: 3.164}  # Demšar (2006), табл. 5(a), α = 0,05
q_check = pd.DataFrame({"таблица Demšar": demsar_q,
                        "вычислено": {k: nemenyi_cd(k, 10)[1] for k in demsar_q}})
q_check["разность"] = (q_check["вычислено"] - q_check["таблица Demšar"]).abs()
display(q_check.T.round(4))
assert q_check["разность"].max() < 1e-3
```
<!-- cell -->
### 4.11. Отчёты и пилотный прогон [Т]

Отчёты:
- сводная таблица «среднее ± ст. откл.» с десятичной запятой;
- CSV с сырыми результатами и сводка в Markdown;
- самодостаточный HTML-отчёт: таблицы и рисунки встроены в файл;
- CD-диаграмма и растр срабатываний (п. 4.8).
```writefile scendrift/reporting/plots.py
```
```writefile scendrift/reporting/report.py
```
```writefile scendrift/reporting/__init__.py
```
<!-- cell -->
**Пилотный прогон** проверяет весь конвейер от набора сценариев до отчёта. Берутся первые 16 сценариев набора Э2 (п. 4.6.6) — начальный отрезок плана Соболя — и все детекторы с базовыми линиями.

Это **не результат исследования**: 16 сценариев и малое число повторов недостаточны для выводов о детекторах. Основной бенчмарк и статистическое сравнение — эксперименты Э2 и Э5 (этап 5).
```python
from scendrift.reporting.report import export_tables, format_table, html_report, summary_table

PILOT = {"FAST": {"scenarios": 16, "repeats": 1}, "FULL": {"scenarios": 64, "repeats": 3}}[MODE]
pilot_specs = list(e2_suite.scenarios[: PILOT["scenarios"]])
start = time.perf_counter()
pilot_runs = run_benchmark(pilot_specs, ALL_DETECTORS, repeats=PILOT["repeats"], n_jobs=4,
                           cache_dir="results/cache")
pilot = runs_to_frame(pilot_runs, pilot_specs)
pilot_time = time.perf_counter() - start
print(f"прогонов: {len(pilot_runs)}, сценариев: {len(pilot_specs)}, повторов: {PILOT['repeats']}, "
      f"время {pilot_time:.0f} с (с учётом кэша)")

PILOT_METRICS = ["f1", "precision", "recall", "mtd", "far", "delta_accuracy", "runtime_s"]
detector_order = [d.name for d in ALL_DETECTORS]
pilot_summary = summary_table(pilot, PILOT_METRICS, order=detector_order)
display(format_table(pilot_summary))
```
<!-- cell -->
```python
river_names = list(DETECTORS)
f1_matrix, dropped = block_matrix(pilot[pilot["detector"].isin(river_names)], "f1",
                                  methods=river_names)
pilot_ranks = average_ranks(f1_matrix)
pilot_friedman = friedman(f1_matrix)
pilot_cd, _ = nemenyi_cd(f1_matrix.shape[1], f1_matrix.shape[0])
print(f"блоков: {f1_matrix.shape[0]} (исключено {dropped}), методов: {f1_matrix.shape[1]}; "
      f"Фридман χ² = {pilot_friedman.chi2:.2f}, p = {pilot_friedman.p_chi2:.3g}; "
      f"Иман–Давенпорт F = {pilot_friedman.f_stat:.2f}, p = {pilot_friedman.p_f:.3g}; "
      f"CD = {pilot_cd:.2f}")
fig, ax = plt.subplots(figsize=(9, 3.4))
cd_diagram(pilot_ranks, pilot_cd, ax=ax)
fig.suptitle(f"Рис. 4.5. CD-диаграмма по F1 (пилотный прогон, N = {f1_matrix.shape[0]})",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
save_figure(fig, "docs/figures/fig_4_5_cd_pilot.png")
plt.show()
pilot_wilcoxon = wilcoxon_holm(f1_matrix)
print("пар с различием после поправки Холма:",
      int((pilot_wilcoxon["различие"] != "не обнаружено").sum()), "из", len(pilot_wilcoxon))

paths = export_tables(pilot, pilot_summary, "results", "pilot")
report_path = html_report(
    "Пилотный прогон: 16 сценариев набора e2_fast",
    [("Сводка по детекторам (среднее ± ст. откл.)", format_table(pilot_summary)),
     ("Средние ранги по F1", pilot_ranks.round(2).to_frame("средний ранг")),
     ("CD-диаграмма", "docs/figures/fig_4_5_cd_pilot.png"),
     ("Попарные сравнения (Уилкоксон, Холм)", pilot_wilcoxon.round(4)),
     ("Срабатывания на профиле medium", "docs/figures/fig_4_4_detections.png")],
    "results/reports/pilot.html")
print("файлы:", [str(p) for p in [*paths.values(), report_path]])
```
<!-- cell -->
#### 4.7–4.11. Проверка утверждений и выводы

```python
pilot_mean = pilot.groupby("detector").mean(numeric_only=True)
one_sided = direction["в реестре"] == "рост ошибки"
oracle_by_kind = pilot[pilot["detector"] == "Oracle"].groupby("kind")["delta_accuracy"].mean()
print("прирост accuracy у Oracle по виду первого события:", oracle_by_kind.round(4).to_dict())
policy_fp = policy_frame.groupby(["detector", "политика"])["fp_out"].mean()
claims_47 = [
    ("векторная модель совпадает с river GaussianNB побитово", int(nb_table["расхождений"].sum()) == 0),
    ("векторная модель быстрее river не менее чем в 20 раз", bool(nb_table["ускорение"].min() >= 20)),
    ("односторонние детекторы после падения ошибки срабатывают не чаще, чем без изменения",
     bool((direction.loc[one_sided, "падение 0,35 → 0,1"]
           <= direction.loc[one_sided, "контроль 0,35"] + 0.1).all())),
    ("двусторонние ADWIN и Page–Hinkley реагируют на падение ошибки",
     bool((direction.set_index("детектор").loc[["ADWIN", "PageHinkley"], "падение 0,35 → 0,1"]
           >= 0.9).all())),
    ("FHDDM с входом e реагирует на падение, а не на рост ошибки",
     bool(direction.iloc[-1]["падение 0,35 → 0,1"] > 0.9 > direction.iloc[-1]["рост 0,1 → 0,35"])),
    ("двусторонние ADWIN и Page–Hinkley дают больше ложных тревог при сбросе модели",
     all(policy_fp[(d, "reset")] > policy_fp[(d, "none")] for d in ["ADWIN", "PageHinkley"])),
    ("Periodic(2000) получает recall 1 (иллюзия прогресса)",
     bool(pilot_mean.loc["Periodic(2000)", "recall"] == 1.0)),
    ("Oracle: F1 = 1 и разрыв до Oracle равен 0", bool(pilot_mean.loc["Oracle", "f1"] == 1.0
                                                     and pilot_mean.loc["Oracle", "oracle_gap"] == 0)),
    ("Oracle повышает accuracy при реальном дрейфе и снижает при виртуальном и prior",
     bool(oracle_by_kind["real"] > 0 > max(oracle_by_kind["virtual"], oracle_by_kind["prior"]))),
    ("метрики совпадают с расчётом вручную", bool(np.allclose(metric_check["вручную"],
                                                              metric_check["модуль"]))),
    ("q_α Неменьи совпадают с таблицей Demšar (до 0,001)", bool(q_check["разность"].max() < 1e-3)),
]
display(pd.DataFrame(claims_47, columns=["утверждение", "выполнено"]))
```
<!-- cell -->
**Выводы по п. 4.7–4.11.** Числа — из прогона FAST.

1. **Базовая модель.** Векторная реализация побитово совпадает с `river.naive_bayes.GaussianNB` на трёх группах семейств: по 20 000 объектов, сбросы модели в произвольные моменты. Она быстрее в десятки раз (точные значения зависят от машины — см. таблицу в п. 4.8), поэтому время прогона определяют детекторы, а не модель.
2. **Детекторы.** Направление реакции соответствует реестру. В river 0.26.1 описание входа FHDDM противоречит реализации, поэтому FHDDM получает индикатор верного ответа. Без этой поправки он обнаруживал бы улучшение модели, а не дрейф.
3. **Протокол.** При политике reset двусторонние детекторы реагируют на переходный процесс после сброса: ложных тревог у ADWIN 3,56 против 1,00 при политике none, у Page–Hinkley — 7,00 против 1,22 (профили каталога, три повтора). У односторонних детекторов разницы нет. Это подтверждает замечание п. 2.6: в Э2 детекторы нужно сравнивать с учётом политики адаптации.
4. **Базовые линии работают как задумано.**
   - Periodic(2000) получает recall 1 при precision 0,28: «иллюзия прогресса» видна только по precision и FAR.
   - Oracle получает F1 = 1, но повышает accuracy только при реальном дрейфе (+0,057), а при виртуальном и prior снижает её (−0,018 и −0,012). Сброс выбрасывает модель, которая всё ещё верна. Поэтому разрыв до Oracle имеет смысл только для реального дрейфа.
5. **Метрики и статистика проверены.** Метрики совпадают с ручным расчётом по определениям п. 2.6, а квантили Неменьи — с таблицей Demšar (до 0,001).
6. **Пилотный прогон** (16 сценариев, 192 прогона) прошёл весь конвейер: набор → прогоны с кэшем → таблица → статистика → CSV, Markdown и HTML-отчёт.
   - KSWIN тратит в среднем 6,3 с на прогон, остальные детекторы — 0,03–0,12 с. В Э2 основное время займёт KSWIN.
   - Критерий Фридмана на пилоте отвергает равенство детекторов по F1 (p = 0,003), но ни одна пара после поправки Холма не различается. 16 сценариев мало для попарных выводов, поэтому основное сравнение — в Э5.

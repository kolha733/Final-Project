### 5.2. Э2. Основной бенчмарк на автоматически сформированном наборе [Т]

**Вопрос.** Как детекторы river работают на наборе сценариев, равномерно покрывающем пространство параметров?

**Метод.**
- Набор — `e2_fast` в режиме FAST (64 сценария) или `e2_full` в режиме FULL (256 сценариев). Он сформирован автоматически: план Соболя по 11 параметрам, условные домены (п. 4.6.6).
- Детекторы — девять детекторов river и три базовые линии (п. 4.7).
- Повторов $R$ по seed реализации: 3 в FAST, 5 в FULL.
- Основной протокол — $a = \text{reset}$: при срабатывании модель сбрасывается, как предусмотрено планом работы.
- Дополнительно тот же набор прогоняется при $a = \text{none}$: п. 4.8 показал, что сброс порождает ложные тревоги у двусторонних детекторов. В режиме FAST KSWIN в дополнительный прогон не входит ради времени (около 6 с на прогон, п. 4.11).

Блок статистического сравнения — сценарий: метрики усредняются по повторам (п. 4.10).
```python
import os

from scendrift.evaluation.analysis import rank_agreement, scenario_level, spearman_table
from scendrift.evaluation.runner import with_protocol

N_JOBS = os.cpu_count() or 1
CACHE = "results/cache"
E2 = {"FAST": {"repeats": 3}, "FULL": {"repeats": 5}}[MODE]
RIVER = list(DETECTORS)
BASELINE_NAMES = ["NoDrift", "Periodic(2000)", "Oracle"]

e2_specs = list(e2_suite.scenarios)
start = time.perf_counter()
e2_runs = run_benchmark(e2_specs, ALL_DETECTORS, repeats=E2["repeats"], n_jobs=N_JOBS,
                        cache_dir=CACHE)
e2 = runs_to_frame(e2_runs, e2_specs)
e2.to_csv("results/e2_runs.csv", index=False)
print(f"набор {e2_suite.name}: {len(e2_specs)} сценариев × {E2['repeats']} повтора × "
      f"{len(ALL_DETECTORS)} детекторов = {len(e2_runs)} прогонов; "
      f"время {time.perf_counter() - start:.0f} с (с учётом кэша), процессов {N_JOBS}")

E2_METRICS = ["f1", "precision", "recall", "mtd", "far", "mtfa_capped", "delta_accuracy",
              "runtime_s"]
e2_summary = summary_table(e2, E2_METRICS, order=detector_order)
display(format_table(e2_summary))
```
<!-- cell -->
Распределение F1 по сценариям (среднее по повторам) и F1 по виду дрейфа первого события:
```python
e2_scen = scenario_level(e2, ["f1", "recall", "precision", "far", "delta_accuracy", "mtd"],
                         keep=["magnitude", "width", "alpha", "pi0", "eta", "d", "K", "n",
                               "kind", "form", "recurring"])
order_f1 = (e2_scen[e2_scen["detector"].isin(RIVER)].groupby("detector")["f1"].median()
            .sort_values(ascending=False).index.tolist())
fig, ax = plt.subplots(figsize=(10, 4.6))
data_box = [e2_scen.loc[e2_scen["detector"] == d, "f1"].to_numpy() for d in order_f1]
bp = ax.boxplot(data_box, orientation="horizontal", widths=0.6, patch_artist=True, showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": CATEGORICAL[1],
                           "markeredgecolor": CATEGORICAL[1], "markersize": 5},
                medianprops={"color": TEXT_SECONDARY})
for box in bp["boxes"]:
    box.set(facecolor=CATEGORICAL[0], alpha=0.35, edgecolor=CATEGORICAL[0])
ax.set_yticks(range(1, len(order_f1) + 1), order_f1)
ax.invert_yaxis()
ax.set(xlabel="F1 сценария (среднее по повторам); ромб — среднее", xlim=(-0.02, 1.02))
decimal_comma(ax)
fig.suptitle(f"Рис. 5.4. Э2: F1 детекторов на наборе {e2_suite.name} ({len(e2_specs)} сценариев)",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
save_figure(fig, "docs/figures/fig_5_4_e2_f1.png")
plt.show()

e2_by_kind = e2_scen[e2_scen["detector"].isin(RIVER)].pivot_table(
    index="detector", columns="kind", values="f1", aggfunc="mean").reindex(order_f1)
print("сценариев по виду первого события:",
      e2_scen[e2_scen["detector"] == order_f1[0]]["kind"].value_counts().to_dict())
display(e2_by_kind.round(3))
```
<!-- cell -->
**Политика адаптации.** Тот же набор при $a = \text{none}$: модель не сбрасывается, срабатывания только фиксируются. Сравниваются средние F1, precision и число ложных тревог, а также согласие ранжирований по F1 (τ Кендалла).
```python
e2_none_specs = [with_protocol(s, on_detection="none") for s in e2_specs]
policy_detectors = [d for d in ALL_DETECTORS if MODE == "FULL" or d.name != "KSWIN"]
e2_none_runs = run_benchmark(e2_none_specs, policy_detectors, repeats=E2["repeats"],
                             n_jobs=N_JOBS, cache_dir=CACHE)
e2_none = runs_to_frame(e2_none_runs, e2_none_specs)
e2_none.to_csv("results/e2_none_runs.csv", index=False)

compare_names = [d.name for d in policy_detectors]
policy_cmp = pd.concat({
    "reset": e2[e2["detector"].isin(compare_names)].groupby("detector")[
        ["f1", "precision", "fp_out"]].mean(),
    "none": e2_none.groupby("detector")[["f1", "precision", "fp_out"]].mean(),
}, axis=1).reindex([n for n in detector_order if n in compare_names])
policy_cmp.set_axis([f"{a}:{b}" for a, b in policy_cmp.columns], axis=1).to_csv(
    "results/e2_policy.csv")
display(policy_cmp.round(3))

river_in_both = [n for n in RIVER if n in compare_names]
ranks_reset = average_ranks(block_matrix(e2[e2["detector"].isin(river_in_both)], "f1",
                                         methods=river_in_both)[0])
ranks_none = average_ranks(block_matrix(e2_none[e2_none["detector"].isin(river_in_both)], "f1",
                                        methods=river_in_both)[0])
policy_agreement = rank_agreement(ranks_reset, ranks_none)
display(pd.DataFrame({"ранг (reset)": ranks_reset, "ранг (none)": ranks_none}).round(2)
        .sort_values("ранг (reset)"))
print(f"согласие ранжирований reset/none: τ = {policy_agreement['tau']:.2f} "
      f"(p = {policy_agreement['p']:.3g}), ρ = {policy_agreement['rho']:.2f}")
```

### 5.4. Э4. Шкала сложности λ [К+Т]

**Вопрос.** Отражает ли шкала сложности λ (п. 4.6.5) реальную трудность обнаружения? Какие факторы шкалы её определяют?

**Метод.** Две части:
1. **Шкала целиком.** $\lambda \in \{0;\ 0{,}1;\ \dots;\ 1\}$; для каждого $\lambda$ — несколько структурных seed (своя геометрия концептов и своя реализация): 3 в FAST, 10 в FULL. Все детекторы. Шкала отражает сложность, если F1 убывает с ростом $\lambda$: $\rho_{\text{Спирмен}}(F_1, \lambda) < 0$. Дополнительно приводится доля шагов, на которых средняя кривая $F_1(\lambda)$ не возрастает.
2. **По одному фактору.** Каждый фактор проходит свой диапазон от «легко» до «трудно» ($\lambda_j \in \{0;\ 0{,}25;\ 0{,}5;\ 0{,}75;\ 1\}$), остальные стоят в центральной точке шкалы ($\lambda = 0{,}5$). Центральная точка выбрана потому, что вокруг крайней точки «легко» многие сочетания недостижимы. Например, при $m = 0{,}4$ и $\pi_0 = 0{,}1$ реальный дрейф не может превысить TV = 0,2. Недопустимые сценарии исключаются, их число выводится. В FAST KSWIN здесь не участвует ради времени.
```python
from scendrift.evaluation.analysis import monotone_share
from scendrift.evaluation.stats import holm
from scendrift.formation.mapping import point_to_dict

E4 = {"FAST": {"seeds": 3}, "FULL": {"seeds": 10}}[MODE]
LAMBDAS = [round(x, 2) for x in np.linspace(0, 1, 11)]
lam_specs = [DIFFICULTY.scenario(lam, concept_seed=r, seed=r, name=f"lambda-{lam:.1f}-c{r}")
             for lam in LAMBDAS for r in range(E4["seeds"])]
lam_runs = run_benchmark(lam_specs, ALL_DETECTORS, repeats=1, n_jobs=N_JOBS, cache_dir=CACHE)
lam_frame = runs_to_frame(lam_runs, lam_specs)
lam_frame["λ"] = lam_frame["scenario_id"].map({sio.scenario_id(s): s.meta["difficulty"]
                                               for s in lam_specs})
lam_frame.to_csv("results/e4_lambda_runs.csv", index=False)
lam_curve = lam_frame.groupby(["detector", "λ"])["f1"].agg(["mean", "std"])

fig, axes = plt.subplots(3, 3, figsize=(11, 8.4), sharex=True, sharey=True)
periodic_curve = lam_curve.loc["Periodic(2000)", "mean"]
for ax, name in zip(axes.ravel(), RIVER):
    c = lam_curve.loc[name]
    ax.fill_between(c.index, c["mean"] - c["std"], c["mean"] + c["std"], color=CATEGORICAL[0],
                    alpha=0.18, linewidth=0)
    ax.plot(c.index, c["mean"], color=CATEGORICAL[0], marker="o", markersize=3.5, linewidth=1.8,
            label="F1 детектора (среднее ± ст. откл.)")
    ax.plot(periodic_curve.index, periodic_curve, color=TEXT_SECONDARY, linestyle="--",
            linewidth=1, label="Periodic(2000)")
    ax.set_title(name, fontsize=10.5)
    ax.set_ylim(-0.03, 1.03)
for ax in axes[-1]:
    ax.set_xlabel("сложность λ")
for ax in axes[:, 0]:
    ax.set_ylabel("F1")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9.5, frameon=False)
decimal_comma(*axes.ravel())
fig.suptitle(f"Рис. 5.7. Э4: F1 вдоль шкалы сложности ({E4['seeds']} структурных seed на точку)",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0.04, 1, 1))
save_figure(fig, "docs/figures/fig_5_7_e4_lambda.png")
plt.show()

lam_rows = []
for name in [*RIVER, *BASELINE_NAMES]:
    part = lam_frame[lam_frame["detector"] == name]
    rho, p = (stats.spearmanr(part["λ"], part["f1"]) if part["f1"].nunique() > 1
              else (np.nan, np.nan))  # у базовых линий F1 постоянен
    curve = lam_curve.loc[name, "mean"]
    lam_rows.append({"детектор": name, "ρ(F1, λ)": rho, "p": p,
                     "доля невозрастающих шагов": monotone_share(curve.to_numpy()),
                     "F1 при λ = 0": curve.iloc[0], "F1 при λ = 1": curve.iloc[-1]})
lam_table = pd.DataFrame(lam_rows).set_index("детектор")
lam_table["p (Холм)"] = np.nan
lam_table.loc[RIVER, "p (Холм)"] = holm(lam_table.loc[RIVER, "p"].tolist())
lam_table.to_csv("results/e4_lambda_spearman.csv")
display(lam_table.round(4))
```
<!-- cell -->
**Окна допуска вдоль шкалы.** Ширина перехода растёт с $\lambda$, а вместе с ней и окна допуска $\Lambda_k$ длины $\ell + \Delta$. Чем большую долю потока покрывают окна, тем чаще случайная ложная тревога засчитывается как обнаружение. Ниже — доля потока после разогрева, покрытая окнами, и средний «случайный recall» детекторов river (п. 5.3) при нескольких значениях $\lambda$.
```python
lam_frame["случайный recall"] = 1 - np.exp(-lam_frame["far"] / 1000
                                           * (lam_frame["width"] + DELTA))
window_rows = []
for lam in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
    spec_l = DIFFICULTY.scenario(lam)
    covered = sum(e.width + DELTA for e in spec_l.events)
    part = lam_frame[(lam_frame["λ"] == lam) & lam_frame["detector"].isin(RIVER)]
    window_rows.append({"λ": lam, "ширина ℓ": spec_l.events[0].width,
                        "доля потока в окнах": covered / (spec_l.stream.n_samples
                                                          - spec_l.evaluation.warmup),
                        "recall (детекторы river)": part["recall"].mean(),
                        "случайный recall": part["случайный recall"].mean()})
window_table = pd.DataFrame(window_rows)
window_table.to_csv("results/e4_windows.csv", index=False)
display(window_table.round(3))
```
<!-- cell -->
**По одному фактору.** Таблица — $\rho(F_1, \lambda_j)$ для каждого детектора и фактора. Отрицательное значение означает, что фактор затрудняет обнаружение. Звёздочка — значимо после поправки Холма по пяти факторам внутри детектора.
```python
OFAT_LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
center = DIFFICULTY.values(0.5)
ofat_detectors = [d for d in ALL_DETECTORS if MODE == "FULL" or d.name != "KSWIN"]
ofat_index, ofat_specs, ofat_invalid = [], {}, {}
for factor in DIFFICULTY.factors:
    others = {k: v for k, v in center.items() if k != factor.name}
    scale_f = DifficultyScale(point_to_dict(others, DIFFICULTY.base), [factor])
    for lam in OFAT_LAMBDAS:
        for r in range(E4["seeds"]):
            spec_f = scale_f.scenario(lam, concept_seed=r, seed=r,
                                      name=f"ofat-{factor.name}-{lam}-c{r}")
            if not check(spec_f).ok:
                ofat_invalid[factor.name] = ofat_invalid.get(factor.name, 0) + 1
                continue
            sid = sio.scenario_id(spec_f)
            ofat_specs[sid] = spec_f
            ofat_index.append({"scenario_id": sid, "фактор": factor.name, "λ_j": lam})
print("исключено недопустимых сценариев:", ofat_invalid or "нет")
ofat_runs = run_benchmark(list(ofat_specs.values()), ofat_detectors, repeats=1, n_jobs=N_JOBS,
                          cache_dir=CACHE)
ofat_frame = runs_to_frame(ofat_runs, ofat_specs).merge(pd.DataFrame(ofat_index),
                                                         on="scenario_id")
ofat_frame.to_csv("results/e4_ofat_runs.csv", index=False)

ofat_rows = []
for name in [d.name for d in ofat_detectors if d.name in RIVER]:
    block = []
    for factor in DIFFICULTY.factors:
        part = ofat_frame[(ofat_frame["detector"] == name) & (ofat_frame["фактор"] == factor.name)]
        rho, p = stats.spearmanr(part["λ_j"], part["f1"]) if part["f1"].nunique() > 1 else (0.0, 1.0)
        block.append({"детектор": name, "фактор": factor.name, "ρ": rho, "p": p})
    for row, p_adj in zip(block, holm([b["p"] for b in block])):
        row["p (Холм)"] = p_adj
    ofat_rows += block
ofat_table = pd.DataFrame(ofat_rows)
ofat_table.to_csv("results/e4_ofat_spearman.csv", index=False)
ofat_view = ofat_table.assign(
    знак=lambda t: [f"{r:+.2f}{'*' if q < 0.05 else ''}".replace(".", ",")
                    for r, q in zip(t["ρ"], t["p (Холм)"])]).pivot(
    index="детектор", columns="фактор", values="знак")[[f.name for f in DIFFICULTY.factors]]
ofat_summary = ofat_table.groupby("фактор").agg(
    средний_ρ=("ρ", "mean"),
    значимо_отрицательных=("ρ", lambda s: int(((s < 0) & (ofat_table.loc[s.index, "p (Холм)"]
                                                          < 0.05)).sum())),
    детекторов=("ρ", "size")).reindex([f.name for f in DIFFICULTY.factors])
ofat_summary.to_csv("results/e4_ofat_summary.csv")
display(ofat_view)
display(ofat_summary.round(3))
```

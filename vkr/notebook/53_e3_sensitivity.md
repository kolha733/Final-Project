### 5.3. Э3. Чувствительность и области отказа [Т]

**Вопрос.** От каких параметров сценария зависит качество обнаружения и где детекторы отказывают?

**Метод.** Четыре части:
1. **Сетка «величина × ширина».** Базовый шаблон каталога: $d = 20$, $\pi_0 = 0{,}5$, $\eta = 0$, три постепенных реальных дрейфа. Варьируются только $m \in \{0{,}03;\ 0{,}05;\ 0{,}1;\ 0{,}2;\ 0{,}4\}$ и $\ell \in \{100, 400, 1000, 2000, 4000\}$. Это контролируемый эксперимент: остальные параметры фиксированы.
2. **Корреляции Спирмена** F1 с параметрами сценария на наборе Э2: один сценарий — одно наблюдение, поправка Холма по параметрам. У prior-событий доля признаков $\alpha$ не определена и в корреляцию для $\alpha$ не входит.
3. **Правила отказа.** Отказ — средний по повторам recall < 0,5. Дерево решений глубины 2 по параметрам сценария (не меньше 6 сценариев в листе); листья с преобладанием отказов выписываются как правила. Надёжность правил оценивается 5-кратной кросс-валидацией и сравнивается с ответом «класс большинства».
4. **«Слепота» к виртуальному дрейфу** (необязательный Э9). Детектор по потоку ошибок может «обнаружить» событие случайно, ложной тревогой. Если ложные тревоги — пуассоновский поток с интенсивностью FAR, вероятность хотя бы одной в окне $\Lambda$ длины $\ell + \Delta$ равна $1 - e^{-\mathrm{FAR}\,(\ell+\Delta)/1000}$. Наблюдаемый recall сравнивается с этим случайным уровнем отдельно для реального и виртуального дрейфа.
```python
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from scendrift.evaluation.analysis import PARAM_LABELS, failure_rules
from scendrift.formation.difficulty import BASE_TEMPLATE
from scendrift.reporting.style import DIVERGING, SEQUENTIAL

E3 = {"FAST": {"repeats": 3}, "FULL": {"repeats": 10}}[MODE]
GRID_M = [0.03, 0.05, 0.1, 0.2, 0.4]
GRID_W = [100, 400, 1_000, 2_000, 4_000]
grid_specs = [point_to_spec({"magnitude": m, "width": w}, BASE_TEMPLATE, concept_seed=0,
                            seed=GLOBAL_SEED, name=f"grid-m{m}-w{w}")
              for m in GRID_M for w in GRID_W]
assert all(check(s).ok for s in grid_specs)
grid_runs = run_benchmark(grid_specs, ALL_DETECTORS, repeats=E3["repeats"], n_jobs=N_JOBS,
                          cache_dir=CACHE)
grid = runs_to_frame(grid_runs, grid_specs)
grid.to_csv("results/e3_grid_runs.csv", index=False)
grid_mean = grid.groupby(["detector", "magnitude", "width"])[["f1", "recall", "mtd"]].mean()

seq_cmap = LinearSegmentedColormap.from_list("seq", list(SEQUENTIAL))
fig, axes = plt.subplots(3, 3, figsize=(11, 10), sharex=True, sharey=True)
for ax, name in zip(axes.ravel(), RIVER):
    mat = grid_mean.loc[name, "f1"].unstack("width").reindex(index=GRID_M, columns=GRID_W)
    ax.imshow(mat.to_numpy(), cmap=seq_cmap, vmin=0, vmax=1, origin="lower", aspect="auto")
    for i in range(len(GRID_M)):
        for j in range(len(GRID_W)):
            value = mat.iloc[i, j]
            ax.text(j, i, f"{value:.2f}".replace(".", ","), ha="center", va="center", fontsize=8,
                    color="white" if value > 0.6 else TEXT_SECONDARY)
    ax.set_title(name, fontsize=10.5)
    ax.set_xticks(range(len(GRID_W)), [str(w) for w in GRID_W])
    ax.set_yticks(range(len(GRID_M)), [f"{m:g}".replace(".", ",") for m in GRID_M])
    ax.grid(False)
for ax in axes[-1]:
    ax.set_xlabel("ширина перехода ℓ")
for ax in axes[:, 0]:
    ax.set_ylabel("величина m")
fig.suptitle(f"Рис. 5.5. Э3: F1 на сетке «величина × ширина» (реальный дрейф, K = 3, "
             f"{E3['repeats']} повтора)", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("docs/figures/fig_5_5_e3_grid.png")
plt.show()

grid_trend = []
for name in RIVER:
    g = grid_mean.loc[name].reset_index()
    rho_m = stats.spearmanr(g["magnitude"], g["f1"])[0]
    rho_w = stats.spearmanr(g["width"], g["f1"])[0]
    grid_trend.append({"детектор": name, "ρ(F1, m)": rho_m, "ρ(F1, ℓ)": rho_w,
                       "F1 при m = 0,03": g.loc[g["magnitude"] == 0.03, "f1"].mean(),
                       "F1 при m = 0,4": g.loc[g["magnitude"] == 0.4, "f1"].mean()})
grid_trend = pd.DataFrame(grid_trend).set_index("детектор")
display(grid_trend.round(3))
```
<!-- cell -->
**Корреляции на наборе Э2.** Звёздочка — значимо после поправки Холма ($\alpha = 0{,}05$).
```python
DELTA = e2_specs[0].evaluation.acceptance_window
E3_PARAMS = ["magnitude", "width", "alpha", "pi0", "eta", "d", "K", "n"]
e2_scen_corr = e2_scen.copy()
e2_scen_corr.loc[e2_scen_corr["kind"] == "prior", "alpha"] = np.nan
e2_rho = spearman_table(e2_scen_corr[e2_scen_corr["detector"].isin(RIVER)], "f1", E3_PARAMS)
e2_rho.to_csv("results/e3_spearman.csv", index=False)
rho_mat = e2_rho.pivot(index="detector", columns="параметр", values="ρ").reindex(
    index=order_f1, columns=E3_PARAMS)
sig_mat = e2_rho.pivot(index="detector", columns="параметр", values="p (Холм)").reindex(
    index=order_f1, columns=E3_PARAMS) < 0.05

div_cmap = LinearSegmentedColormap.from_list("div", list(DIVERGING))
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.imshow(rho_mat.to_numpy(), cmap=div_cmap, vmin=-1, vmax=1, aspect="auto")
for i in range(rho_mat.shape[0]):
    for j in range(rho_mat.shape[1]):
        v = rho_mat.iloc[i, j]
        mark = "*" if sig_mat.iloc[i, j] else ""
        ax.text(j, i, f"{v:+.2f}{mark}".replace(".", ","), ha="center", va="center", fontsize=8.5)
ax.set_xticks(range(len(E3_PARAMS)), [PARAM_LABELS[p] for p in E3_PARAMS])
ax.set_yticks(range(len(order_f1)), order_f1)
ax.grid(False)
fig.suptitle(f"Рис. 5.6. Э3: ρ Спирмена между F1 и параметрами сценария (набор "
             f"{e2_suite.name}, N = {len(e2_specs)})", x=0.01, ha="left", fontsize=12,
             fontweight="bold")
fig.tight_layout()
fig.savefig("docs/figures/fig_5_6_e3_spearman.png")
plt.show()
print("значимых пар (детектор, параметр):", int(sig_mat.to_numpy().sum()), "из", sig_mat.size)
display(e2_rho[e2_rho["p (Холм)"] < 0.05].round(4))
```
<!-- cell -->
**Механизм положительной связи F1 с шумом.** По утверждению 6 шум меток уменьшает наблюдаемый эффект реального дрейфа, поэтому положительная связь F1 с $\eta$ неожиданна. Проверим, не объясняется ли она ложными тревогами: шум увеличивает разброс потока ошибок, и тревоги, попавшие в окна допуска, засчитываются как обнаружения. Для детекторов со значимой положительной связью сравниваются $\rho$ шума с частотой ложных тревог, со случайным уровнем recall и с превышением recall над случайным уровнем.
```python
eta_detectors = e2_rho[(e2_rho["параметр"] == "eta") & (e2_rho["ρ"] > 0)
                       & (e2_rho["p (Холм)"] < 0.05)]["detector"].tolist()
e2_scen["случайный recall"] = 1 - np.exp(-e2_scen["far"] / 1000 * (e2_scen["width"] + DELTA))
e2_scen["превышение recall"] = e2_scen["recall"] - e2_scen["случайный recall"]
eta_rows = []
for name in eta_detectors:
    part = e2_scen[e2_scen["detector"] == name]
    eta_rows.append({"детектор": name, **{f"ρ(η, {col})": stats.spearmanr(part["eta"], part[col])[0]
                                         for col in ["f1", "far", "случайный recall",
                                                     "превышение recall"]}})
eta_mechanism = pd.DataFrame(eta_rows).set_index("детектор")
display(eta_mechanism.round(3))
```
<!-- cell -->
**Правила отказа.** Для каждого детектора приводится правило, покрывающее больше всего отказов.
```python
RULE_FEATURES = ["magnitude", "width", "alpha", "pi0", "eta", "d", "K", "n", "kind", "form",
                 "recurring"]
rule_rows = []
for name in order_f1:
    part = e2_scen[e2_scen["detector"] == name].copy()
    part["отказ"] = part["recall"] < 0.5
    res = failure_rules(part, RULE_FEATURES, "отказ", max_depth=2, min_samples_leaf=6,
                        seed=GLOBAL_SEED)
    top = res.rules.iloc[0] if len(res.rules) else None
    rule_rows.append({
        "детектор": name, "доля отказов": res.failure_share,
        "правило": top["правило"] if top is not None else "—",
        "сценариев": top["сценариев"] if top is not None else 0,
        "отказов в листе": top["доля отказов"] if top is not None else np.nan,
        "покрыто отказов": top["покрыто отказов"] if top is not None else np.nan,
        "точность (CV)": res.cv_accuracy, "класс большинства": res.baseline_accuracy})
rules_table = pd.DataFrame(rule_rows)
rules_table.to_csv("results/e3_failure_rules.csv", index=False)
display(rules_table.round(3))
```
<!-- cell -->
**«Слепота» к виртуальному дрейфу (Э9).** Для каждого детектора сравниваются наблюдаемый recall и случайный уровень, рассчитанный по его же частоте ложных тревог. Вид — вид первого события сценария.
```python
blind = e2[e2["detector"].isin(RIVER) & e2["kind"].isin(["real", "virtual"])].copy()
blind["случайный recall"] = 1 - np.exp(-blind["far"] / 1000 * (blind["width"] + DELTA))
blind_table = blind.groupby(["detector", "kind"])[["recall", "случайный recall"]].mean().unstack(
    "kind").reindex(order_f1)
blind_table.columns = [f"{a} ({b})" for a, b in blind_table.columns]
blind_table["превышение real"] = blind_table["recall (real)"] - blind_table["случайный recall (real)"]
blind_table["превышение virtual"] = (blind_table["recall (virtual)"]
                                     - blind_table["случайный recall (virtual)"])
blind_table.to_csv("results/e3_virtual_blindness.csv")
display(blind_table.round(3))
```

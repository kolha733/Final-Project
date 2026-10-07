### 5.6. Э6. Сравнение с ручным набором классических потоков [Т]

**Вопрос.** Совпадают ли выводы о детекторах на «ручном» наборе классических потоков и на автоматически сформированном наборе? Насколько разные области пространства сценариев покрывают эти наборы?

**Метод.**
- Ручной набор — 10 потоков: пять классических генераторов river (SEA [@sea2001], Agrawal [@agrawal1993], STAGGER [@stagger1986], Sine и Mixed [@ddm2004]) × две формы перехода. Формы — внезапная и постепенная с шириной 1000, как у ConceptDriftStream в MOA [@moa2010]. Три события, $n = 20\,000$, параметры river по умолчанию.
- Детекторы, повторы и протокол — как в Э2.
- Согласие ранжирований по F1 — τ Кендалла [@kendall1938] между средними рангами на ручном наборе и на наборе Э2.
- Покрытие — распределение фактической величины событий (TV) и число классов таксономии.

**Отличие от плана.** План предусматривал также Hyperplane и RandomRBFDrift из river. В них дрейф непрерывный: параметры меняются на каждом объекте, а дискретных событий с интервалами перехода нет. Поэтому для них не определены ground truth и метрики п. 2.6, и в набор они не включены.
```python
from scendrift.generators.base import get_generator
from scendrift.generators.classic import CLASSIC
from scendrift.generators.ground_truth import build_ground_truth

E6 = {"FAST": {"repeats": 3}, "FULL": {"repeats": 10}}[MODE]
manual_specs = []
for family, cs in CLASSIC.items():
    for form in ["sudden", "gradual"]:
        defaults = {"form": form, **({"width": 1_000} if form == "gradual" else {})}
        manual_specs.append(sio.from_dict({
            "name": f"{family.split(':')[1]}-{form}",
            "stream": {"family": family, "n_samples": 20_000, "n_features": cs.n_features},
            "drifts": {"schedule": {"count": 3}, "defaults": defaults}, "seed": GLOBAL_SEED}))
assert all(check(s).ok for s in manual_specs)
manual_runs = run_benchmark(manual_specs, ALL_DETECTORS, repeats=E6["repeats"], n_jobs=N_JOBS,
                            cache_dir=CACHE)
manual = runs_to_frame(manual_runs, manual_specs)
manual.to_csv("results/e6_manual_runs.csv", index=False)

f1_side = pd.DataFrame({
    "F1 (ручной)": manual.groupby("detector")["f1"].mean(),
    "F1 (Э2)": e2.groupby("detector")["f1"].mean(),
}).reindex(detector_order)
manual_matrix, _ = block_matrix(manual[manual["detector"].isin(RIVER)], "f1", methods=RIVER)
ranks_manual = average_ranks(manual_matrix)
ranks_auto = e5_ranks["f1"]
f1_side["ранг (ручной)"] = ranks_manual
f1_side["ранг (Э2)"] = ranks_auto
manual_agreement = rank_agreement(ranks_manual, ranks_auto)
display(f1_side.round(3))
print(f"согласие ранжирований по F1: τ = {manual_agreement['tau']:.2f} "
      f"(p = {manual_agreement['p']:.3g}), ρ = {manual_agreement['rho']:.2f}")
```
<!-- cell -->
**Покрытие.** Фактическая величина события — наибольшая из компонент TV (real, virtual, prior), вычисленных генератором.
```python
def event_magnitudes(specs: list) -> np.ndarray:
    values = []
    for spec in specs:
        gt = build_ground_truth(spec, get_generator(spec.stream.family).realized(spec))
        values += [max(iv.realized.values()) for iv in gt.intervals]
    return np.array(values)


mag_manual, mag_auto = event_magnitudes(manual_specs), event_magnitudes(e2_specs)
coverage = pd.DataFrame({
    "событий": [len(mag_manual), len(mag_auto)],
    "доля TV < 0,1": [np.mean(mag_manual < 0.1), np.mean(mag_auto < 0.1)],
    "доля TV < 0,25": [np.mean(mag_manual < 0.25), np.mean(mag_auto < 0.25)],
    "медиана TV": [np.median(mag_manual), np.median(mag_auto)],
    "классов таксономии": [len({signature(s) for s in manual_specs}),
                           len({signature(s) for s in e2_specs})],
    "видов дрейфа": [len({e.kind for s in manual_specs for e in s.events if e.kind}),
                     len({e.kind for s in e2_specs for e in s.events if e.kind})],
}, index=["ручной набор", f"набор Э2 ({e2_suite.name})"])
display(coverage.round(3))

fig, (ax_h, ax_r) = plt.subplots(1, 2, figsize=(12, 4.4))
bins = np.linspace(0, 1, 21)
ax_h.hist(mag_auto, bins=bins, density=True, color=CATEGORICAL[0], alpha=0.6,
          label=f"набор Э2 ({len(mag_auto)} событий)")
ax_h.hist(mag_manual, bins=bins, density=True, color=CATEGORICAL[1], alpha=0.6,
          label=f"ручной набор ({len(mag_manual)} событий)")
ax_h.set(xlabel="фактическая величина события (TV)", ylabel="плотность",
         title="Величина событий")
ax_h.legend(loc="upper right")
lim = (0.5, len(RIVER) + 0.5)
ax_r.plot(lim, lim, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
ax_r.scatter(ranks_auto[RIVER], ranks_manual[RIVER], color=CATEGORICAL[0], s=30)
for i, name in enumerate(ranks_manual[RIVER].sort_values().index):
    ax_r.annotate(name, (ranks_auto[name], ranks_manual[name]), fontsize=8,
                  xytext=(5, 4 if i % 2 == 0 else -11), textcoords="offset points")
ax_r.set(xlim=lim, ylim=lim, xlabel="средний ранг по F1, набор Э2",
         ylabel="средний ранг по F1, ручной набор",
         title=f"Ранжирования: τ Кендалла = {manual_agreement['tau']:.2f}".replace(".", ","))
decimal_comma(ax_h, ax_r)
fig.suptitle("Рис. 5.9. Э6: ручной набор классических потоков и автоматический набор",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("docs/figures/fig_5_9_e6_manual.png")
plt.show()
```

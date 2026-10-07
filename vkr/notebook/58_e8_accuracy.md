### 5.8. Э8. Влияние на accuracy [Т]

**Вопрос.** Как обнаружение дрейфа сказывается на prequential accuracy базовой модели? Согласуется ли ранжирование по accuracy с ранжированием по качеству обнаружения?

**Метод** (данные Э2, $a = \text{reset}$):
- $\Delta$accuracy относительно NoDrift на той же реализации потока;
- разрыв до Oracle — только для сценариев, у которых первое событие — реальный дрейф (п. 4.11: при виртуальном и prior-дрейфе сброс модели оракулом не помогает);
- согласие ранжирований по F1 и по $\Delta$accuracy (τ Кендалла по средним рангам);
- ρ Спирмена между F1 и $\Delta$accuracy по сценариям — для каждого детектора.
```python
acc_table = e2.groupby("detector")[["delta_accuracy", "f1"]].agg(["mean", "std"]).reindex(
    detector_order)
real_first = e2[e2["kind"] == "real"]
acc_table[("oracle_gap (real)", "mean")] = real_first.groupby("detector")["oracle_gap"].mean()
acc_table[("oracle_gap (real)", "std")] = real_first.groupby("detector")["oracle_gap"].std()
display(format_table(acc_table))

ranks_acc = e5_ranks["delta_accuracy"]
acc_agreement = rank_agreement(e5_ranks["f1"], ranks_acc)
print(f"согласие ранжирований F1 и Δaccuracy: τ = {acc_agreement['tau']:.2f} "
      f"(p = {acc_agreement['p']:.3g})")

corr_rows = []
for name in RIVER:
    part = e2_scen[e2_scen["detector"] == name]
    rho, p = stats.spearmanr(part["f1"], part["delta_accuracy"])
    corr_rows.append({"детектор": name, "ρ(F1, Δacc)": rho, "p": p})
corr_acc = pd.DataFrame(corr_rows).set_index("детектор")
corr_acc["p (Холм)"] = holm(corr_acc["p"].tolist())
display(corr_acc.reindex(order_f1).round(4))

means = e2.groupby("detector")[["f1", "delta_accuracy"]].mean()
fig, ax = plt.subplots(figsize=(7.5, 5))
for name, row in means.iterrows():
    is_base = name in BASELINE_NAMES
    ax.scatter(row["f1"], row["delta_accuracy"], s=36, marker="s" if is_base else "o",
               color=TEXT_SECONDARY if is_base else CATEGORICAL[0])
    ax.annotate(name, (row["f1"], row["delta_accuracy"]), fontsize=8.5, xytext=(4, 3),
                textcoords="offset points")
ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
ax.set(xlabel="средний F1", ylabel="средний прирост accuracy относительно NoDrift",
       title=f"τ Кендалла (F1 и Δaccuracy, детекторы river) = {acc_agreement['tau']:.2f}"
       .replace(".", ","))
decimal_comma(ax)
fig.suptitle("Рис. 5.10. Э8: качество обнаружения и влияние на accuracy", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("docs/figures/fig_5_10_e8_accuracy.png")
plt.show()
```

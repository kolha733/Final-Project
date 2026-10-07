### 5.5. Э5. Статистическое сравнение [Т]

**Вопрос.** Какие различия между детекторами на наборе Э2 статистически значимы?

**Метод** (п. 4.10). Блок — сценарий набора Э2, значение метрики усредняется по повторам. Сравниваются девять детекторов river по четырём метрикам:
- F1 (больше — лучше);
- MTD (меньше — лучше); блоки, где какой-либо детектор ни разу не обнаружил событие, исключаются;
- FAR (меньше — лучше);
- $\Delta$accuracy относительно NoDrift (больше — лучше).

Для каждой метрики — критерий Фридмана с поправкой Имана–Давенпорта и CD-диаграмма Неменьи ($\alpha = 0{,}05$). Для F1 дополнительно — попарный критерий Уилкоксона с поправкой Холма и сравнение каждого детектора с периодическим сбросом.
```python
E5_METRICS = [("f1", True, "F1"), ("mtd", False, "MTD"), ("far", False, "FAR"),
              ("delta_accuracy", True, "Δaccuracy")]
e5_rows, e5_ranks = [], {}
fig, axes = plt.subplots(4, 1, figsize=(9.5, 13))
for ax, (metric, better_high, label) in zip(axes, E5_METRICS):
    mat, dropped = block_matrix(e2[e2["detector"].isin(RIVER)], metric, methods=RIVER)
    ranks = average_ranks(mat, higher_is_better=better_high)
    fr = friedman(mat)
    cd, _ = nemenyi_cd(mat.shape[1], mat.shape[0])
    e5_ranks[metric] = ranks
    cd_diagram(ranks, cd, ax=ax, title=f"{label}: N = {mat.shape[0]}, Фридман p = "
               f"{fr.p_chi2:.2g}, Иман–Давенпорт p = {fr.p_f:.2g}".replace(".", ","))
    e5_rows.append({"метрика": label, "блоков": mat.shape[0], "исключено": dropped,
                    "χ²_F": fr.chi2, "p": fr.p_chi2, "F_F": fr.f_stat, "p (F)": fr.p_f, "CD": cd,
                    "лучший по рангу": ranks.index[0], "худший по рангу": ranks.index[-1]})
fig.suptitle("Рис. 5.8. Э5: CD-диаграммы (Неменьи, α = 0,05) на наборе Э2", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("docs/figures/fig_5_8_e5_cd.png")
plt.show()
e5_table = pd.DataFrame(e5_rows)
e5_table.to_csv("results/e5_friedman.csv", index=False)
display(e5_table.round(4))
```
<!-- cell -->
**Попарные сравнения по F1** (Уилкоксон, поправка Холма по всем 36 парам). Приводятся пары с обнаруженным различием.
```python
f1_matrix_e2, _ = block_matrix(e2[e2["detector"].isin(RIVER)], "f1", methods=RIVER)
e5_pairs = wilcoxon_holm(f1_matrix_e2)
e5_pairs.to_csv("results/e5_wilcoxon_f1.csv", index=False)
significant_pairs = e5_pairs[e5_pairs["различие"] != "не обнаружено"]
print(f"пар с различием: {len(significant_pairs)} из {len(e5_pairs)}")
display(significant_pairs.round(4))
```
<!-- cell -->
**Сравнение с периодическим сбросом** (иллюзия прогресса, п. 2.6). Для каждого детектора — критерий Уилкоксона против Periodic(2000) по F1 и по $\Delta$accuracy, поправка Холма по девяти детекторам. Положительная медиана разности означает, что детектор лучше периодического сброса.
```python
vs_rows = []
for metric in ["f1", "delta_accuracy"]:
    mat, _ = block_matrix(e2[e2["detector"].isin([*RIVER, "Periodic(2000)"])], metric,
                          methods=[*RIVER, "Periodic(2000)"])
    pvals, rows_m = [], []
    for name in RIVER:
        diff = mat[name] - mat["Periodic(2000)"]
        p = 1.0 if np.all(diff == 0) else stats.wilcoxon(diff, zero_method="zsplit").pvalue
        pvals.append(p)
        rows_m.append({"метрика": metric, "детектор": name, "медиана разности": diff.median(),
                       "доля сценариев, где лучше": float((diff > 0).mean()), "p": p})
    for row, p_adj in zip(rows_m, holm(pvals)):
        row["p (Холм)"] = p_adj
        row["вывод"] = ("лучше Periodic" if p_adj < 0.05 and row["медиана разности"] > 0 else
                        "хуже Periodic" if p_adj < 0.05 else "не обнаружено")
    vs_rows += rows_m
vs_periodic = pd.DataFrame(vs_rows)
vs_periodic.to_csv("results/e5_vs_periodic.csv", index=False)
display(vs_periodic.pivot(index="детектор", columns="метрика", values="вывод").reindex(order_f1))
display(vs_periodic.round(4))
```

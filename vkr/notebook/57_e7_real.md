### 5.7. Э7. Реальные данные с внедрённым дрейфом [К]

**Вопрос.** Переносятся ли выводы о детекторах с синтетического семейства на реальные данные с внедрённым дрейфом?

**Метод.**
- Наборы Bananas ($d = 2$) и Phishing ($d = 9$), п. 4.5.2.
- Для каждого набора — три вида дрейфа × две величины $m \in \{0{,}1;\ 0{,}3\}$: три постепенных события с шириной 1000, $\pi_0 = 0{,}45$, $n = 20\,000$.
- Синтетический двойник — те же параметры в семействе `hyperplane_gauss` той же размерности. В итоге 24 сценария.
- Сравниваются recall по видам дрейфа и ранжирования по F1 (τ Кендалла) на реальных данных и на двойниках.
- На реальных данных реальный и виртуальный дрейф побочно меняют $P(y)$. Сценарии с предупреждением W4 отмечаются.
```python
E7 = {"FAST": {"repeats": 3}, "FULL": {"repeats": 10}}[MODE]
REAL_SETS = {"real:bananas": 2, "real:phishing": 9}
e7_specs, e7_index = [], []
for family, d_set in REAL_SETS.items():
    for source in ["реальные", "синтетика"]:
        for kind in ["real", "virtual", "prior"]:
            for m in [0.1, 0.3]:
                stream = {"family": family if source == "реальные" else "hyperplane_gauss",
                          "n_samples": 20_000, "n_features": d_set, "minority_share": 0.45}
                spec_r = sio.from_dict({
                    "name": f"{family.split(':')[1]}-{source}-{kind}-{m}", "stream": stream,
                    "drifts": {"schedule": {"count": 3}, "defaults": {
                        "form": "gradual", "width": 1_000, "kind": kind, "magnitude": m}},
                    "seed": GLOBAL_SEED})
                report_r = check(spec_r)
                assert report_r.ok, report_r
                e7_specs.append(spec_r)
                e7_index.append({"scenario_id": sio.scenario_id(spec_r), "набор": family.split(":")[1],
                                 "источник": source, "W4": "W4" in report_r.codes()})
e7_runs = run_benchmark(e7_specs, ALL_DETECTORS, repeats=E7["repeats"], n_jobs=N_JOBS,
                        cache_dir=CACHE)
e7 = runs_to_frame(e7_runs, e7_specs).merge(pd.DataFrame(e7_index), on="scenario_id")
e7.to_csv("results/e7_runs.csv", index=False)
print("сценариев с W4:", int(pd.DataFrame(e7_index)["W4"].sum()), "из", len(e7_specs))

e7_recall = e7[e7["detector"].isin(RIVER)].pivot_table(
    index="detector", columns=["источник", "kind"], values="recall", aggfunc="mean"
).reindex(order_f1)
e7_recall.set_axis([f"{a}:{b}" for a, b in e7_recall.columns], axis=1).to_csv(
    "results/e7_recall.csv")
display(e7_recall.round(3))
```
<!-- cell -->
**Согласие ранжирований** по F1: реальные данные против синтетических двойников. Блоки — 12 сценариев каждого источника.
```python
e7_ranks = {}
for source in ["реальные", "синтетика"]:
    part = e7[(e7["источник"] == source) & e7["detector"].isin(RIVER)]
    mat, _ = block_matrix(part, "f1", methods=RIVER)
    e7_ranks[source] = average_ranks(mat)
e7_agreement = rank_agreement(e7_ranks["реальные"], e7_ranks["синтетика"])
e7_kind_gap = (e7_recall["реальные"] - e7_recall["синтетика"]).mean()
display(pd.DataFrame(e7_ranks).round(2).sort_values("синтетика"))
print(f"согласие ранжирований: τ = {e7_agreement['tau']:.2f} (p = {e7_agreement['p']:.3g}); "
      f"средняя разность recall «реальные − синтетика» по видам: "
      f"{ {k: round(v, 3) for k, v in e7_kind_gap.items()} }")
```

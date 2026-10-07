### 5.1. Э1. Валидация генератора [К]

**Вопрос.** Совпадают ли параметры дрейфа в порождённых данных с заданными: величина, форма перехода, доля класса, шум? Воспроизводим ли поток по seed?

**Метод.** Фактическая величина оценивается **по данным**, независимо от ground truth (модуль `diagnostics`, п. 4.4):
- real — доля смены метки;
- virtual — сдвиг проекций, пересчитанный в TV;
- prior — разность долей класса.

У каждой оценки есть стандартная ошибка (SE). Генератор считается корректным, если отклонения согласуются с выборочной погрешностью: нормированные отклонения $z = (\hat m - m)/\mathrm{SE}$ распределены примерно как $\mathcal N(0, 1)$.
```python
from pathlib import Path

from scendrift.generators.diagnostics import digest, empirical_magnitude, transition_curve

E1 = {
    "FAST": {"n": 20_000, "seeds": 5, "random_scenarios": 900},
    "FULL": {"n": 40_000, "seeds": 20, "random_scenarios": 3000},
}[MODE]
RESULTS, FIGURES = Path("results"), Path("docs/figures")
print(E1)
```
<!-- cell -->
#### Э1.1. Заданная и фактическая величина на сетке

Одно внезапное событие в середине потока: $d = 10$, $\pi_0 = 0{,}3$, $\alpha = 0{,}5$ для real и virtual, без шума. Для каждой пары (вид, $m$) — несколько seed реализации.

При одном и том же seed базовые случайные величины выборки совпадают для всех $m$: меняется только детерминированное преобразование. Поэтому строки одного вида при разных $m$ **не независимы**. Особенно это заметно для virtual: сдвиг центра не меняет «шумовую» часть проекций, и $z$ почти одинаковы для всех $m$. Независимую проверку даёт п. Э1.2.
```python
def single_event(kind: str, m: float, n: int, **stream) -> ScenarioSpec:
    event = {"position": n // 2, "kind": kind, "magnitude": m}
    if kind != "prior":
        event["affected_share"] = 0.5
    return sio.from_dict({"stream": {"n_samples": n, "n_features": 10, "minority_share": 0.3,
                                     **stream}, "events": [event]})


grid_rows = []
for kind in ["real", "virtual", "prior"]:
    for m in [0.05, 0.1, 0.2, 0.3, 0.4]:
        spec_m = single_event(kind, m, E1["n"])
        for seed in range(E1["seeds"]):
            est, se = empirical_magnitude(spec_m, generate(spec_m, seed=seed), 0)
            grid_rows.append({"вид": kind, "m": m, "seed": seed, "m̂": est, "SE": se,
                              "z": (est - m) / se})
grid = pd.DataFrame(grid_rows)
grid.to_csv(RESULTS / "e1_grid.csv", index=False)
summary_grid = grid.groupby(["вид", "m"]).agg(
    m̂_среднее=("m̂", "mean"), m̂_ст_откл=("m̂", "std"), SE_среднее=("SE", "mean"),
    z_среднее=("z", "mean"), max_abs_z=("z", lambda z: z.abs().max()))
display(summary_grid.round(4))
print(f"Доля |z| ≤ 3: {(grid['z'].abs() <= 3).mean():.3f};  "
      f"ст. откл. z: {grid['z'].std():.2f} (ожидается ≈ 1)")
```
<!-- cell -->
```python
fig, ax = plt.subplots(figsize=(6.4, 5.2))
offsets = {"real": -0.006, "virtual": 0.0, "prior": 0.006}
labels = {"real": "real (доля смены метки)", "virtual": "virtual (TV по проекции)",
          "prior": "prior (|Δπ|)"}
ax.plot([0, 0.45], [0, 0.45], color=TEXT_SECONDARY, linewidth=1, linestyle="--",
        label="m̂ = m")
for color, kind in zip(CATEGORICAL, ["real", "virtual", "prior"]):
    part = grid[grid["вид"] == kind].groupby("m")["m̂"].agg(["mean", "std"])
    ax.errorbar(part.index + offsets[kind], part["mean"], yerr=part["std"], fmt="o",
                color=color, markersize=6, capsize=3, linewidth=1.5, label=labels[kind])
ax.set(xlabel="заданная величина m", ylabel="оценка по данным m̂ (среднее ± ст. откл.)",
       title="Заданная и фактическая величина дрейфа", xlim=(0, 0.45), ylim=(0, 0.45))
ax.legend(loc="upper left")
decimal_comma(ax)
fig.suptitle("Рис. 5.1. Э1: калибровка величины на уровне данных", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "fig_5_1_e1_calibration.png")
plt.show()
```
<!-- cell -->
#### Э1.2. Случайные сценарии: согласуются ли отклонения с выборочной погрешностью

Берутся случайные сценарии со случайными параметрами в допустимой области:
- вид дрейфа;
- величина $m$;
- доля класса $\pi_0 \in [0{,}05;\,0{,}5]$;
- доля затронутых признаков $\alpha$;
- размерность $d \in [3, 40]$.

Если генератор реализует модель, нормированные отклонения $z$ имеют среднее около 0 и стандартное отклонение около 1.

Объём выбран так, чтобы отличить ст. откл. 1 от заметно меньшего значения, например при завышенной SE. Выборочное ст. откл. по $n$ значениям имеет относительную погрешность около $1/\sqrt{2n}$.

В пробных прогонах было 100 и 300 сценариев (≈ 33 и ≈ 100 на вид). Ст. откл. $z$ для virtual составило 0,76 и 0,84. С учётом поправки на три сравнения это ещё совместимо с 1, но уверенного вывода не даёт. Поэтому объём увеличен до 900 сценариев: ≈ 300 на вид, погрешность ≈ 4 %.

Критерий: для каждого вида строится доверительный интервал для ст. откл. по распределению $\chi^2$. Проверяются три вида одновременно, поэтому применяется поправка Бонферрони: уровень каждого интервала равен $1 - 0{,}05/3$. Если 1 попадает в интервал, отклонение от модели не обнаружено.
```python
rand_rng = np.random.default_rng(GLOBAL_SEED + 1)
rand_rows = []
for i in range(E1["random_scenarios"]):
    kind = str(rand_rng.choice(["real", "virtual", "prior"]))
    d_i = int(rand_rng.integers(3, 41))
    pi0 = float(rand_rng.uniform(0.05, 0.5))
    alpha = float(rand_rng.uniform(2.5 / d_i, 1.0))
    if kind == "real":
        upper = sem.max_real_severity(sem.affected_count(alpha, d_i) / d_i, pi0, pi0)
    elif kind == "virtual":
        upper = 0.8
    else:
        upper = 1 - 0.01 - pi0
    m = float(rand_rng.uniform(0.03, 0.9) * upper)
    event = {"position": E1["n"] // 2, "kind": kind, "magnitude": m, "affected_share": alpha}
    spec_r = sio.from_dict({"stream": {"n_samples": E1["n"], "n_features": d_i,
                                       "minority_share": pi0, "concept_seed": i},
                            "events": [event], "seed": i})
    est, se = empirical_magnitude(spec_r, generate(spec_r), 0)
    rand_rows.append({"вид": kind, "d": d_i, "π₀": pi0, "α": alpha, "m": m, "m̂": est, "SE": se,
                      "z": (est - m) / se})
rand = pd.DataFrame(rand_rows)
rand.to_csv(RESULTS / "e1_random.csv", index=False)


def sd_interval(z: pd.Series, level: float = 0.95) -> tuple[float, float]:
    """Доверительный интервал для ст. откл. нормальной выборки (через χ²)."""
    from scipy.stats import chi2

    df_z, s = len(z) - 1, float(z.std())
    return (s * np.sqrt(df_z / chi2.ppf((1 + level) / 2, df_z)),
            s * np.sqrt(df_z / chi2.ppf((1 - level) / 2, df_z)))


rand_summary = rand.groupby("вид").agg(
    сценариев=("z", "size"), z_среднее=("z", "mean"), z_ст_откл=("z", "std"),
    max_abs_z=("z", lambda z: z.abs().max()),
    max_abs_ошибка=("m", lambda s: (rand.loc[s.index, "m̂"] - s).abs().max()))
ci = rand.groupby("вид")["z"].apply(sd_interval, level=1 - 0.05 / 3)
rand_summary["ДИ_ст_откл"] = [f"[{lo:.2f}; {hi:.2f}]" for lo, hi in ci]
display(rand_summary.round(3))
```
<!-- cell -->
```python
from scipy.stats import norm, shapiro

fig, ax = plt.subplots(figsize=(6.4, 3.8))
bins = np.linspace(-4, 4, 25)
ax.hist(rand["z"], bins=bins, density=True, color=CATEGORICAL[0], alpha=0.75,
        edgecolor="white", label=f"z по {len(rand)} сценариям")
grid_z = np.linspace(-4, 4, 200)
ax.plot(grid_z, norm.pdf(grid_z), color=CATEGORICAL[1], label="плотность N(0, 1)")
ax.set(xlabel="нормированное отклонение z = (m̂ − m)/SE", ylabel="плотность",
       title="Отклонения оценки от заданной величины")
ax.legend(loc="upper right")
decimal_comma(ax)
fig.suptitle("Рис. 5.2. Э1: распределение нормированных отклонений", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "fig_5_2_e1_zscores.png")
plt.show()
print(f"Критерий Шапиро–Уилка для z: p = {shapiro(rand['z']).pvalue:.3f}")
```
<!-- cell -->
#### Э1.3. Форма перехода

Реальный дрейф величины 0,3 с шириной перехода $\ell = 6000$ для четырёх вариантов: внезапный, постепенный с линейной и сигмоидной функцией, инкрементальный.

Эмпирическая кривая — скользящая (окно 800) доля объектов, у которых метка расходится с правилом старого концепта. Сравнивать её нужно с теоретической кривой, сглаженной тем же окном: иначе ступенька внезапного дрейфа даёт расхождение около $m/2$, вызванное только сглаживанием. Теоретическая кривая:
- для смеси (sudden, gradual) — $p(t)\,m$;
- для инкрементального перехода — $m(\theta(t))$, где $\theta(t)$ — угол при повороте на $\psi\,p(t)$.
```python
variants = [("внезапный", {"form": "sudden"}),
            ("постепенный, линейный", {"form": "gradual", "width": 6000}),
            ("постепенный, сигмоидный", {"form": "gradual", "width": 6000, "shape": "sigmoid"}),
            ("инкрементальный, линейный", {"form": "incremental", "width": 6000})]
fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True, sharey=True)
curve_errors = []
for ax, (title, form) in zip(axes.ravel(), variants):
    spec_c = sio.from_dict({"stream": {"n_samples": 40_000, "n_features": 10,
                                       "minority_share": 0.3},
                            "events": [{"position": 20_000, "kind": "real", "magnitude": 0.3,
                                        **form}]})
    curve = transition_curve(spec_c, generate(spec_c), 0, window=800)
    ax.plot(curve["t"], curve["empirical"], color=CATEGORICAL[0], linewidth=1.5,
            label="эмпирическая (окно 800)")
    ax.plot(curve["t"], curve["theory"], color=CATEGORICAL[1], linewidth=2, linestyle="--",
            label="теоретическая")
    ax.plot(curve["t"], curve["theory_smoothed"], color=CATEGORICAL[2], linewidth=1.5,
            linestyle=":", label="теоретическая, сглаженная")
    ax.set_title(title, fontsize=10.5)
    valid = curve.dropna()
    curve_errors.append({
        "переход": title,
        "max |эмп. − теор.|": (valid["empirical"] - valid["theory"]).abs().max(),
        "max |эмп. − сглаж. теор.|": (valid["empirical"] - valid["theory_smoothed"]).abs().max(),
    })
for ax in axes[1]:
    ax.set_xlabel("номер объекта t")
for ax in axes[:, 0]:
    ax.set_ylabel("доля смены метки")
axes[0, 0].legend(loc="upper left")
decimal_comma(*axes.ravel())
fig.suptitle("Рис. 5.3. Э1: форма перехода реального дрейфа (m = 0,3)", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "fig_5_3_e1_transitions.png")
plt.show()
display(pd.DataFrame(curve_errors).round(4))
```
<!-- cell -->
#### Э1.4. Дисбаланс классов и шум меток
```python
noise_rows = []
for pi0 in [0.05, 0.1, 0.3, 0.5]:
    for eta in [0.0, 0.1, 0.2]:
        spec_n = sio.from_dict({"stream": {"n_samples": E1["n"], "n_features": 10,
                                           "minority_share": pi0, "label_noise": eta},
                                "events": []})
        stream_n = generate(spec_n)
        noise_rows.append({"π₀": pi0, "η": eta, "P̂(y_чист=1)": stream_n.y_clean.mean(),
                           "доля инверсий": np.mean(stream_n.y != stream_n.y_clean),
                           "SE(π)": np.sqrt(pi0 * (1 - pi0) / E1["n"])})
noise_table = pd.DataFrame(noise_rows)
noise_table.to_csv(RESULTS / "e1_noise.csv", index=False)
display(noise_table.round(4))
```
<!-- cell -->
#### Э1.5. Воспроизводимость

Проверяются пять свойств для каждого семейства:
1. одинаковый seed даёт побитово одинаковый поток;
2. другой seed даёт другой поток;
3. концепты (ground truth) не зависят от seed реализации;
4. сохранение сценария в YAML и загрузка дают тот же поток;
5. изменение шума не меняет признаки.
```python
repro_rows = []
for family, spec_f in demo_specs.items():
    a, b = generate(spec_f, seed=1), generate(spec_f, seed=1)
    c = generate(spec_f, seed=2)
    reloaded = sio.loads_yaml(sio.dumps_yaml(spec_f))
    noisy = spec_f.model_copy(update={"stream": spec_f.stream.model_copy(
        update={"label_noise": 0.2})})
    repro_rows.append({
        "семейство": family,
        "тот же seed → тот же поток": digest(a) == digest(b),
        "другой seed → другой поток": digest(a) != digest(c),
        "ground truth не зависит от seed": a.ground_truth == c.ground_truth,
        "YAML → тот же поток": digest(generate(reloaded, seed=1)) == digest(a),
        "шум не меняет X": np.array_equal(generate(noisy, seed=1).X, a.X),
        "хэш потока": digest(a),
    })
repro = pd.DataFrame(repro_rows)
display(repro)
assert repro.drop(columns=["семейство", "хэш потока"]).all().all()
```
<!-- cell -->
#### Э1.6. Классические потоки на единой шкале

Насколько велики изменения в «ручных» бенчмарках? Для каждого классического генератора вычисляем TV смены соседних вариантов и побочное изменение доли класса $|\Delta P(y)|$.
```python
from scendrift.generators.classic import CLASSIC, switch_magnitude

classic_rows = []
for family, cs in CLASSIC.items():
    for a in range(cs.n_variants):
        b = (a + 1) % cs.n_variants
        if cs.n_variants == 2 and a == 1:
            continue
        tv, d_prior = switch_magnitude(family, a, b)
        classic_rows.append({"генератор": family.split(":")[1], "смена": f"{a} → {b}",
                             "TV": tv, "|ΔP(y)|": d_prior})
classic_table = pd.DataFrame(classic_rows)
classic_table.to_csv(RESULTS / "e1_classic.csv", index=False)
display(classic_table.round(4))
print(classic_table.groupby("генератор")["TV"].agg(["min", "median", "max"]).round(3))
```
<!-- cell -->
#### Э1.7. Внедрение дрейфа в реальные данные

Заданная и фактическая величина: фактическая вычисляется точно на эмпирической мере. Отдельно приводится побочное изменение $P(y)$.
```python
from scendrift.generators.real import build_real_chain, load

real_rows = []
for family, d_f in [("real:bananas", 2), ("real:phishing", 9)]:
    n_rows = load(family).n_rows
    for kind in ["real", "virtual", "prior"]:
        for m in [0.1, 0.2, 0.3]:
            spec_rd = sio.from_dict({"stream": {"family": family, "n_samples": 20_000,
                                                "n_features": d_f, "minority_share": 0.45},
                                     "events": [{"position": 10_000, "kind": kind,
                                                 "magnitude": m}]})
            chain_rd = build_real_chain(spec_rd)
            realized_rd = chain_rd.geometry[0].realized[kind]
            real_rows.append({"набор": family.split(":")[1], "строк": n_rows, "вид": kind,
                              "m": m, "m̃": realized_rd, "|m̃ − m|": abs(realized_rd - m),
                              "|ΔP(y)|": abs(chain_rd.states[1].prior - chain_rd.states[0].prior),
                              "предупреждения": ", ".join(sorted(check(spec_rd).codes())) or "—"})
real_table = pd.DataFrame(real_rows)
real_table.to_csv(RESULTS / "e1_real.csv", index=False)
display(real_table.round(5))
```
<!-- cell -->
#### Э1.8. Производительность генерации
```python
perf_rows = []
for family, spec_f in demo_specs.items():
    big = spec_f.model_copy(update={"stream": spec_f.stream.model_copy(
        update={"n_samples": 100_000})})
    start = time.perf_counter()
    generate(big)
    perf_rows.append({"семейство": family,
                      "секунд на 10⁵ объектов": round(time.perf_counter() - start, 3)})
display(pd.DataFrame(perf_rows))
```
<!-- cell -->
#### Э1. Проверка утверждений выводов

Каждое утверждение раздела «Выводы» проверяется по результатам ячеек выше. Таблица защищает текст выводов от расхождения с числами при перезапуске, например в режиме FULL. Если в столбце «выполнено» появится False, выводы нужно пересмотреть.
```python
claims = [
    ("сетка: все |z| ≤ 3", f"max |z| = {grid['z'].abs().max():.2f}",
     bool((grid["z"].abs() <= 3).all())),
    ("сетка: |среднее m̂ − m| ≤ 0,005",
     f"{(summary_grid['m̂_среднее'] - summary_grid.index.get_level_values('m')).abs().max():.4f}",
     bool(((summary_grid["m̂_среднее"] - summary_grid.index.get_level_values("m")).abs()
           <= 0.005).all())),
    ("случайные сценарии: 1 ∈ ДИ ст. откл. z для каждого вида",
     ", ".join(f"{k}: [{lo:.2f}; {hi:.2f}]" for k, (lo, hi) in ci.items()),
     bool(all(lo <= 1 <= hi for lo, hi in ci))),
    ("случайные сценарии: |среднее z| ≤ 3/√n для каждого вида",
     ", ".join(f"{k}: {v:.3f}" for k, v in rand_summary["z_среднее"].items()),
     bool((rand_summary["z_среднее"].abs() <= 3 / np.sqrt(rand_summary["сценариев"])).all())),
    ("форма перехода: max |эмп. − сглаж. теор.| < 0,05",
     f"{pd.DataFrame(curve_errors)['max |эмп. − сглаж. теор.|'].max():.4f}",
     bool((pd.DataFrame(curve_errors)["max |эмп. − сглаж. теор.|"] < 0.05).all())),
    ("доля класса: |P̂ − π₀| ≤ 2·SE",
     f"{((noise_table['P̂(y_чист=1)'] - noise_table['π₀']).abs() / noise_table['SE(π)']).max():.2f}·SE",
     bool(((noise_table["P̂(y_чист=1)"] - noise_table["π₀"]).abs()
           <= 2 * noise_table["SE(π)"]).all())),
    ("воспроизводимость: все проверки True", "—",
     bool(repro.drop(columns=["семейство", "хэш потока"]).all().all())),
    ("реальные данные: |m̃ − m| ≤ 1/число строк",
     f"{(real_table['|m̃ − m|'] * real_table['строк']).max():.2f} строки",
     bool((real_table["|m̃ − m|"] <= 1 / real_table["строк"] + 1e-12).all())),
    ("реальные данные: virtual не вызывает W4",
     f"max |ΔP(y)| = {real_table.loc[real_table['вид'] == 'virtual', '|ΔP(y)|'].max():.4f}",
     bool(~real_table.loc[real_table["вид"] == "virtual", "предупреждения"]
          .str.contains("W4").any())),
]
display(pd.DataFrame(claims, columns=["утверждение", "значение", "выполнено"]))
```
<!-- cell -->
#### Э1. Выводы

Числа приведены по прогону FAST (п. 0.5). При том же режиме и seed они воспроизводятся точно, кроме времени в Э1.8, которое зависит от машины.

1. **Калибровка величины** (Э1.1, Э1.2, рис. 5.1, 5.2).
   - На сетке из 75 прогонов все $|z| \le 3$ (максимум 2,36), ст. откл. $z$ равно 0,99. Средняя оценка отличается от заданной величины не более чем на 0,0025.
   - Сетка использует общие seed для разных $m$, поэтому её строки не независимы. Для virtual $z$ почти одинаковы при всех $m$: сдвиг центра не меняет «шумовую» часть проекций.
   - На 900 независимых случайных сценариях средние $z$ лежат в пределах от −0,02 до 0,07, ст. откл. — от 0,95 до 0,97.
   - Доверительные интервалы для ст. откл. (с поправкой Бонферрони) накрывают 1 для всех трёх видов. Критерий Шапиро–Уилка не отвергает нормальность ($p = 0{,}23$).
   - Систематического смещения и ошибки в стандартной ошибке не обнаружено. Это верно для всех видов дрейфа при $\pi_0 \in [0{,}05;\,0{,}5]$, $d \in [3, 40]$ и частичном охвате признаков.
   - Наибольшая абсолютная ошибка — у prior (0,022): у разности долей класса больше выборочная дисперсия. Для real она равна 0,011, для virtual — 0,016.
2. **Форма перехода** (Э1.3, рис. 5.3).
   - Для всех четырёх форм эмпирическая кривая отличается от сглаженной теоретической не более чем на 0,026–0,040. Для окна 800 при доле 0,3 это 1,6–2,5 стандартной ошибки скользящей доли.
   - Несглаженная ступенька внезапного дрейфа даёт формальное расхождение 0,16 ≈ m/2. Это артефакт сравнения, а не генератора.
   - **Ограничение:** при $m = 0{,}3$ кривая $m(\theta(t))$ инкрементального поворота почти линейна. Поэтому по доле смены метки инкрементальный переход почти неотличим от постепенного линейного.
   - Эти формы различаются механизмом. При gradual каждый объект порождён старым или новым концептом. При incremental объект порождён промежуточным концептом. Поэтому детекторы, сравнивающие распределения в окнах, могут реагировать на них по-разному. Это проверяется в Э3.
3. **Дисбаланс и шум** (Э1.4).
   - Доля класса воспроизводится с отклонением не более 1,5 SE.
   - Доля инверсий меток равна 0,1036 при $\eta = 0{,}1$ и 0,2052 при $\eta = 0{,}2$. Это в пределах 2 SE.
   - Доля инверсий одинакова при разных $\pi_0$, потому что шум берётся из отдельного подпотока ГСЧ. Это следствие разделения подпотоков, а не ошибка.
4. **Воспроизводимость** (Э1.5) выполнена для всех трёх групп семейств.
   - Поток побитово определяется парой (сценарий, seed).
   - Ground truth от seed реализации не зависит.
   - Сохранение и загрузка YAML дают тот же поток.
   - Изменение шума не меняет признаки.
5. **Классические потоки на единой шкале** (Э1.6, параметры river по умолчанию).
   - TV смены соседних вариантов:

     | генератор | TV |
     |---|---|
     | SEA | 0,086–0,203 |
     | Agrawal | 0,388–0,676 |
     | STAGGER | 0,486–0,778 |
     | Sine | 0,730–1,0 |
     | Mixed | 1,0 |

   - Из 22 смен лишь 4 (все у SEA) имеют TV < 0,25. При этом у SEA TV совпадает с $|\Delta P(y)|$: все изменённые метки меняются в одну сторону, поэтому каждая смена концепта SEA одновременно является сдвигом доли класса той же величины.
   - Значит, «смена функции» — внутренний параметр с неконтролируемой величиной от 0,09 до 1,0, несопоставимый между генераторами. Это количественно подтверждает пробел 1 из п. 1.5.
   - Область малых и средних величин чистого реального дрейфа ручные бенчмарки почти не покрывают.
6. **Реальные данные** (Э1.7).
   - Величина внедрённого дрейфа воспроизводится точно на эмпирической мере: для virtual и prior — до округления, для real — с точностью до массы одной строки ($|\tilde m - m| \le 0{,}0004$).
   - Побочное изменение $P(y)$ у real-событий достигает 0,040. Предупреждение W4 выдано для 3 из 6 real-событий.
   - У virtual-событий побочное изменение не превышает 0,014, и W4 не выдаётся.
   - Предупреждение W2 у всех virtual-событий ожидаемо: оно сообщает, что $P(y|X)$ не меняется.
7. **Производительность** (Э1.8). На порождение $10^5$ объектов уходит 0,05–0,2 с в зависимости от семейства и машины. Генерация не станет узким местом экспериментов. Поток порождается один раз на пару «сценарий, seed» и используется всеми детекторами, а прогон одного детектора в пилотном прогоне занимает в среднем 0,03–0,12 с, у KSWIN — около 6 с (п. 4.11).

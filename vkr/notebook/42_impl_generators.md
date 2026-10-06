### 4.3. Генерация потоков: функции перехода и семейство `hyperplane_gauss` [К]

#### 4.3.1. Функции перехода и временная разметка

Для каждого объекта $t$ временная разметка задаёт тройку $(src_t, dst_t, p_t)$: переход идёт из состояния цепочки концептов $src$ в состояние $dst$ на доле пути $p_t$. Функции перехода:
- **линейная**;
- **сигмоидная**, как в MOA. Она отнормирована так, что весь переход лежит внутри интервала ground truth $[\mathrm{onset}, \mathrm{end})$.
```writefile scendrift/generators/transitions.py
```
<!-- cell -->
#### 4.3.2. Общий конвейер и реестр генераторов

Общая часть для всех семейств:
1. проверка допустимости;
2. временная разметка;
3. шум меток;
4. ground truth;
5. упаковка результата.

Выборка объектов и шум используют **разные подпотоки ГСЧ**. Поэтому сценарии, отличающиеся только уровнем шума, порождают одни и те же признаки, и их результаты можно сравнивать попарно.
```writefile scendrift/generators/ground_truth.py
```
```writefile scendrift/generators/base.py
```
<!-- cell -->
#### 4.3.3. Калиброванное семейство `hyperplane_gauss`

Генератор берёт состояния $(w, \mu, \pi)$ из цепочки концептов (п. 4.1) и порождает объекты по схеме label shift:
- сначала метка;
- затем проекция на нормаль — из усечённого нормального распределения, через обратную функцию распределения, без отбора;
- затем ортогональная составляющая.

Инкрементальный переход интерполирует параметры **для каждого объекта**: угол поворота, сдвиг и долю класса, а для возврата — сферическую интерполяцию нормали. Всё векторизовано.
```writefile scendrift/generators/hyperplane.py
```
<!-- cell -->
### 4.4. Ground truth и диагностика генератора [К]

Ground truth строится по сценарию (п. 4.3.2). Модуль диагностики **независимо оценивает по порождённым данным** фактическую величину и форму перехода. На нём основан эксперимент Э1 (п. 5.1).
```writefile scendrift/generators/diagnostics.py
```
<!-- cell -->
### 4.5. Классические генераторы river и внедрение дрейфа в реальные данные [К]

#### 4.5.1. Классические генераторы (некалиброванные семейства `river:*`)

Классический сценарий — смена функции классификации генератора SEA, Agrawal, STAGGER, Sine или Mixed. Величину такой смены задать нельзя, но её можно измерить:
- при одном seed признаки $X$ у всех вариантов совпадают (проверка встроена в код);
- значит, $\mathrm{TV} = P(y_a \ne y_b)$ точно оценивается на общей эталонной выборке из 20 000 объектов.

Для этих семейств величина $m$ в сценарии не задаётся (ограничение C2). Фактическое значение попадает в ground truth.
```writefile scendrift/generators/classic.py
```
<!-- cell -->
#### 4.5.2. Внедрение дрейфа в реальные данные (семейства `real:bananas`, `real:phishing`)

Концепт — дискретное распределение на строках набора: веса строк и их метки. Поток порождается бутстрепом. Каждый вид дрейфа меняет ровно один множитель своего разложения $P(X, y)$ и **точно сохраняет** другой:

| Вид | Преобразование | Сохраняется точно |
|---|---|---|
| real | инверсия меток в полупространстве $\{u^\top z > c\}$ | $P(X)$ |
| virtual | экспоненциальный наклон весов строк $\propto e^{\lambda v^\top z}$ | $P(y \mid X)$ |
| prior | перевзвешивание классов | $P(X \mid y)$ |

Величина — TV совместных распределений на эмпирической мере, с группировкой строк-дубликатов (в Phishing дубликатов половина). Калибровка:
- real — точный перебор размера области (точность — масса одной строки);
- virtual — бисекция по $\lambda$;
- prior — аналитически.

На реальных данных события real и virtual могут попутно менять $P(y)$. Если изменение больше 0,02, выдаётся предупреждение W4.
```writefile scendrift/generators/real.py
```
```writefile scendrift/generators/__init__.py
```
<!-- cell -->
#### 4.5.3. Демонстрация: один сценарий — три семейства

Порождаем потоки трёх семейств и смотрим ground truth. Для классического семейства величина вычислена по эталонной выборке, для реальных данных — точно на эмпирической мере.
```python
import time

from scendrift.generators import generate

demo_specs = {
    "hyperplane_gauss": demo,
    "river:SEA": sio.from_dict({
        "stream": {"family": "river:SEA", "n_samples": 20_000, "n_features": 3},
        "drifts": {"schedule": {"count": 3}, "defaults": {"form": "gradual", "width": 1000}},
    }),
    "real:bananas": sio.from_dict({
        "stream": {"family": "real:bananas", "n_samples": 20_000, "n_features": 2,
                   "minority_share": 0.45},
        "events": [
            {"position": 4_000, "kind": "real", "magnitude": 0.2},
            {"position": 10_000, "kind": "virtual", "magnitude": 0.3, "form": "gradual",
             "width": 1_000},
            {"position": 16_000, "kind": "prior", "magnitude": 0.15, "form": "incremental",
             "width": 1_500},
        ],
    }),
}
gt_rows = []
for family, spec_f in demo_specs.items():
    start = time.perf_counter()
    stream_f = generate(spec_f)
    elapsed = time.perf_counter() - start
    print(f"{family:<17} X {stream_f.X.shape}, P(y=1) = {stream_f.y.mean():.3f}, "
          f"время {elapsed:.2f} с; {check(spec_f)}")
    for iv in stream_f.ground_truth.intervals:
        gt_rows.append({"семейство": family, "k": iv.index, "onset": iv.onset, "end": iv.end,
                        "форма": iv.form, "вид": iv.kind, "m задано": iv.magnitude,
                        **{f"m̃_{key}": round(val, 4) for key, val in iv.realized.items()}})
display(pd.DataFrame(gt_rows))
```

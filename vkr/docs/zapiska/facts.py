"""Числа и таблицы пояснительной записки — из результатов экспериментов и кода.

Ответственные: Будаев К. В. (генерация, формирование, Э1, Э7),
Гарифзянов Т. Р. (оценка, Э2–Э6, Э8).

Все значения вычисляются по файлам ``results/*.csv``, которые сохраняет
ноутбук, и по коду пакета ``scendrift``. Поэтому после прогона в режиме
FULL записка пересобирается без ручной правки чисел. Текстовые утверждения
главы 4 защищены проверкой: если в ``results/chapter5_claims.csv`` хотя бы
одно утверждение не выполнено, сборка останавливается — текст нужно
пересмотреть.

``collect()`` возвращает значения для ``{{ имя }}``, ``tables()`` — таблицы
Markdown для ``<!-- table:имя -->``.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from functools import cache
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NBSP = " "
RIVER = [
    "ADWIN",
    "KSWIN",
    "PageHinkley",
    "PageHinkley-up",
    "DDM",
    "EDDM",
    "HDDM_A",
    "HDDM_W",
    "FHDDM",
]
BASELINES = ["NoDrift", "Periodic(2000)", "Oracle"]
ONE_SIDED = ["PageHinkley-up", "DDM", "HDDM_A", "HDDM_W", "FHDDM"]
KIND_RU = {"real": "реальный", "virtual": "виртуальный", "prior": "априорный"}
PLAN_ORDER = ["сетка", "случайная выборка", "латинский гиперкуб", "Соболь"]
POLICY_RU = {"reject": "отклонение", "repair": "проекция", "adapt": "условные домены"}
PARAM_RU = {
    "n_samples": "n",
    "n_features": "d",
    "minority_share": "π₀",
    "label_noise": "η",
    "n_drifts": "K",
    "recurring": "возврат",
    "form": "форма",
    "width": "ℓ",
    "kind": "вид",
    "magnitude": "m",
    "affected_share": "α",
}
FACTOR_RU = {
    "magnitude": "величина m",
    "affected_share": "доля признаков α",
    "width": "ширина ℓ",
    "label_noise": "шум η",
    "minority_share": "доля класса π₀",
}


# ---------------------------------------------------------------------------
# форматирование
# ---------------------------------------------------------------------------


def num(x: float, nd: int = 2) -> str:
    """Число с десятичной запятой и знаком минус «−»."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    text = f"{x:.{nd}f}".replace(".", ",")
    if text.startswith("-"):
        text = "−" + text[1:]
        if set(text[1:]) <= set("0,"):
            text = text[1:]
    return text


def pct(x: float, nd: int | None = None) -> str:
    """Процент; без ``nd`` — целое число, если оно точное, иначе один знак."""
    if nd is None:
        nd = 0 if abs(100 * x - round(100 * x)) < 1e-9 else 1
    return f"{num(100 * x, nd)}{NBSP}%"


def pcell(p: float) -> str:
    """p-значение для ячейки таблицы: «< 0,001» или «0,006»."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    return f"<{NBSP}0,001" if p < 0.001 else num(p, 3 if p < 0.01 else 2)


def _plain_numbers(text: str) -> str:
    """«1,85e+03» → «1850» в текстах правил дерева решений."""

    def repl(m: re.Match) -> str:
        return num(float(m.group(0).replace(",", ".")), 0)

    return re.sub(r"\d+,\d+e[+-]\d+", repl, text)


def pval(p: float) -> str:
    """«p < 0,001» или «p = 0,18»."""
    if p < 0.001:
        return f"p{NBSP}<{NBSP}0,001"
    return f"p{NBSP}={NBSP}{num(p, 3 if p < 0.01 else 2)}"


def _esc(text: object) -> str:
    return str(text).replace("|", "\\|").replace("_", "\\_").replace("*", "\\*")


def md_table(header: list[str], rows: list[list[object]], widths: list[int]) -> str:
    """Pipe-таблица; ширины столбцов задаются числом дефисов (сумма ≥ 100)."""
    assert len(header) == len(widths)
    total = sum(widths)
    dashes = [max(3, round(100 * w / total)) for w in widths]
    lines = [
        "| " + " | ".join(h.replace("|", "\\|") for h in header) + " |",
        "|" + "|".join("-" * d for d in dashes) + "|",
    ]
    lines += ["| " + " | ".join(_esc(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def read(name: str, **kwargs) -> pd.DataFrame:
    """Таблица ``results/<name>.csv``."""
    return pd.read_csv(RESULTS / f"{name}.csv", **kwargs)


# ---------------------------------------------------------------------------
# данные
# ---------------------------------------------------------------------------


@cache
def _e2() -> pd.DataFrame:
    return read("e2_runs")


def _detector_order() -> list[str]:
    """Детекторы river по убыванию среднего F1 на наборе Э2, затем базовые линии."""
    means = _e2().groupby("detector")["f1"].mean()
    return list(means[RIVER].sort_values(ascending=False).index) + BASELINES


def _mode() -> str:
    return (RESULTS / "run_mode.txt").read_text(encoding="utf-8").strip()


def _manifest() -> dict:
    name = "e2_full" if _mode() == "FULL" else "e2_fast"
    return yaml.safe_load((RESULTS / "suites" / name / "manifest.yaml").read_text("utf-8"))


def _check_claims() -> int:
    claims = read("chapter5_claims")
    failed = claims.loc[~claims["выполнено"].astype(bool), "утверждение"].tolist()
    if failed:
        raise RuntimeError(
            "утверждения главы 5 не выполнены — пересмотрите текст записки:\n  "
            + "\n  ".join(failed)
        )
    return len(claims)


def _count_tests() -> int:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    match = re.search(r"(\d+) tests? collected", out)
    if match is not None:
        return int(match.group(1))
    per_file = [int(n) for n in re.findall(r"^tests/\S+: (\d+)$", out, re.M)]
    if not per_file:
        raise RuntimeError("не удалось определить число тестов:\n" + out[-500:])
    return sum(per_file)


# ---------------------------------------------------------------------------
# значения для {{ … }}
# ---------------------------------------------------------------------------


def collect() -> dict[str, object]:
    """Значения для подстановки ``{{ имя }}`` в текст записки."""
    import scendrift
    from scendrift.scenario.report import CONSTRAINTS
    from scendrift.scenario.space import DEFAULT_SPACE

    f: dict[str, object] = {}
    f["claims.n"] = _check_claims()
    mode = _mode()
    f["run.mode"] = mode
    f["run.full_note"] = (
        ""
        if mode == "FULL"
        else "[ЗАПУСТИТЬ FULL НА СЕРВЕРЕ и пересобрать записку: числа обновятся автоматически, "
        "утверждения выводов будут проверены повторно.]"
    )
    f["pkg.version"] = scendrift.__version__
    f["tests.count"] = _count_tests()
    f["model.constraints"] = sum(code.startswith("C") for code in CONSTRAINTS)
    f["model.warnings"] = sum(code.startswith("W") for code in CONSTRAINTS)
    f["space.free"] = len(DEFAULT_SPACE.free_names())
    modules = sorted((ROOT / "scendrift").rglob("*.py"))
    f["pkg.modules"] = len(modules)
    f["pkg.lines"] = sum(len(m.read_text(encoding="utf-8").splitlines()) for m in modules)

    # --- Э1 ----------------------------------------------------------------
    grid = read("e1_grid")
    grid_mean = grid.groupby(["вид", "m"])["m̂"].mean().reset_index()
    f["e1.grid_runs"] = len(grid)
    f["e1.grid_max_z"] = num(grid["z"].abs().max())
    f["e1.grid_sd_z"] = num(grid["z"].std())
    f["e1.grid_bias"] = num((grid_mean["m̂"] - grid_mean["m"]).abs().max(), 4)
    rand = read("e1_random")
    rs = read("e1_random_summary").set_index("вид")
    from scipy.stats import shapiro

    f["e1.rand_n"] = len(rand)
    per_kind = int(rand["вид"].value_counts().min())
    f["e1.rand_per_kind"] = per_kind
    f["e1.rand_rel_err"] = pct(1 / math.sqrt(2 * per_kind), 0)
    f["e1.rand_mean_min"] = num(rs["z_среднее"].min())
    f["e1.rand_mean_max"] = num(rs["z_среднее"].max())
    f["e1.rand_sd_min"] = num(rs["z_ст_откл"].min())
    f["e1.rand_sd_max"] = num(rs["z_ст_откл"].max())
    f["e1.shapiro_p"] = pval(shapiro(rand["z"]).pvalue)
    for kind in ("real", "virtual", "prior"):
        f[f"e1.err_{kind}"] = num(rs.loc[kind, "max_abs_ошибка"], 3)
    shape = read("e1_shape").set_index("переход")
    f["e1.shape_min"] = num(shape["max |эмп. − сглаж. теор.|"].min(), 3)
    f["e1.shape_max"] = num(shape["max |эмп. − сглаж. теор.|"].max(), 3)
    f["e1.shape_raw_sudden"] = num(shape.loc["внезапный", "max |эмп. − теор.|"])
    noise = read("e1_noise")
    noise["dev"] = (noise["P̂(y_чист=1)"] - noise["π₀"]).abs() / noise["SE(π)"]
    f["e1.prior_dev_se"] = num(noise["dev"].max(), 1)
    for eta in (0.1, 0.2):
        flips = noise.loc[noise["η"] == eta, "доля инверсий"]
        f[f"e1.flip_{int(eta * 10)}"] = num(flips.iloc[0], 4)
    classic = read("e1_classic")
    f["e1.classic_n"] = len(classic)
    f["e1.classic_small"] = int((classic["TV"] < 0.25).sum())
    f["e1.classic_min"] = num(classic["TV"].min())
    f["e1.classic_max"] = num(classic["TV"].max(), 1)
    real = read("e1_real")
    is_real = real["вид"] == "real"
    f["e1.real_err"] = num(real["|m̃ − m|"].max(), 4)
    f["e1.real_dpy"] = num(real.loc[is_real, "|ΔP(y)|"].max(), 3)
    f["e1.real_w4"] = int(real.loc[is_real, "предупреждения"].str.contains("W4").sum())
    f["e1.real_n"] = int(is_real.sum())
    f["e1.virtual_dpy"] = num(real.loc[real["вид"] == "virtual", "|ΔP(y)|"].max(), 3)

    # --- формирование наборов ------------------------------------------------
    plans = read("formation_plans").set_index("план")
    f["form.cd_sobol"] = num(plans.loc["Соболь", "CD-расхождение"], 3)
    f["form.cd_lhs"] = num(plans.loc["латинский гиперкуб", "CD-расхождение"], 3)
    f["form.cd_random"] = num(plans.loc["случайная выборка", "CD-расхождение"], 3)
    f["form.grid_points"] = int(plans.loc["сетка", "точек"])
    f["form.n"] = int(plans.loc["Соболь", "точек"])
    f["form.seeds"] = int(read("formation_comparison").query("план == 'Соболь'")["seed"].nunique())
    pol = read("formation_policies").set_index("способ")
    f["form.ks_noise"] = num(pol["порог КС"].iloc[0], 3)
    f["form.mc"] = int(pol["N_MC"].iloc[0])
    f["form.reject_accept"] = pct(pol.loc["reject", "доля принятых"])
    f["form.reject_ks_f"] = num(pol.loc["reject", "max КС: равномерное на F"], 3)
    f["form.repair_boundary"] = pct(pol.loc["repair", "доля на границе"])
    f["form.adapt_accept"] = pct(pol.loc["adapt", "доля принятых"], 1)
    f["form.adapt_ks_plan"] = num(pol.loc["adapt", "max КС: исходный план"], 3)
    f["form.real_plan"] = pct(pol.loc["repair", "доля real"])
    f["form.real_reject"] = pct(pol.loc["reject", "доля real"])
    f["form.prior_plan"] = pct(pol.loc["repair", "доля prior"])
    f["form.prior_reject"] = pct(pol.loc["reject", "доля prior"])
    cmp_ = read("formation_comparison").groupby(["план", "способ"])["классов"].mean()
    f["form.classes_sobol_adapt"] = num(cmp_[("Соболь", "adapt")], 1)
    f["form.classes_sobol_reject"] = num(cmp_[("Соболь", "reject")], 1)
    man = _manifest()
    f["suite.id"] = man["suite_id"]
    f["suite.name"] = man["suite"]
    f["suite.size"] = man["size"]
    f["suite.valid"] = man["formation"]["valid"]
    f["suite.adapted"] = man["formation"]["adapted"]
    f["suite.classes"] = len({s["signature"] for s in man["scenarios"]})

    # --- детекторы и протокол (п. 3.6) --------------------------------------
    dr = read("detector_direction").set_index("детектор")
    raw = dr.index[dr.index.str.startswith("FHDDM, вход")][0]
    f["dir.fhddm_fall"] = pct(dr.loc[raw, "падение 0,35 → 0,1"], 1)
    f["dir.fhddm_rise"] = pct(dr.loc[raw, "рост 0,1 → 0,35"], 1)
    f["dir.eddm_ctrl"] = pct(dr.loc["EDDM", "контроль 0,1"], 1)
    f["dir.hddmw_ctrl"] = pct(dr.loc["HDDM_W", "контроль 0,35"])
    f["dir.kswin_rise"] = pct(dr.loc["KSWIN", "рост 0,1 → 0,35"])
    pp = read("protocol_policy").set_index(["detector", "политика"])["fp_out"]
    for det, key in (("ADWIN", "adwin"), ("PageHinkley", "ph")):
        f[f"prot.{key}_reset"] = num(pp[(det, "reset")])
        f[f"prot.{key}_none"] = num(pp[(det, "none")])
    pilot = read("pilot_runs")
    oracle = pilot[pilot["detector"] == "Oracle"].groupby("kind")["delta_accuracy"].mean()
    for kind in ("real", "virtual", "prior"):
        f[f"prot.oracle_{kind}"] = ("+" if oracle[kind] > 0 else "") + num(oracle[kind], 3)
    f["pilot.scenarios"] = pilot["scenario_id"].nunique()
    runtime = pilot.groupby("detector")["runtime_s"].mean()
    f["pilot.kswin_s"] = num(runtime["KSWIN"], 1)
    others = runtime[[d for d in RIVER if d != "KSWIN"]]
    f["pilot.other_min"] = num(others.min())
    f["pilot.other_max"] = num(others.max())

    # --- Э2 ----------------------------------------------------------------
    e2 = _e2()
    means = e2.groupby("detector")[["f1", "precision", "recall", "far", "delta_accuracy"]].mean()
    f1r = means.loc[RIVER, "f1"]
    f["e2.scenarios"] = e2["scenario_id"].nunique()
    f["e2.repeats"] = e2.groupby(["scenario_id", "detector"]).size().max()
    f["e2.runs"] = len(e2)
    f["e2.detectors"] = e2["detector"].nunique()
    f["e2.events"] = int(e2.drop_duplicates("scenario_id")["K"].sum())
    f["e2.f1_min"] = num(f1r.min())
    f["e2.f1_min_det"] = f1r.idxmin()
    f["e2.f1_max"] = num(f1r.max())
    f["e2.f1_max_det"] = f1r.idxmax()
    f["e2.f1_second"] = f1r.sort_values(ascending=False).index[1]
    f["e2.f1_second_v"] = num(f1r.sort_values(ascending=False).iloc[1])
    f["e2.f1_third"] = f1r.sort_values(ascending=False).index[2]
    f["e2.f1_third_v"] = num(f1r.sort_values(ascending=False).iloc[2])
    f["e2.f1_periodic"] = num(means.loc["Periodic(2000)", "f1"])
    f["e2.prec_periodic"] = num(means.loc["Periodic(2000)", "precision"])
    f["e2.rec_periodic"] = num(means.loc["Periodic(2000)", "recall"])
    f["e2.far_best"] = means.loc[RIVER, "far"].idxmin()
    pol2 = read("e2_policy").set_index("detector")
    f["e2.ph_fp_reset"] = num(pol2.loc["PageHinkley", "reset:fp_out"], 1)
    f["e2.ph_fp_none"] = num(pol2.loc["PageHinkley", "none:fp_out"], 1)
    f["e2.eddm_fp_reset"] = num(pol2.loc["EDDM", "reset:fp_out"], 1)
    f["e2.eddm_fp_none"] = num(pol2.loc["EDDM", "none:fp_out"], 1)
    agree = read("rank_agreements", index_col=0)
    f["e2.tau_policy"] = num(agree.loc["reset/none (Э2)", "tau"])
    f["e2.tau_policy_p"] = pval(agree.loc["reset/none (Э2)", "p"])

    # --- Э3 ----------------------------------------------------------------
    gt = read("e3_grid_trend").set_index("детектор")
    f["e3.rho_m_min"] = num(gt["ρ(F1, m)"].min())
    f["e3.rho_m_max"] = num(gt["ρ(F1, m)"].max())
    f["e3.rho_w_max"] = num(gt["ρ(F1, ℓ)"].abs().max())
    sp = read("e3_spearman")
    mag = sp[sp["параметр"] == "magnitude"].set_index("detector")
    f["e3.e2_rho_m_min"] = num(mag["ρ"].min())
    f["e3.e2_rho_m_max"] = num(mag["ρ"].max())
    f["e3.e2_rho_m_sig"] = ", ".join(mag.index[mag["p (Холм)"] < 0.05]) or "ни у одного"
    blind = read("e3_virtual_blindness").set_index("detector")
    f["e3.virt_excess_max"] = num(blind["превышение virtual"].max())
    f["e3.real_excess_max"] = num(blind["превышение real"].max(), 1)
    f["e3.eddm_excess"] = num(blind.loc["EDDM", "превышение real"], 3)
    eta = read("e3_eta_mechanism").set_index("детектор")
    f["e3.eta_dets"] = ", ".join(eta.index)
    f["e3.eta_far_min"] = num(eta["ρ(η, far)"].min())
    f["e3.eta_excess_max"] = num(eta["ρ(η, превышение recall)"].abs().max())
    rules = read("e3_failure_rules")
    better = (rules["точность (CV)"] - rules["класс большинства"]) >= 0.05
    f["e3.rules_better"] = int(better.sum())
    f["e3.rules_not_real"] = int(rules["правило"].str.contains("вид").sum())

    # --- Э4 ----------------------------------------------------------------
    lam = read("e4_lambda_spearman").set_index("детектор").loc[RIVER]
    sig_neg = (lam["ρ(F1, λ)"] < 0) & (lam["p (Холм)"] < 0.05)
    f["e4.sig_neg"] = int(sig_neg.sum())
    f["e4.nonsig"] = ", ".join(lam.index[lam["p (Холм)"] >= 0.05])
    f["e4.mono_min"] = num(lam.loc[sig_neg, "доля невозрастающих шагов"].min(), 1)
    f["e4.mono_max"] = num(lam.loc[sig_neg, "доля невозрастающих шагов"].max(), 1)
    f["e4.ph_rho"] = num(lam.loc["PageHinkley", "ρ(F1, λ)"])
    f["e4.hddmw_f1_hard"] = num(lam.loc["HDDM_W", "F1 при λ = 1"])
    win = read("e4_windows").set_index("λ")
    f["e4.win_easy"] = pct(win.loc[0.0, "доля потока в окнах"])
    f["e4.win_hard"] = pct(win.loc[1.0, "доля потока в окнах"])
    f["e4.width_hard"] = int(win.loc[1.0, "ширина ℓ"])
    ofat = read("e4_ofat_summary").set_index("фактор")
    f["e4.ofat_dets"] = int(ofat.loc["magnitude", "детекторов"])
    f["e4.ofat_mag_sig"] = int(ofat.loc["magnitude", "значимо_отрицательных"])

    # --- Э5 ----------------------------------------------------------------
    fr = read("e5_friedman").set_index("метрика")
    f["e5.cd"] = num(fr.loc["F1", "CD"])
    f["e5.mtd_blocks"] = int(fr.loc["MTD", "блоков"])
    pairs = read("e5_wilcoxon_f1")
    f["e5.pairs"] = len(pairs)
    f["e5.pairs_diff"] = int((pairs["p (Холм)"] < 0.05).sum())
    vp = read("e5_vs_periodic")
    f1p = vp[vp["метрика"] == "f1"]
    f["e5.worse_periodic"] = int((f1p["вывод"] == "хуже Periodic").sum())
    f["e5.not_worse"] = ", ".join(f1p.loc[f1p["вывод"] != "хуже Periodic", "детектор"])
    accp = vp[vp["метрика"] == "delta_accuracy"]
    f["e5.acc_better"] = int((accp["вывод"] == "лучше Periodic").sum())

    # --- Э6 ----------------------------------------------------------------
    e6 = read("e6_f1").set_index("detector").loc[RIVER]
    f["e6.f1_min"] = num(e6["F1 (ручной)"].min())
    f["e6.f1_max"] = num(e6["F1 (ручной)"].max())
    f["e6.tau"] = num(agree.loc["ручной/Э2 (Э6)", "tau"])
    f["e6.p"] = pval(agree.loc["ручной/Э2 (Э6)", "p"])
    cov = read("e6_coverage", index_col=0)
    f["e6.classes_manual"] = int(cov.iloc[0]["классов таксономии"])
    f["e6.classes_auto"] = int(cov.iloc[1]["классов таксономии"])
    f["e6.median_manual"] = num(cov.iloc[0]["медиана TV"])
    f["e6.median_auto"] = num(cov.iloc[1]["медиана TV"])
    f["e6.small_manual"] = pct(cov.iloc[0]["доля TV < 0,1"])
    f["e6.small_auto"] = pct(cov.iloc[1]["доля TV < 0,1"])

    # --- Э7 ----------------------------------------------------------------
    e7r = read("e7_recall").set_index("detector")
    gap = {k: (e7r[f"реальные:{k}"] - e7r[f"синтетика:{k}"]).mean() for k in KIND_RU}
    f["e7.gap_virtual"] = num(gap["virtual"])
    f["e7.gap_real"] = num(gap["real"])
    f["e7.gap_prior"] = num(gap["prior"])
    f["e7.tau"] = num(agree.loc["реальные/синтетика (Э7)", "tau"])
    f["e7.p"] = pval(agree.loc["реальные/синтетика (Э7)", "p"])
    e7 = read("e7_runs", usecols=["scenario_id", "источник", "W4"]).drop_duplicates("scenario_id")
    f["e7.scenarios"] = len(e7)
    f["e7.w4"] = int(e7["W4"].sum())

    # --- Э8 ----------------------------------------------------------------
    f["e8.tau"] = num(agree.loc["F1/Δaccuracy (Э8)", "tau"])
    acc = read("e8_accuracy").set_index("detector")["delta_accuracy:mean"]
    best = acc[RIVER].sort_values(ascending=False)
    f["e8.best"] = (
        f"{best.index[0]} и {best.index[1]}"
        if round(best.iloc[0], 3) == round(best.iloc[1], 3)
        else best.index[0]
    )
    f["e8.best_v"] = num(best.iloc[0], 3)
    f["e8.periodic"] = num(acc["Periodic(2000)"], 3)
    f["e8.ph"] = num(acc["PageHinkley"], 3)
    corr = read("e8_corr").set_index("детектор")
    f["e8.corr_sig"] = ", ".join(corr.index[(corr["p (Холм)"] < 0.05) & (corr["ρ(F1, Δacc)"] > 0)])
    return f


# ---------------------------------------------------------------------------
# таблицы для <!-- table:… -->
# ---------------------------------------------------------------------------


def _ks(row: pd.Series, against: str) -> str:
    """Наибольшая статистика КС и параметр, на котором она достигается."""
    return f"{num(row[f'max КС: {against}'], 3)} ({PARAM_RU[row[f'где ({against})']]})"


def tables() -> dict[str, str]:
    """Таблицы Markdown для ``<!-- table:имя -->``."""
    from scendrift.scenario.report import CONSTRAINTS
    from scendrift.scenario.space import DEFAULT_SPACE

    t: dict[str, str] = {}
    order = _detector_order()

    t["constraints"] = md_table(
        ["Код", "Уровень", "Условие"],
        [[code, scope, text] for code, (scope, text) in CONSTRAINTS.items()],
        [10, 18, 72],
    )
    t["space"] = md_table(
        ["Параметр", "Обозн.", "Домен", "По умолч.", "Активен, если"],
        [
            [r["смысл"], r["обозначение"], r["домен"], r["по умолчанию"], r["активен, если"]]
            for r in DEFAULT_SPACE.table()
        ],
        [34, 9, 22, 12, 23],
    )

    rs = read("e1_random_summary").set_index("вид")
    t["e1_random"] = md_table(
        ["Вид дрейфа", "Сценариев", "Среднее z", "Ст. откл. z", "ДИ для ст. откл.", "Max |m̂ − m|"],
        [
            [
                KIND_RU[k],
                int(rs.loc[k, "сценариев"]),
                num(rs.loc[k, "z_среднее"]),
                num(rs.loc[k, "z_ст_откл"]),
                f"[{num(rs.loc[k, 'ДИ_нижн'])}; {num(rs.loc[k, 'ДИ_верх'])}]",
                num(rs.loc[k, "max_abs_ошибка"], 3),
            ]
            for k in ("real", "virtual", "prior")
        ],
        [18, 14, 14, 14, 22, 18],
    )

    classic = read("e1_classic")
    rows = []
    for gen, part in classic.groupby("генератор", sort=False):
        rows.append(
            [
                gen,
                len(part),
                f"{num(part['TV'].min())} – {num(part['TV'].max())}",
                f"{num(part['|ΔP(y)|'].min())} – {num(part['|ΔP(y)|'].max())}",
            ]
        )
    t["e1_classic"] = md_table(
        ["Генератор", "Смен концепта", "TV смены", "|ΔP(y)|"], rows, [25, 20, 27, 28]
    )

    plans = read("formation_plans").set_index("план").loc[PLAN_ORDER]
    t["plans"] = md_table(
        ["План", "Точек", "CD-расхождение", "Maximin", "Доля слоёв"],
        [
            [
                name,
                int(r["точек"]),
                num(r["CD-расхождение"], 4),
                num(r["maximin"], 3),
                num(r["доля слоёв"], 3),
            ]
            for name, r in plans.iterrows()
        ],
        [30, 12, 22, 16, 20],
    )

    pol = read("formation_policies").set_index("способ")
    t["policies"] = md_table(
        ["Способ", "Принято", "На границе", "Доля real", "КС против F", "КС против плана"],
        [
            [
                POLICY_RU[p],
                pct(r["доля принятых"], 1),
                pct(r["доля на границе"], 1),
                pct(r["доля real"]),
                _ks(r, "равномерное на F"),
                _ks(r, "исходный план"),
            ]
            for p, r in pol.iterrows()
        ],
        [20, 13, 14, 13, 20, 20],
    )

    cmp_ = (
        read("formation_comparison")
        .groupby(["план", "способ"], sort=False)[["доля принятых", "доля на границе", "классов"]]
        .agg(["mean", "std"])
    )
    rows = []
    for plan in PLAN_ORDER:
        for policy in ("reject", "repair", "adapt"):
            r = cmp_.loc[(plan, policy)]
            sd = r[("классов", "std")]
            classes = num(r[("классов", "mean")], 1) + ("" if pd.isna(sd) else f" ± {num(sd, 1)}")
            rows.append(
                [
                    plan,
                    POLICY_RU[policy],
                    pct(r[("доля принятых", "mean")], 1),
                    pct(r[("доля на границе", "mean")], 1),
                    classes,
                ]
            )
    t["plan_policy"] = md_table(
        ["План", "Способ", "Принято", "На границе", "Классов таксономии"],
        rows,
        [26, 22, 15, 16, 21],
    )

    dr = read("detector_direction")
    t["direction"] = md_table(
        [
            "Детектор",
            "Реагирует на",
            "Рост 0,1→0,35",
            "Контроль 0,1",
            "Падение 0,35→0,1",
            "Контроль 0,35",
        ],
        [
            [
                r["детектор"].replace(" (как в описании river)", ""),
                r["в реестре"],
                pct(r["рост 0,1 → 0,35"]),
                pct(r["контроль 0,1"]),
                pct(r["падение 0,35 → 0,1"]),
                pct(r["контроль 0,35"]),
            ]
            for _, r in dr.iterrows()
        ],
        [24, 15, 15, 15, 16, 15],
    )

    e2 = _e2()
    g = e2.groupby("detector")
    means = g[["f1", "precision", "recall", "mtd", "far", "delta_accuracy"]].mean()
    sds = g["f1"].std()
    t["e2"] = md_table(
        ["Метод", "F1", "Precision", "Recall", "MTD", "FAR", "Δacc"],
        [
            [
                d,
                f"{num(means.loc[d, 'f1'])} ± {num(sds[d])}",
                num(means.loc[d, "precision"]),
                num(means.loc[d, "recall"]),
                num(means.loc[d, "mtd"], 0),
                num(means.loc[d, "far"]),
                num(means.loc[d, "delta_accuracy"], 3),
            ]
            for d in order
        ],
        [22, 16, 13, 12, 11, 11, 13],
    )

    pol2 = read("e2_policy").set_index("detector")
    t["e2_policy"] = md_table(
        ["Детектор", "F1 (reset)", "F1 (none)", "Ложных тревог (reset)", "Ложных тревог (none)"],
        [
            [
                d,
                num(pol2.loc[d, "reset:f1"]),
                num(pol2.loc[d, "none:f1"]),
                num(pol2.loc[d, "reset:fp_out"]),
                num(pol2.loc[d, "none:fp_out"]),
            ]
            for d in order
            if d in pol2.index and d in RIVER
        ],
        [24, 16, 16, 22, 22],
    )

    gt = read("e3_grid_trend").set_index("детектор")
    t["e3_grid"] = md_table(
        ["Детектор", "ρ(F1, m)", "ρ(F1, ℓ)", "F1 при m = 0,03", "F1 при m = 0,4"],
        [
            [
                d,
                num(gt.loc[d, "ρ(F1, m)"]),
                num(gt.loc[d, "ρ(F1, ℓ)"]),
                num(gt.loc[d, "F1 при m = 0,03"]),
                num(gt.loc[d, "F1 при m = 0,4"]),
            ]
            for d in order
            if d in gt.index
        ],
        [26, 17, 17, 20, 20],
    )

    bl = read("e3_virtual_blindness").set_index("detector")
    t["e3_blind"] = md_table(
        [
            "Детектор",
            "R (real)",
            "R₀ (real)",
            "R − R₀ (real)",
            "R (virtual)",
            "R₀ (virtual)",
            "R − R₀ (virtual)",
        ],
        [
            [
                d,
                num(bl.loc[d, "recall (real)"]),
                num(bl.loc[d, "случайный recall (real)"]),
                num(bl.loc[d, "превышение real"]),
                num(bl.loc[d, "recall (virtual)"]),
                num(bl.loc[d, "случайный recall (virtual)"]),
                num(bl.loc[d, "превышение virtual"]),
            ]
            for d in order
            if d in bl.index
        ],
        [19, 13, 13, 14, 13, 14, 14],
    )

    rules = read("e3_failure_rules").set_index("детектор")
    t["e3_rules"] = md_table(
        [
            "Детектор",
            "Доля отказов",
            "Правило (лист с наибольшим числом отказов)",
            "Точность (CV)",
            "Класс большинства",
        ],
        [
            [
                d,
                num(rules.loc[d, "доля отказов"]),
                _plain_numbers(rules.loc[d, "правило"]),
                num(rules.loc[d, "точность (CV)"]),
                num(rules.loc[d, "класс большинства"]),
            ]
            for d in order
            if d in rules.index
        ],
        [19, 12, 33, 15, 21],
    )

    lam = read("e4_lambda_spearman").set_index("детектор")
    t["e4_lambda"] = md_table(
        [
            "Детектор",
            "ρ(F1, λ)",
            "p (Холм)",
            "Невозрастающих шагов",
            "F1 при λ = 0",
            "F1 при λ = 1",
        ],
        [
            [
                d,
                num(lam.loc[d, "ρ(F1, λ)"]),
                pcell(lam.loc[d, "p (Холм)"]),
                num(lam.loc[d, "доля невозрастающих шагов"], 1),
                num(lam.loc[d, "F1 при λ = 0"]),
                num(lam.loc[d, "F1 при λ = 1"]),
            ]
            for d in order
            if d in RIVER
        ],
        [21, 12, 13, 22, 16, 16],
    )

    win = read("e4_windows")
    t["e4_windows"] = md_table(
        ["λ", "Ширина ℓ", "Доля потока в окнах допуска", "Recall (river)", "Случайный recall"],
        [
            [
                num(r["λ"], 1),
                int(r["ширина ℓ"]),
                pct(r["доля потока в окнах"], 1),
                num(r["recall (детекторы river)"]),
                num(r["случайный recall"]),
            ]
            for _, r in win.iterrows()
        ],
        [10, 15, 30, 20, 25],
    )

    ofat = read("e4_ofat_summary").set_index("фактор")
    t["e4_ofat"] = md_table(
        ["Фактор", "Средний ρ(F1, λⱼ)", "Значимо отрицательных", "Детекторов"],
        [
            [
                FACTOR_RU[k],
                num(r["средний_ρ"]),
                int(r["значимо_отрицательных"]),
                int(r["детекторов"]),
            ]
            for k, r in ofat.iterrows()
        ],
        [34, 22, 24, 20],
    )

    fr = read("e5_friedman")
    t["e5"] = md_table(
        ["Метрика", "Блоков", "χ²F", "p", "FF", "CD", "Лучший по рангу", "Худший по рангу"],
        [
            [
                r["метрика"],
                int(r["блоков"]),
                num(r["χ²_F"], 1),
                pcell(r["p"]),
                num(r["F_F"], 1),
                num(r["CD"]),
                r["лучший по рангу"],
                r["худший по рангу"],
            ]
            for _, r in fr.iterrows()
        ],
        [15, 10, 9, 10, 8, 8, 20, 20],
    )

    vp = read("e5_vs_periodic")
    rows = []
    for d in [x for x in order if x in RIVER]:
        a = vp[(vp["детектор"] == d) & (vp["метрика"] == "f1")].iloc[0]
        b = vp[(vp["детектор"] == d) & (vp["метрика"] == "delta_accuracy")].iloc[0]
        rows.append(
            [
                d,
                num(a["медиана разности"]),
                pcell(a["p (Холм)"]),
                a["вывод"].replace(" Periodic", ""),
                num(b["медиана разности"], 3),
                pcell(b["p (Холм)"]),
                b["вывод"].replace(" Periodic", ""),
            ]
        )
    t["e5_periodic"] = md_table(
        [
            "Детектор",
            "F1: медиана разности",
            "p (Холм)",
            "Вывод",
            "Δacc: медиана разности",
            "p (Холм)",
            "Вывод",
        ],
        rows,
        [19, 15, 12, 14, 15, 12, 13],
    )

    e6 = read("e6_f1").set_index("detector")
    t["e6"] = md_table(
        ["Метод", "F1 (ручной набор)", "F1 (набор Э2)", "Ранг (ручной)", "Ранг (Э2)"],
        [
            [
                d,
                num(e6.loc[d, "F1 (ручной)"]),
                num(e6.loc[d, "F1 (Э2)"]),
                num(e6.loc[d, "ранг (ручной)"]),
                num(e6.loc[d, "ранг (Э2)"]),
            ]
            for d in order
        ],
        [24, 20, 20, 18, 18],
    )
    cov = read("e6_coverage", index_col=0)
    t["e6_coverage"] = md_table(
        ["Набор", "Событий", "TV < 0,1", "TV < 0,25", "Медиана TV", "Классов", "Видов"],
        [
            [
                name.replace(" (e2_fast)", ""),
                int(r["событий"]),
                pct(r["доля TV < 0,1"]),
                pct(r["доля TV < 0,25"]),
                num(r["медиана TV"]),
                int(r["классов таксономии"]),
                int(r["видов дрейфа"]),
            ]
            for name, r in cov.iterrows()
        ],
        [22, 14, 12, 13, 14, 13, 12],
    )

    e7r = read("e7_recall").set_index("detector")
    t["e7"] = md_table(
        ["Детектор", "real", "virtual", "prior", "real", "virtual", "prior"],
        [
            [
                d,
                *(
                    num(e7r.loc[d, f"{src}:{k}"])
                    for src in ("реальные", "синтетика")
                    for k in ("real", "virtual", "prior")
                ),
            ]
            for d in order
            if d in e7r.index
        ],
        [19, 13, 14, 13, 13, 14, 14],
    )

    acc = read("e8_accuracy").set_index("detector")
    corr = read("e8_corr").set_index("детектор")
    rows = []
    for d in order:
        rho = corr.loc[d, "ρ(F1, Δacc)"] if d in corr.index else float("nan")
        p = corr.loc[d, "p (Холм)"] if d in corr.index else float("nan")
        rows.append(
            [
                d,
                f"{num(acc.loc[d, 'delta_accuracy:mean'], 3)} ± "
                f"{num(acc.loc[d, 'delta_accuracy:std'], 3)}",
                num(acc.loc[d, "oracle_gap (real):mean"], 3),
                num(rho),
                pcell(p),
            ]
        )
    t["e8"] = md_table(
        [
            "Метод",
            "Δacc (среднее ± ст. откл.)",
            "Разрыв до Oracle (real)",
            "ρ(F1, Δacc)",
            "p (Холм)",
        ],
        rows,
        [22, 26, 20, 16, 16],
    )
    return t


if __name__ == "__main__":  # отладка: python docs/zapiska/facts.py
    for key, value in collect().items():
        print(f"{key:<24} {value}")
    for key, value in tables().items():
        print(f"\n[{key}]\n{value}")

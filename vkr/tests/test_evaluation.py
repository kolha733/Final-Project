"""Тесты оценки: базовая модель, детекторы, раннер, метрики, статистика, отчёты."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from river import naive_bayes

from scendrift.detectors import DETECTORS, make_detector
from scendrift.evaluation import runner as R
from scendrift.evaluation.metrics import detection_metrics, match_detections
from scendrift.evaluation.models import PrequentialGaussianNB
from scendrift.evaluation.protocol import EvaluationSpec
from scendrift.evaluation.results import runs_to_frame
from scendrift.evaluation.stats import (
    average_ranks,
    friedman,
    holm,
    nemenyi_cd,
    nemenyi_cliques,
    wilcoxon_holm,
)
from scendrift.generators import generate
from scendrift.interfaces import DriftInterval, GroundTruth
from scendrift.reporting.report import format_table, html_report, summary_table
from scendrift.scenario import io

# --- базовая модель ---------------------------------------------------------


def _river_errors(X: np.ndarray, y: np.ndarray, resets: set[int]) -> np.ndarray:
    out, model = [], naive_bayes.GaussianNB()
    for t in range(len(y)):
        if t in resets:
            model = naive_bayes.GaussianNB()
        x = dict(enumerate(X[t].tolist()))
        out.append(model.predict_one(x) != int(y[t]))
        model.learn_one(x, int(y[t]))
    return np.array(out)


def _our_errors(X: np.ndarray, y: np.ndarray, resets: set[int], block: int) -> np.ndarray:
    nb = PrequentialGaussianNB(X, y)
    points = sorted({0, *resets, len(y)})
    out = []
    for a, b in zip(points[:-1], points[1:], strict=True):
        state, pos = nb.empty_state(), a
        while pos < b:
            stop = min(b, pos + block)
            err, state = nb.block(pos, stop, state)
            out.append(err)
            pos = stop
    return np.concatenate(out)


@pytest.mark.parametrize(
    "stream",
    [
        {"n_samples": 3_500, "n_features": 4, "minority_share": 0.3, "label_noise": 0.1},
        {"family": "real:phishing", "n_samples": 3_500, "n_features": 9, "minority_share": 0.45},
        {"family": "river:SEA", "n_samples": 3_500, "n_features": 3},
    ],
)
def test_vectorized_nb_matches_river(stream: dict) -> None:
    spec = io.from_dict({"stream": stream, "events": []})
    data = generate(spec)
    resets = {1, 2, 500, 1_337}
    expected = _river_errors(data.X, data.y, resets)
    for block in (97, 2_048):
        assert np.array_equal(_our_errors(data.X, data.y, resets, block), expected)


# --- детекторы -------------------------------------------------------------


def _stream(a: float, b: float, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return np.r_[rng.random(3_000) < a, rng.random(1_000) < b].astype(float).tolist()


def _fires_after_change(name: str, a: float, b: float) -> int:
    hits = 0
    for seed in range(5):
        det = make_detector(name)
        hits += any(det.update(v) and t >= 3_000 for t, v in enumerate(_stream(a, b, seed)))
    return hits


@pytest.mark.parametrize("name", ["DDM", "HDDM_A", "HDDM_W", "FHDDM", "PageHinkley-up"])
def test_one_sided_detectors_react_to_error_increase_only(name: str) -> None:
    assert _fires_after_change(name, 0.1, 0.35) == 5
    assert _fires_after_change(name, 0.35, 0.1) == 0


def test_two_sided_detectors_react_to_decrease() -> None:
    for name in ("ADWIN", "PageHinkley"):
        assert DETECTORS[name].two_sided and _fires_after_change(name, 0.35, 0.1) == 5


def test_detector_names_clone_and_reset() -> None:
    det = make_detector("ADWIN", delta=0.01)
    assert det.name == "ADWIN(delta=0.01)" and make_detector("ADWIN").name == "ADWIN"
    clone = det.clone()
    for v in _stream(0.1, 0.4, 0):
        det.update(v)
    assert clone._inner.width == 0 and det._inner.width > 0  # type: ignore[attr-defined]
    det.reset()
    assert det._inner.width == 0  # type: ignore[attr-defined]
    with pytest.raises(KeyError):
        make_detector("NoSuchDetector")


def test_baselines() -> None:
    periodic = make_detector("Periodic", period=100)
    fires = [t for t in range(450) if periodic.update(0.0) and (periodic.reset() or True)]
    assert fires == [99, 199, 299, 399]
    assert not any(make_detector("NoDrift").update(1.0) for _ in range(100))
    gt = GroundTruth((DriftInterval(0, 400, 500, 600, "gradual", "real", 0.2),), 2_000)
    oracle = make_detector("Oracle").bind(gt, start=100)  # type: ignore[attr-defined]
    assert [t for t in range(100, 2_000) if oracle.update(0.0)] == [500]


# --- раннер ----------------------------------------------------------------


@pytest.fixture(scope="module")
def easy_spec():
    return io.from_dict({"extends": "catalog/easy"})


def test_runner_protocol(easy_spec) -> None:
    data = generate(easy_spec)
    res = R.run_stream(data, easy_spec.evaluation, make_detector("PageHinkley-up"))
    assert all(t >= easy_spec.evaluation.warmup for t in res["detections"])
    assert res["n_resets"] == len(res["detections"]) and 0.5 < res["accuracy"] <= 1.0
    no_reset = easy_spec.evaluation.model_copy(update={"on_detection": "none"})
    as_string = easy_spec.evaluation.model_copy(update={"on_detection": "reset"})  # строка, не enum
    res_str = R.run_stream(data, as_string, make_detector("PageHinkley-up"))
    assert res_str["n_resets"] == len(res_str["detections"]) > 0
    res_none = R.run_stream(data, no_reset, make_detector("Periodic", period=1_000))
    assert res_none["n_resets"] == 0 and len(res_none["detections"]) > 0
    base = R.run_stream(data, easy_spec.evaluation, make_detector("NoDrift"))
    nb = PrequentialGaussianNB(data.X, data.y)
    expected = 1 - nb.errors()[easy_spec.evaluation.warmup :].mean()
    assert base["accuracy"] == pytest.approx(expected)


def test_benchmark_cache_and_parallel(easy_spec, tmp_path) -> None:
    dets = [make_detector("HDDM_A"), make_detector("NoDrift"), make_detector("Oracle")]
    first = R.run_benchmark([easy_spec], dets, repeats=2, cache_dir=tmp_path)
    again = R.run_benchmark([easy_spec], dets, repeats=2, cache_dir=tmp_path)
    parallel = R.run_benchmark([easy_spec], dets, repeats=2, n_jobs=2)
    assert first == again
    assert [r.detections for r in parallel] == [r.detections for r in first]
    assert [r.seed for r in first] == [0, 0, 0, 1, 1, 1]
    frame = runs_to_frame(first, [easy_spec])
    nodrift, oracle = frame[frame.detector == "NoDrift"], frame[frame.detector == "Oracle"]
    assert (nodrift["delta_accuracy"] == 0).all() and (oracle["oracle_gap"] == 0).all()
    assert (oracle["f1"] == 1.0).all()


# --- метрики ---------------------------------------------------------------


def _gt() -> GroundTruth:
    return GroundTruth(
        (
            DriftInterval(0, 1_000, 1_000, 1_000, "sudden", "real", 0.2),
            DriftInterval(1, 3_000, 3_100, 3_200, "gradual", "real", 0.2),
            DriftInterval(2, 6_000, 6_000, 6_000, "sudden", "real", 0.2),
        ),
        8_000,
    )


def test_matching_and_metrics_worked_example() -> None:
    protocol = EvaluationSpec(warmup=500, acceptance_window=500)
    detections = [100, 700, 1_049, 1_200, 3_050, 4_000, 4_100]
    m = match_detections(detections, _gt(), protocol)
    assert m.detections == (700, 1_049, 1_200, 3_050, 4_000, 4_100)  # 100 < W не учитывается
    assert m.hits == (1_049, 3_050, None)
    assert m.delays == (50, 51, None)
    assert m.redundant == (1_200,) and m.false_alarms == (700, 4_000, 4_100)
    assert m.t_stable == (8_000 - 500) - (500 + 700 + 500)
    met = detection_metrics(detections, _gt(), protocol)
    assert met["precision"] == pytest.approx(2 / 6) and met["recall"] == pytest.approx(2 / 3)
    assert met["f1"] == pytest.approx(2 * (1 / 3) * (2 / 3) / (1 / 3 + 2 / 3))
    assert met["mtd"] == 50.5 and met["mdr"] == pytest.approx(1 / 3)
    assert met["mtfa"] == pytest.approx(5_800 / 3)
    assert met["mtr"] == pytest.approx(5_800 / 3 / 50.5 * (2 / 3))
    assert met["far"] == pytest.approx(1_000 * 3 / 5_800)


def test_metric_edge_cases() -> None:
    protocol = EvaluationSpec(warmup=500, acceptance_window=500)
    none = detection_metrics([], _gt(), protocol)
    assert math.isnan(none["precision"]) and none["f1"] == 0.0 and math.isnan(none["mtd"])
    assert none["mtfa"] == math.inf and none["mtfa_capped"] == none["t_stable"]
    no_events = detection_metrics([900], GroundTruth((), 8_000), protocol)
    assert math.isnan(no_events["recall"]) and math.isnan(no_events["f1"])
    center = EvaluationSpec(warmup=500, acceptance_window=500, delay_reference="center")
    m = match_detections([3_050], _gt(), center)
    assert m.delays[1] == -50 and math.isnan(detection_metrics([3_050], _gt(), center)["mtr"])


# --- статистика ------------------------------------------------------------


def test_nemenyi_matches_demsar_table() -> None:
    table = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    }  # Demšar (2006), табл. 5(a), α = 0,05
    for k, q in table.items():
        assert nemenyi_cd(k, 10)[1] == pytest.approx(q, abs=1e-3)


def test_ranks_friedman_and_wilcoxon() -> None:
    rng = np.random.default_rng(0)
    base = rng.random((30, 1))
    matrix = pd.DataFrame(np.hstack([base + 0.3, base + 0.15, base]), columns=["a", "b", "c"])
    matrix += rng.normal(0, 0.01, matrix.shape)
    ranks = average_ranks(matrix)
    assert list(ranks.index) == ["a", "b", "c"] and ranks["a"] == 1.0
    res = friedman(matrix)
    assert res.p_chi2 < 1e-6 and res.p_f < 1e-6
    table = wilcoxon_holm(matrix)
    assert (table["различие"] == "A лучше").all()
    ties = average_ranks(pd.DataFrame({"x": [1.0, 1.0], "y": [1.0, 0.0]}))
    assert ties["x"] == 1.25 and ties["y"] == 1.75
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    cliques = nemenyi_cliques(pd.Series({"a": 1.0, "b": 1.5, "c": 3.0}), cd=1.0)
    assert cliques == [("a", "b")]


# --- отчёты ----------------------------------------------------------------


def test_report_outputs(tmp_path) -> None:
    frame = pd.DataFrame({"detector": ["a", "a", "b"], "f1": [0.5, 0.7, 0.2]})
    summary = summary_table(frame, ["f1"])
    formatted = format_table(summary, digits=2)
    assert formatted.loc["a", "f1"] == "0,60 ± 0,14" and formatted.loc["b", "f1"] == "0,20"
    path = html_report(
        "Отчёт", [("Сводка", formatted), ("Комментарий", "текст")], tmp_path / "r.html"
    )
    text = path.read_text(encoding="utf-8")
    assert "<table" in text and "0,60 ± 0,14" in text and "текст" in text

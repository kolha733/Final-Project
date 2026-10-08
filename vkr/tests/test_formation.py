"""Тесты формирования сценариев: планы, параметризация, ограничения, шаблоны, наборы."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scendrift.formation import compose as comp
from scendrift.formation.difficulty import DIFFICULTY, PROFILES
from scendrift.formation.mapping import (
    magnitude_on_boundary,
    point_to_spec,
    realize,
    spec_to_point,
    summarize,
)
from scendrift.formation.samplers import (
    GridSampler,
    LHSSampler,
    SobolSampler,
    design_metrics,
    grid_levels,
    make_sampler,
)
from scendrift.formation.suite import (
    SuiteConfig,
    SuiteIntegrityError,
    build_space,
    build_suite,
    load_suite,
)
from scendrift.formation.templates import TemplateError, deep_merge, resolve
from scendrift.scenario import io
from scendrift.scenario.space import DEFAULT_SPACE, IntDomain
from scendrift.scenario.validation import check

D = len(DEFAULT_SPACE.free_names())

# --- обратное отображение и планы ------------------------------------------


def test_to_unit_inverts_from_unit() -> None:
    rng = np.random.default_rng(0)
    for _ in range(300):
        u = rng.random(D)
        point = DEFAULT_SPACE.from_unit(u)
        again = DEFAULT_SPACE.from_unit(DEFAULT_SPACE.to_unit(point, u))
        for key, value in point.items():
            if isinstance(value, float):
                assert math.isclose(value, again[key], rel_tol=1e-12)
            else:
                assert value == again[key]
    dom = IntDomain(100, 4_000, log=True)
    assert all(dom.from_unit(dom.to_unit(v)) == v for v in range(100, 4_001, 7))


@pytest.mark.parametrize("method", ["random", "lhs", "sobol"])
def test_designs_shape_and_reproducibility(method: str) -> None:
    a = make_sampler(method).design(DEFAULT_SPACE, 64, seed=3)
    b = make_sampler(method).design(DEFAULT_SPACE, 64, seed=3)
    c = make_sampler(method).design(DEFAULT_SPACE, 64, seed=4)
    assert a.shape == (64, D) and np.all((a >= 0) & (a < 1))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_lhs_and_sobol_stratify_every_axis() -> None:
    for sampler in (LHSSampler(), SobolSampler()):
        assert design_metrics(sampler.design(DEFAULT_SPACE, 64, seed=1))["strata"] == 1.0


def test_sobol_beats_random_in_discrepancy() -> None:
    cd = {
        m: np.mean(
            [design_metrics(make_sampler(m).design(DEFAULT_SPACE, 128, s))["cd"] for s in range(5)]
        )
        for m in ("random", "sobol")
    }
    assert cd["sobol"] < cd["random"]


def test_grid_respects_budget() -> None:
    levels = grid_levels(DEFAULT_SPACE, 128)
    assert math.prod(levels.values()) <= 128
    assert levels["form"] == 3 and levels["magnitude"] == 1  # проклятие размерности
    small = DEFAULT_SPACE.fix(
        **{n: DEFAULT_SPACE[n].default for n in DEFAULT_SPACE.free_names() if n != "magnitude"}
    )
    assert len(GridSampler().sample(small, 5, 0)) == 5
    with pytest.raises(ValueError):
        grid_levels(DEFAULT_SPACE, 10)


# --- параметризация и учёт ограничений --------------------------------------


def test_point_to_spec_and_back() -> None:
    point = {**DEFAULT_SPACE.defaults(), "n_drifts": 3, "recurring": True, "form": "gradual"}
    spec = point_to_spec(point, concept_seed=1, seed=1)
    assert [e.returns_to for e in spec.events] == [None, 0, 1]  # A → B → A → B
    assert check(spec).ok
    back = spec_to_point(spec, DEFAULT_SPACE)
    assert {k: back[k] for k in ("n_drifts", "recurring", "form", "width", "magnitude")} == {
        "n_drifts": 3,
        "recurring": True,
        "form": "gradual",
        "width": point["width"],
        "magnitude": point["magnitude"],
    }


@pytest.fixture(scope="module")
def sobol_points() -> np.ndarray:
    return SobolSampler().design(DEFAULT_SPACE, 128, seed=11)


def test_policies(sobol_points: np.ndarray) -> None:
    res = {
        p: realize(DEFAULT_SPACE, sobol_points, policy=p, seed=11)
        for p in ("reject", "repair", "adapt")
    }
    rej, rep, ada = (summarize(res[p]) for p in ("reject", "repair", "adapt"))
    assert rej["valid"] == rej["accepted"] < 128 and rej["rejected"] > 0
    assert rep["accepted"] >= rej["accepted"] and rep["repaired"] > 0
    assert ada["accepted"] >= rej["accepted"]
    for cands in res.values():
        for c in cands:
            if c.spec is not None:
                assert check(c.spec).ok
                assert not DEFAULT_SPACE.out_of_domain(spec_to_point(c.spec, DEFAULT_SPACE))
    # исходные точки одинаковы при любом способе; отклоняются только недопустимые
    for a, b in zip(res["reject"], res["repair"], strict=True):
        assert a.point == b.point and (a.status == "valid") == (b.status == "valid")


def test_boundary_atoms_only_after_repair(sobol_points: np.ndarray) -> None:
    on = {
        p: [
            magnitude_on_boundary(c.spec)
            for c in realize(DEFAULT_SPACE, sobol_points, policy=p, seed=11)
            if c.spec
        ]
        for p in ("reject", "repair", "adapt")
    }
    assert sum(on["repair"]) > 0 and sum(on["adapt"]) == 0 and sum(on["reject"]) == 0


def test_adapt_maps_into_conditional_domain(sobol_points: np.ndarray) -> None:
    for c in realize(DEFAULT_SPACE, sobol_points, policy="adapt", seed=11):
        if c.spec is None:
            continue
        pt = spec_to_point(c.spec, DEFAULT_SPACE)
        if pt["kind"] != "prior":
            assert math.floor(pt["affected_share"] * pt["n_features"] + 0.5) >= 2  # C9
        assert pt["magnitude"] >= DEFAULT_SPACE["magnitude"].domain.low


def test_realize_is_deterministic(sobol_points: np.ndarray) -> None:
    a = realize(DEFAULT_SPACE, sobol_points[:16], seed=2)
    b = realize(DEFAULT_SPACE, sobol_points[:16], seed=2)
    assert [io.scenario_id(c.spec) for c in a if c.spec] == [
        io.scenario_id(c.spec) for c in b if c.spec
    ]


# --- шаблоны и наследование -------------------------------------------------


def test_deep_merge_rules() -> None:
    base = {"stream": {"n_samples": 10_000, "n_features": 5}, "drifts": {"schedule": {"count": 2}}}
    out = deep_merge(base, {"stream": {"n_features": None, "label_noise": 0.1}})
    assert out["stream"] == {"n_samples": 10_000, "label_noise": 0.1}
    assert "drifts" not in deep_merge(base, {"events": []})  # альтернативы взаимоисключающи
    assert base["stream"]["n_features"] == 5  # аргументы не меняются


def test_catalog_profiles_follow_difficulty_scale() -> None:
    for name, lam in PROFILES.items():
        spec = io.from_dict({"extends": f"catalog/{name}"})
        assert spec == io.from_dict(DIFFICULTY.template(lam)).model_copy(
            update={"description": spec.description}
        )
        assert check(spec).ok
    child = io.from_dict({"extends": "catalog/medium", "drifts": {"defaults": {"magnitude": 0.3}}})
    assert child.events[0].magnitude == 0.3 and child.n_events == 3


def test_file_inheritance_and_cycles(tmp_path) -> None:
    (tmp_path / "parent.yaml").write_text(
        "extends: catalog/easy\nstream: {n_features: 12}\n", encoding="utf-8"
    )
    (tmp_path / "child.yaml").write_text(
        "extends: parent\ndrifts: {schedule: {count: 2}}\n", encoding="utf-8"
    )
    spec = io.load(tmp_path / "child.yaml")
    assert spec.stream.n_features == 12 and spec.n_events == 2
    (tmp_path / "a.yaml").write_text("extends: b\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a\n", encoding="utf-8")
    with pytest.raises(TemplateError, match="цикл"):
        resolve({"extends": "a"}, base_dir=tmp_path)


# --- композиция ------------------------------------------------------------


def test_composition_positions() -> None:
    blocks = [
        comp.Stable(6_000),
        comp.Drift(kind="real", magnitude=0.2, form="gradual", width=1_000),
        comp.Stable(6_000),
        comp.Recur(to=0),
        comp.Stable(5_000),
    ]
    spec = comp.compose(blocks)
    assert spec.stream.n_samples == 18_000
    assert [(e.onset, e.end) for e in spec.events] == [(6_000, 7_000), (13_000, 13_000)]
    assert check(spec).ok
    data = {"blocks": comp.block_dicts(blocks)}
    assert io.from_dict(data) == spec.model_copy(update={"name": "unnamed"})
    short = comp.compose([comp.Stable(500), comp.Drift(magnitude=0.2), comp.Stable(5_000)])
    assert "C4" in check(short).codes()


# --- шкала сложности -------------------------------------------------------


def test_difficulty_is_monotone_and_valid() -> None:
    lams = np.linspace(0, 1, 11)
    table = DIFFICULTY.table(lams)
    for f in DIFFICULTY.factors:
        values = [row[f.name] for row in table]
        diffs = np.diff(values)
        assert np.all(diffs >= 0) if f.hard > f.easy else np.all(diffs <= 0)
        assert values[0] == f.easy and values[-1] == f.hard
    for lam in lams:
        for cs in range(20):
            assert check(DIFFICULTY.scenario(float(lam), concept_seed=cs)).ok


# --- наборы сценариев ------------------------------------------------------


def _config(**design) -> SuiteConfig:
    return SuiteConfig(
        name="t",
        template={"extends": "catalog/base"},
        design={"method": "sobol", "n": 16, "seed": 5, **design},
    )


def test_suite_is_deterministic_and_roundtrips(tmp_path) -> None:
    a, b = build_suite(_config()), build_suite(_config())
    assert a.suite_id == b.suite_id and len(a) == 16
    assert build_suite(_config(seed=6)).suite_id != a.suite_id
    a.save(tmp_path)
    loaded = load_suite(tmp_path)
    assert loaded.suite_id == a.suite_id and loaded.ids == a.ids
    assert a.diff(build_suite(_config(n=32)))["common"] > 0


def test_suite_detects_tampering(tmp_path) -> None:
    suite = build_suite(_config())
    suite.save(tmp_path)
    path = tmp_path / "scenarios" / f"{suite.scenarios[0].name}.yaml"
    spec = io.load(path)
    io.save(spec.model_copy(update={"seed": spec.seed + 1}), path)
    with pytest.raises(SuiteIntegrityError):
        load_suite(tmp_path)


def test_build_space_selection() -> None:
    cfg = SuiteConfig(
        name="s",
        vary=("magnitude", "width"),
        domains={"magnitude": {"low": 0.05, "high": 0.3}},
        design={"method": "lhs", "n": 8},
    )
    space = build_space(cfg)
    assert space.free_names() == ["width", "magnitude"]
    assert space["width"].active_if == ()  # условие на form снято: form берётся из шаблона
    with pytest.raises(KeyError):
        build_space(SuiteConfig(name="s", vary=("no_such",)))

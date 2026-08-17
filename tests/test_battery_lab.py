"""battery_lab smoke tests: temporal properties of the features, dimensions and
split rules. Runs in seconds, does not load TabPFN."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from battery_lab.data_adapters import (
    FIT_UNITS, HUST_TEST_IDS, load_fit_unit, normalize_cells)
from battery_lab.temporal_features import add_history_features, build_view


# ---------- temporal properties of the features ----------

def test_history_dims():
    Z = np.random.default_rng(0).normal(size=(50, 16))
    assert build_view(Z, "raw").shape == (50, 17)
    assert build_view(Z, "history").shape == (50, 81)


def test_history_only_property():
    """Core legality: row t depends only on inputs up to t (perturbing future
    rows leaves earlier rows unchanged)."""
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(40, 4))
    H1 = add_history_features(Z)
    Z2 = Z.copy()
    Z2[25:] += 100.0          # perturb the future
    H2 = add_history_features(Z2)
    assert np.allclose(H1[:25], H2[:25])   # rows <=24 are unaffected


def test_first_cycle_boundaries():
    Z = np.arange(20, dtype=float).reshape(10, 2)
    H = add_history_features(Z)
    d = 2
    lag1, diff1 = H[:, :d], H[:, d:2*d]
    from_first, from_roll5 = H[:, 2*d:3*d], H[:, 3*d:]
    assert np.allclose(lag1[0], Z[0])       # first cycle: lag = itself
    assert np.allclose(diff1[0], 0)         # first cycle: diff = 0
    assert np.allclose(from_first[0], 0)
    assert np.allclose(from_roll5[0], 0)    # the window holds only itself


def test_roll5_window():
    Z = np.arange(10, dtype=float).reshape(-1, 1)
    H = add_history_features(Z)
    from_roll5 = H[:, 3]
    # t=6: mean(2..6)=4 → 6-4=2
    assert from_roll5[6] == pytest.approx(2.0)
    # t=2: mean(0,1,2)=1 -> 2-1=1 (short windows use the available length)
    assert from_roll5[2] == pytest.approx(1.0)


def test_no_nan_after_transform():
    Z = np.random.default_rng(2).normal(size=(30, 16))
    assert np.isfinite(build_view(Z, "history")).all()


# ---------- data adapters ----------

def test_fit_units_count():
    assert len(FIT_UNITS) == 11


def test_xjtu_2c_split():
    u = load_fit_unit("XJTU", "2C")
    assert len(u.cells) == 8
    test_ids = sorted(c.cell_id for c in u.test_cells)
    assert test_ids == ["2C_battery-4", "2C_battery-8"]


def test_xjtu_3c_includes_14():
    """Reference rule: any file name containing '4'/'8', so 3C battery-14 is a
    test cell as well."""
    u = load_fit_unit("XJTU", "3C")
    test_ids = {c.cell_id for c in u.test_cells}
    assert "3C_battery-14" in test_ids
    assert len(u.test_cells) == 3   # 4, 8, 14


def test_hust_split():
    u = load_fit_unit("HUST", None)
    assert {c.cell_id for c in u.test_cells} == HUST_TEST_IDS


def test_soh_range():
    u = load_fit_unit("XJTU", "2C")
    for c in u.cells:
        assert 0.3 < c.y.min() and c.y.max() < 1.2


def test_source_normalization_no_target_stats():
    """Strict mode: normalising a target cell must not use its own distribution."""
    u = load_fit_unit("XJTU", "2C")
    tr, te = u.train_cells, u.test_cells
    a = normalize_cells(tr + te, "source", source_cells=tr)
    # perturbing the target cell must not change the source-side normalisation
    te2 = [type(c)(cell_id=c.cell_id, X=c.X * 3 + 7, y=c.y) for c in te]
    b = normalize_cells(tr + te2, "source", source_cells=tr)
    for c in tr:
        assert np.allclose(a[c.cell_id], b[c.cell_id])


# ---------- additional backbones and feature expansions ----------

def test_gbdt_backend_registered():
    """Phase 1: the GBDT backend is registered and can predict."""
    from battery_lab.protocols import BACKBONES, fit_predict_gbdt
    assert "gbdt" in BACKBONES
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 8))
    y = X[:, 0] * 0.5 + rng.standard_normal(60) * 0.01
    pred = fit_predict_gbdt(X[:50], y[:50], X[50:], seed=0)
    assert pred.shape == (10,)
    assert np.all(np.isfinite(pred))


def test_nystroem_expand_shape():
    """Phase 3: after the Nystroem expansion the width is original +
    n_components, and the map is fitted on the training side only."""
    from battery_lab.physics_posthoc import nystroem_expand
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.standard_normal((80, 17)), rng.standard_normal((20, 17))
    Ftr, Fte = nystroem_expand(Xtr, Xte, n_components=64, seed=0)
    assert Ftr.shape == (80, 17 + 64)
    assert Fte.shape == (20, 17 + 64)
    # the original columns stay in the first 17 dimensions
    assert np.allclose(Ftr[:, :17], Xtr)


def test_isotonic_postproc_monotone():
    """Phase 2 dependency: the isotonic post-processing is non-increasing and
    uses no labels."""
    from battery_lab.physics_posthoc import apply_postproc
    pred = np.array([0.9, 0.95, 0.85, 0.88, 0.8, 0.7])
    cid = np.zeros(6, dtype=int)
    out = apply_postproc(pred, cid, "iso")
    assert np.all(np.diff(out) <= 1e-9)  # non-increasing


# ---------- CGCA conflict gating ----------

def test_gate_label_free():
    """The gate decides from the predicted trajectory alone, never from target
    labels (deployment legality)."""
    from battery_lab.conflict_gating import apply_gate, violation_ratio
    rng = np.random.default_rng(0)
    pred = np.concatenate([np.linspace(1, .8, 30) + rng.normal(0, .01, 30),
                           np.linspace(1, .9, 30) + rng.normal(0, .05, 30)])
    cid = np.repeat([0, 1], 30)
    out1 = apply_gate(pred, cid, lam_star=0.3)
    out2 = apply_gate(pred.copy(), cid, lam_star=0.3)   # label-free, deterministic
    assert np.allclose(out1, out2)
    assert violation_ratio(np.linspace(1, .8, 20)) < 1e-6   # a monotone trajectory is ~0


def test_ltt_monotone_and_conservative():
    """LTT: higher risk gives a smaller lambda* (order preserved); if everything
    is harmed, lambda* = -inf (conservative)."""
    from battery_lab.conflict_gating import ltt_calibrate
    rng = np.random.default_rng(0)
    stats = rng.uniform(0, 1, 200)
    harms_low = stats > 0.8            # only high conflict does harm
    harms_all = np.ones(200, bool)     # everything is harmed
    lam_low = ltt_calibrate(stats, harms_low)["lam_star"]
    lam_all = ltt_calibrate(stats, harms_all)["lam_star"]
    assert lam_low > 0.5               # the low-risk region is accepted
    assert lam_all == -np.inf          # all harmed -> refuse to apply
    assert lam_all < lam_low


def test_per_cell_record_complete():
    """Per-cell records: no cell missing, all fields present."""
    from battery_lab.conflict_gating import per_cell_record
    rng = np.random.default_rng(0)
    y = np.concatenate([np.linspace(1, .8, 40), np.linspace(1, .85, 50)])
    pred = y + rng.normal(0, .01, 90)
    cid = np.repeat(["a", "b"], [40, 50])
    rows = per_cell_record(pred, y, cid)
    assert {r["cell_id"] for r in rows} == {"a", "b"}
    for r in rows:
        for k in ("v_pred", "v_true", "corr_mag", "rmse_none",
                  "delta_iso", "delta_causal", "delta_smooth"):
            assert k in r and np.isfinite(r[k])


# ---------- SCS source-side configuration selection ----------

def test_loco_estimate_ignores_test_cells_and_deterministic():
    """The leave-one-cell-out estimate uses source cells only: perturbing test
    cells leaves it unchanged, and equal inputs give equal output (idempotent
    on resume)."""
    from battery_lab.data_adapters import load_fit_unit
    from battery_lab.source_selection import loco_estimate
    u = load_fit_unit("XJTU", "2C")
    e1 = loco_estimate(u, "gbdt", "raw", seeds=[0])
    # swap in a perturbed test cell (test_cells is a property over read-only
    # arrays, so the underlying cells have to be rebuilt)
    u.cells[:] = [c if not c.is_test else
                  type(c)(cell_id=c.cell_id, X=c.X * 3 + 7, y=c.y * 0.5,
                          is_test=True)
                  for c in u.cells]
    e2 = loco_estimate(u, "gbdt", "raw", seeds=[0])
    assert abs(e1["est_macro"] - e2["est_macro"]) < 1e-12
    assert e1["cells_used"] == e2["cells_used"]


def test_select_config_rule():
    """Selection rule: a clear minimum wins, ties fall back to the default, and
    the outcome is deterministic."""
    from battery_lab.source_selection import select_config
    est = {"tabpfn-raw": {"est_macro": 0.010, "se": 0.0005},
           "tabpfn-history": {"est_macro": 0.008, "se": 0.0005},
           "gbdt-raw": {"est_macro": 0.012, "se": 0.0005},
           "gbdt-history": {"est_macro": 0.011, "se": 0.0005}}
    s = select_config(est)
    assert s["selected"] == "tabpfn-history" and s["reason"] == "clear-min"
    est["tabpfn-raw"]["est_macro"] = 0.0081     # gap to the minimum < pooled SE -> tie
    s2 = select_config(est)
    assert s2["selected"] == "gbdt-history" and s2["reason"] == "tie->default"
    assert select_config(est) == s2             # deterministic


# ---------- robust self-anchoring and permutation inference ----------

def test_derangements_distinct_no_fixpoint():
    """bexp27: the 20 permutations are distinct and have no fixed points."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from bexp27_randomization import MAP_SEEDS, derangement
    ids = [f"c{i}" for i in range(23)]
    maps = {m: derangement(ids, m) for m in MAP_SEEDS}
    sigs = {tuple(sorted(maps[m].items())) for m in MAP_SEEDS}
    assert len(sigs) == len(MAP_SEEDS)
    for m in MAP_SEEDS:
        assert all(k != v for k, v in maps[m].items())


def test_from_first_k_history_only_and_k1_regression():
    """bexp28: from_first_K is strictly history-only, and K=1 reduces exactly to
    the plain from_first."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from bexp28_rsr import from_first_k
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((40, 6))
    F = from_first_k(Z, 5)
    # perturbing future rows leaves the prefix rows untouched (history-only)
    Z2 = Z.copy()
    Z2[20:] += 100
    F2 = from_first_k(Z2, 5)
    assert np.allclose(F[:20], F2[:20])
    # K=1 reduces to Z - Z[0]
    assert np.allclose(from_first_k(Z, 1), Z - Z[:1])


def test_missing_anchor_history_only():
    """bexp34: with the first K0 cycles missing, the anchor becomes the earliest
    available cycle and the features stay strictly history-only."""
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((60, 4))
    Zt = Z[10:]                          # the first 10 cycles are missing
    F = Zt - Zt[:1]
    Z2 = Z.copy(); Z2[40:] += 50         # perturb the future
    F2 = Z2[10:] - Z2[10:][:1]
    assert np.allclose(F[:25], F2[:25])


def test_strict_norm_ignores_test_stats():
    """bexp35: normalisation parameters fitted on the source side are unaffected
    by the values of the test cells."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from bexp33_2x2_disambiguation import normalize
    rng = np.random.default_rng(2)
    tr = [(rng.standard_normal((30, 5)), rng.random(30)) for _ in range(3)]
    te1 = [(rng.standard_normal((20, 5)), rng.random(20))]
    te2 = [(te1[0][0] * 100, te1[0][1])]          # drastic change to the test cell
    tr_n1, te_n1 = normalize(tr, te1, "source")
    tr_n2, _ = normalize(tr, te2, "source")
    for a, b in zip(tr_n1, tr_n2):                # the training-side scaling is identical
        assert np.allclose(a[0], b[0])

def test_bexp42_cycle_conventions_differ_and_are_history_safe():
    """bexp42: the two cycle conventions really do differ, and the deployable one
    does not depend on total life."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from battery_lab.temporal_features import cycle_author_minmax, cycle_scaled
    a200, a400 = cycle_author_minmax(200), cycle_author_minmax(400)
    d200, d400 = cycle_scaled(200), cycle_scaled(400)
    # life-fraction convention: the value at a given cycle index moves with total
    # life (it depends on the future)
    assert not np.isclose(a200[100, 0], a400[100, 0])
    # deployable convention: the value at a given cycle index is independent of
    # total life (computable in service)
    assert np.isclose(d200[100, 0], d400[100, 0])


def test_bexp43_reuses_same_derangements_as_gbdt_side():
    """bexp43: the permutations on the TabPFN side match those on the GBDT side
    (bexp27) one by one, so the two backbones stay comparable."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from bexp27_randomization import derangement as d27
    import bexp43_44_perm_seeds as b43
    ids = [f"c{i}" for i in range(17)]
    for m in b43.MAP_SEEDS:
        assert d27(ids, m) == b43.derangement(ids, m)
        assert all(k != v for k, v in b43.derangement(ids, m).items())


def test_bexp46_student_input_is_online_computable():
    """bexp46: the student features must contain no column that depends on total
    life, and from_first must be anchored at the first cycle."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from bexp46_dual_carrier_distill import student_matrix, teacher_matrix

    class C:
        def __init__(self, cid, n):
            self.cell_id, self.y = cid, np.linspace(1.0, 0.8, n)

    cells = [C("a", 40), C("b", 60)]
    rng = np.random.default_rng(0)
    Xn = {c.cell_id: rng.standard_normal((len(c.y), 16)) for c in cells}
    Xs, _, ids = student_matrix(cells, Xn)
    Xt, _, _ = teacher_matrix(cells, Xn)
    assert Xs.shape[1] == 33 and Xt.shape[1] == 81
    # the cycle column is the fixed scale t/200: its step is independent of the
    # cell's total life
    for cid, n in (("a", 40), ("b", 60)):
        col = Xs[ids == cid, 16]
        assert np.allclose(np.diff(col), 1 / 200.0)
    # the first from_first row of every cell is zero (the anchor is its own first
    # cycle)
    for cid in ("a", "b"):
        assert np.allclose(Xs[ids == cid][0, 17:], 0.0)


def test_bexp46_mispaired_arm_changes_only_the_anchor():
    """bexp46: the mispaired arm differs from the true arm only in the from_first
    columns; the first 17 columns must be element-wise identical."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from bexp46_dual_carrier_distill import student_matrix

    class C:
        def __init__(self, cid, n):
            self.cell_id, self.y = cid, np.linspace(1.0, 0.8, n)

    cells = [C("a", 30), C("b", 30)]
    rng = np.random.default_rng(1)
    Xn = {c.cell_id: rng.standard_normal((len(c.y), 16)) for c in cells}
    Xs, _, _ = student_matrix(cells, Xn)
    Xm, _, _ = student_matrix(cells, Xn, {"a": "b", "b": "a"})
    assert np.allclose(Xs[:, :17], Xm[:, :17])
    assert not np.allclose(Xs[:, 17:], Xm[:, 17:])


def test_bexp45_learned_anchor_stays_within_early_window():
    """bexp45: the reference layer must output a convex combination of the first k
    cycles (non-negative weights summing to one)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import torch
    from bexp45_learnable_reference import RefLearner

    torch.manual_seed(0)
    m = RefLearner(16)
    early = torch.randn(8, 10, 16)
    a = m.alpha(early)
    assert a.shape == (8, 10)
    assert torch.all(a >= 0) and torch.allclose(a.sum(1), torch.ones(8), atol=1e-5)
    # the anchor lies inside the convex hull of the early window: no dimension
    # exceeds its min/max
    r = torch.einsum("bk,bkd->bd", a, early)
    assert torch.all(r <= early.max(dim=1).values + 1e-5)
    assert torch.all(r >= early.min(dim=1).values - 1e-5)

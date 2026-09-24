"""Curve-unit and window-boundary unit tests (F3-03 acceptance: known constant-current samples
plus upstream boundaries; no data dependency).

Covers bexp62's curve utilities and unit conventions (the publisher source
Battery-dataset-preprocessing-code-library@e986a740 default definitions):
  1) trapz_ah: Ah of a constant-current sample (seconds axis; the minutes axis is converted by
     callers via x60);
  2) MIT convention: t in minutes, I as relative current (1C = 1.1 Ah) -> qchg = Qnom x I dt[min]/60;
  3) boundary inclusivity of the publisher default windows: CC=(V<=4.199)&(I>=3.9), CV=(V>=4.199),
     measured tail of the released CV column I<=0.5 (all inclusive);
  4) longest_run: the longest consecutive run under multi-segment currents (the satellite batch's
     3-segment case).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str):
    spec = importlib.util.spec_from_file_location("b62_units", ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B62 = _load("experiments/bexp62_curve_features.py")


# ---------- 1) Ah of a constant-current sample (seconds axis) ----------

def test_trapz_ah_constant_current_seconds():
    # 4 A constant current for 1 h = 4 Ah
    t = np.arange(0, 3601, 1.0)
    i = np.full(t.size, 4.0)
    assert B62.trapz_ah(t, i) == pytest.approx(4.0, rel=1e-12)


def test_trapz_ah_published_cc_window_shape():
    # released CC Q ~ I*t/3600 (constant 4 A x 314.94 s = 0.34993 Ah)
    t = np.array([0.0, 314.94])
    i = np.array([4.0, 4.0])
    assert B62.trapz_ah(t, i) == pytest.approx(0.34993, rel=1e-4)


def test_trapz_ah_minutes_convention():
    # with t in minutes callers must multiply by 60: 1 A for 60 min = 1 Ah
    t_min = np.arange(0, 61, 1.0)
    i = np.ones(t_min.size)
    assert B62.trapz_ah(t_min * 60.0, i) == pytest.approx(1.0, rel=1e-12)
    # if the x60 is missed (the v1 MIT-path error), the result shrinks to 1/60
    assert B62.trapz_ah(t_min, i) == pytest.approx(1.0 / 60.0, rel=1e-12)


# ---------- 2) MIT relative-current convention ----------

def test_mit_c_rate_convention():
    # I=1.0 (1C) at 1C=1.1 Ah rated charging for 1 h should give 1.1 Ah
    t_min = np.arange(0, 61, 1.0)
    i = np.ones(t_min.size)
    q = B62.trapz_ah(t_min * 60.0, i) * B62.MIT_QNOM
    assert q == pytest.approx(1.1, rel=1e-12)


def test_mit_qnom_constant():
    assert B62.MIT_QNOM == 1.1


# ---------- 3) Window-boundary inclusivity (the publisher source's <= / >=) ----------

def test_cc_default_window_boundaries():
    v = np.array([4.199, 4.1995, 4.2, 3.9])
    i = np.array([3.9, 3.9, 3.9, 3.9])
    # CC = (V<=4.199) & (I>=3.9): V=4.199 and I=3.9 are both included
    cc = (v <= 4.199) & (i >= 3.9)
    assert cc.tolist() == [True, False, False, True]
    # the old mis-copied version (V>=4.199 and I>3.9) is empty on this sample
    wrong = (v >= 4.199) & (i > 3.9)
    assert wrong.tolist() == [False, False, False, False]


def test_cv_default_window_is_not_complement():
    v = np.array([4.199, 4.1995, 3.99, 3.9])
    i = np.array([3.9, 1.0, 0.5, 3.9])
    cv = v >= 4.199
    cc = (v <= 4.199) & (i >= 3.9)
    # CV is not the complement of CC: the 3.99/3.9 V rows are neither in CC (current/voltage fail) nor in CV
    assert cv.tolist() == [True, True, False, False]
    assert cc.tolist() == [True, False, False, True]
    assert not (~cc == cv).all()


def test_tail_window_threshold_inclusive():
    # the tail segment I<=0.5 includes equality: constant 4 A for the first 50 min, then 50 min ramping from 0.5 down to 0.1
    t_min = np.arange(0, 101, 1.0)
    i = np.concatenate([np.full(50, 4.0), np.linspace(0.5, 0.1, 50)])
    w = B62.tail_window(t_min * 60.0, i)
    assert w is not None
    assert w["s"] == pytest.approx(49 * 60.0)          # from minute 50 to minute 99
    assert w["i_mean"] == pytest.approx(0.3, rel=1e-9)  # mean of linspace(0.5,0.1,50)


# ---------- 4) longest_run ----------

def test_longest_run_multisegment():
    i = np.array([0.0, 1, 1, 1, 0, -1, -1, 0, 1, 1, 0])
    s = B62.longest_run(i > 0)
    assert (s.start, s.stop) == (1, 4)
    s2 = B62.longest_run(i < 0)
    assert (s2.start, s2.stop) == (5, 7)


def test_dur_s_endpoints_only():
    t = np.array([0.0, 10.0, 25.0, 30.0])
    m = np.array([True, True, False, True])
    assert B62.dur_s(t, m) == pytest.approx(30.0)     # endpoint difference over selected points 0/10/30
    assert B62.dur_s(t, np.array([True, False, False, False])) is None


# ---------- 5) Positive-segment integration (v5: complete-array segmentation; fourth-round audit item A regression) ----------

def test_positive_segment_ah_two_segments():
    # fourth-round audit reproduction sample: two positive segments of 1 minute each; correct = (1x60 + 2x60)/3600 = 0.05 Ah
    t = np.arange(7, dtype=float) * 60.0
    i = np.array([1.0, 1, 0, 0, 2, 2, 0])
    assert B62.positive_segment_ah(t, i) == pytest.approx(0.05, rel=1e-12)


def test_positive_segment_ah_leading_rest():
    t = np.arange(6, dtype=float) * 60.0
    i = np.array([0.0, 0, 3, 3, 0, 0])               # leading rest
    assert B62.positive_segment_ah(t, i) == pytest.approx(3 * 60 / 3600.0, rel=1e-12)


def test_positive_segment_ah_single_segment():
    t = np.arange(4, dtype=float) * 60.0
    i = np.array([2.0, 2, 2, 0])
    assert B62.positive_segment_ah(t, i) == pytest.approx(2 * 120 / 3600.0, rel=1e-12)


def test_positive_segment_ah_empty():
    t = np.arange(3, dtype=float) * 60.0
    assert B62.positive_segment_ah(t, np.array([0.0, 0, -1])) is None


class _FakeH5(dict):
    """Minimal h5py.File stand-in (context protocol + dict access) to regress bexp64's integration path."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_mit_qchg_series_full_index_space(monkeypatch):
    """v5 regression (fourth-round audit item A): calls the real bexp64.qchg_series_mit (h5py mocked).

    The two-segment sample must yield 0.055 Ah (= 1.1 x (1x60 + 2x60)/3600); the v3/v4 implementation
    returned only 0.018333 Ah (the second segment was mis-sliced and lost).
    """
    import sys as _sys
    import types
    fake = _FakeH5({"batch": {"cycles": np.array([["c1"]])},
                    "c1": {"V": np.zeros((1, 1)),
                           "I": np.array([["Id"]]),
                           "t": np.array([["Td"]])},
                    "Id": np.array([1.0, 1, 0, 0, 2, 2, 0]),
                    "Td": np.arange(7, dtype=float)})
    monkeypatch.setitem(_sys.modules, "h5py", types.SimpleNamespace(File=lambda *a, **k: fake))
    b64 = _load("experiments/bexp64_physical_ref.py")
    got = float(b64.qchg_series_mit("in-memory", 0)[0])
    assert got == pytest.approx(1.1 * 3 * 60 / 3600.0, rel=1e-12)

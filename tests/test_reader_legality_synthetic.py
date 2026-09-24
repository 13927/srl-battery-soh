"""**Data-free** tests of reading-path legality (W01 key invariants; wired into the routine
`make check`).

Division of labour with `test_strict_reader.py`:
  - this file asserts the reading-path rules themselves on **synthetic DataFrames**, so it needs no
    dataset under external/ and can run in the routine `make check`;
  - `test_strict_reader.py` makes the same assertions on the real releases (with row-level
    measurements), depends on the datasets, is marked `needs_data`, and runs only when the data is
    present (`make test`).

Facts pinned (corresponding to reviewer comment M4 and plan W01):
  L1 `_strict_row_keep` decides only by **the current row's input columns**, independent of `capacity`;
  L2 rewriting any `capacity` (or NaN-ing the whole column) must not change the strict reader's
     keep set;
  L3 `_strict_row_keep` is prefix-invariant under **trajectory truncation** (decidable at time t);
  L4 `_three_sigma_mask` returns the **kept rows' original row numbers** (not all pre-filter row
     numbers);
  L5 `_three_sigma_mask` uses **full-lifetime** statistics and **includes capacity** -- a fact of
     the author offline protocol and the reason it must be recorded separately from the strict
     online reader (a documentation test against misstatement).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from battery_lab.data_adapters import (  # noqa: E402
    FEATURE_COLS,
    _strict_row_keep,
    _three_sigma_mask,
)


def _frame(n=40, n_bad=3, bad_col="voltage entropy", seed=0, label_shift=0.0):
    """Build one frame of synthetic cell data: the first rows carry non-finite inputs (mimicking the XJTU missing rows in the release)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    df["capacity"] = 2.0 - np.linspace(0, 0.3, n) + label_shift
    for r in range(n_bad):
        df.loc[r, bad_col] = np.inf if r == 0 else np.nan
    return df


def test_L1_strict_rule_is_row_local():
    """L1: the keep flag depends only on that row's own 16 input columns."""
    df = _frame(n_bad=3)
    keep = _strict_row_keep(df)
    assert keep.shape == (len(df),)
    # the first 3 rows carry non-finite inputs -> must be dropped; the remaining rows must be kept
    assert not keep[:3].any()
    assert keep[3:].all()
    # the decision is independent of capacity finiteness: NaN-ing the whole label column keeps
    # the keep set unchanged
    df2 = df.copy()
    df2["capacity"] = np.nan
    assert np.array_equal(_strict_row_keep(df2), keep)


def test_L2_label_cannot_change_row_set():
    """L2: rewriting the label arbitrarily must not change the keep set (target-label isolation)."""
    df = _frame()
    base = _strict_row_keep(df)
    for shift in (0.0, +10.0, -10.0):
        assert np.array_equal(_strict_row_keep(_frame(label_shift=shift)), base)
    df3 = df.copy()
    df3.loc[5:15, "capacity"] = 999.0
    assert np.array_equal(_strict_row_keep(df3), base)


def test_L3_strict_rule_is_prefix_invariant():
    """L3: after truncation, keep flags before the cut are row-wise unchanged (decidable at time t)."""
    df = _frame()
    full = _strict_row_keep(df)
    for cut in (10, 17, 33):
        assert np.array_equal(_strict_row_keep(df.iloc[:cut]), full[:cut])


def test_L4_three_sigma_returns_kept_row_numbers():
    """L4: the author path must return the **kept rows'** original row numbers, not the pre-filter row numbers."""
    df = _frame(n=60, n_bad=4, seed=3)
    kept_idx, kept_df = _three_sigma_mask(df)
    assert len(kept_idx) == len(kept_df) < len(df)
    assert kept_idx.max() < len(df)
    # the row numbers must agree with the kept frame's content: taking the kept row numbers, the
    # capacity sequence must equal kept_df's
    assert np.allclose(df.loc[kept_idx, "capacity"].to_numpy(),
                       kept_df["capacity"].to_numpy())


def test_L5_author_filter_uses_full_lifetime_and_label():
    """L5 (documentation): the author 3-sigma filter includes capacity and deletes by full-lifetime statistics, so prefixes change."""
    df = _frame(n=60, n_bad=0, seed=7)
    # create an obvious capacity outlier: outside mean+/-3sigma, deleted by the author filter
    df.loc[0, "capacity"] = 100.0
    kept_idx, _ = _three_sigma_mask(df)
    assert 0 not in set(kept_idx.tolist()), "the author filter should delete this capacity-outlier row"
    # precondition: changing only the label, not the inputs, leaves the strict reader untouched
    assert _strict_row_keep(df).all()
    # prefix sensitivity: the author filter decides on full-lifetime statistics, so truncation
    # changes decisions before the cut
    short = df.iloc[:30]
    kept_short, _ = _three_sigma_mask(short)
    assert not np.array_equal(kept_short, kept_idx[kept_idx < 30])

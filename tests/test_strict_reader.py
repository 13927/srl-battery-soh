"""Legality tests for the strict-online reading path (W01).

These tests pin the requirement that "the target's future / labels must not enter the prediction
path":

  T1 prefix invariance: truncating the trajectory must not change the keep flags, record indices
     or feature values of rows **at or before the cut**
  T2 label isolation  : rewriting capacity arbitrarily must not change kept rows, record indices
     or feature values
  T3 anchor definition: the strict_online anchor is the first **valid-input** record (usually row 0)
  T4 author-path guard: author_offline's numerical behaviour matches the archive (regression guard)
  T5 leakage fact on record: the authors' 3-sigma filter deletes leading rows, so its "first-cycle
     anchor" is not the first cycle (a **documentation test** fixing a known fact to prevent
     misstatement)

Data dependency: requires the released CSVs under external/PINN4SOH; when missing, tests skip
explicitly with a reason.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from battery_lab.data_adapters import (  # noqa: E402
    FEATURE_COLS,
    _load_cell_csv,
    _strict_row_keep,
    _three_sigma_mask,
)

DATA = ROOT / "external" / "PINN4SOH" / "data"
CELL_2C1 = DATA / "XJTU data" / "2C_battery-1.csv"
CELL_MIT1 = DATA / "MIT data" / "2017-05-12" / "2017-05-12_battery-1.csv"

needs_data = pytest.mark.skipif(
    not CELL_2C1.exists() or not CELL_MIT1.exists(),
    reason="requires the released CSVs under external/PINN4SOH (skipped when missing; see ARCHIVE.md section 3)",
)
# Also carries the needs_data marker: `make check` runs only the data-free subset
# (`-m "not needs_data"`); this file (with row-level measurements) runs under
# `make test` / when the data is present.
_needs_data_marker = pytest.mark.needs_data


# --------------------------------------------------------------------------- #
# T1 prefix invariance
# --------------------------------------------------------------------------- #
@needs_data
@_needs_data_marker
@pytest.mark.parametrize("path", [CELL_2C1, CELL_MIT1])
def test_strict_keep_is_prefix_invariant(path):
    """After truncating the trajectory, the keep flags of every row at or before the cut must match the full trajectory."""
    df = pd.read_csv(path)
    full = _strict_row_keep(df)
    for t in (0, 1, 5, 17, 128, len(df) // 2, len(df) - 1):
        if t >= len(df):
            continue
        prefix = df.iloc[: t + 1]
        part = _strict_row_keep(prefix)
        assert np.array_equal(part, full[: t + 1]), f"keep flags changed at cut {t}"


# --------------------------------------------------------------------------- #
# T2 label isolation
# --------------------------------------------------------------------------- #
@needs_data
@_needs_data_marker
@pytest.mark.parametrize("path", [CELL_2C1, CELL_MIT1])
def test_strict_reader_ignores_labels(path):
    """Rewriting capacity must not affect strict_online's row set, record indices or feature values."""
    nominal = 2.0 if "XJTU" in str(path) else 1.1
    base = _load_cell_csv(path, nominal, reader="strict_online")

    df = pd.read_csv(path)
    df["capacity"] = np.linspace(0.1, 9.9, len(df))  # a clearly absurd label
    tmp = path.with_name("_tmp_label_swap.csv")
    df.to_csv(tmp, index=False)
    try:
        swapped = _load_cell_csv(tmp, nominal, reader="strict_online")
    finally:
        tmp.unlink(missing_ok=True)

    assert np.array_equal(base.record_index, swapped.record_index)
    assert base.X.shape == swapped.X.shape
    assert np.array_equal(base.X, swapped.X)
    assert not np.array_equal(base.y, swapped.y)  # only the label itself may differ


# --------------------------------------------------------------------------- #
# T3 anchor definition
# --------------------------------------------------------------------------- #
@needs_data
@_needs_data_marker
def test_strict_anchor_is_first_valid_record():
    """strict_online's X[0] must come from CSV row 0 (when its input is valid)."""
    raw = pd.read_csv(CELL_2C1)
    first_valid = int(np.flatnonzero(_strict_row_keep(raw))[0])
    cell = _load_cell_csv(CELL_2C1, 2.0, reader="strict_online")
    assert first_valid == 0, "this file's row 0 input should be valid"
    assert cell.record_index[0] == first_valid
    assert np.allclose(cell.X[0], raw.loc[first_valid, FEATURE_COLS].to_numpy(float))
    assert np.allclose(cell.y[0], raw.loc[first_valid, "capacity"] / 2.0)


# --------------------------------------------------------------------------- #
# T4 author-path guard (regression)
# --------------------------------------------------------------------------- #
@needs_data
@_needs_data_marker
def test_author_reader_unchanged():
    """author_offline's row count matches the archive (2C_battery-1 -> 355 rows)."""
    idx, d = _three_sigma_mask(pd.read_csv(CELL_2C1))
    assert len(d) == 355
    assert d[FEATURE_COLS].to_numpy(float).shape[1] == 16
    cell = _load_cell_csv(CELL_2C1, 2.0, reader="author_offline")
    assert cell.X.shape[0] == 355
    assert np.array_equal(cell.record_index, np.asarray(idx))


# --------------------------------------------------------------------------- #
# T5 leakage fact (documentation)
# --------------------------------------------------------------------------- #
@needs_data
@_needs_data_marker
@pytest.mark.parametrize("path", [CELL_2C1, CELL_MIT1])
def test_author_filter_drops_leading_rows(path):
    """The authors' 3-sigma filter deletes leading rows -> its "first-cycle anchor" is not the true
    first cycle (a known fact).

    The test pins this fact: Pipeline A's x_1 is the "first surviving row", not CSV row 0; any
    statement claiming anchor=first-cycle disagrees with the implementation.
    """
    df = pd.read_csv(path)
    kept_idx, _ = _three_sigma_mask(df)
    assert kept_idx[0] > 0, "the author filter indeed removes leading rows for this file"
    strict_kept = np.flatnonzero(_strict_row_keep(df))
    assert strict_kept[0] == 0
    assert kept_idx[0] >= strict_kept[0]

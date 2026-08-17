"""Adapters for the four libraries: the reference PINN4SOH loading, preprocessing
and splitting, reproduced.

Reference protocol (aligned line by line with
external/PINN4SOH/dataloader/dataloader.py):
  1. Per-cell 3-sigma filtering (all columns including capacity; dropna first,
     then remove outlier rows)
  2. capacity / nominal_capacity -> SOH label
  3. Feature columns min-max scaled to [-1, 1] by the cell's own full lifetime
     (the reference default normalization_method='min-max')
  4. Official cell-level splits:
     XJTU: file name of a batch contains '4'/'8' -> test
     HUST: fixed list of 20 test cells
     MIT : trailing id in the file name, id % 5 == 0 -> test
     TJU : 1-based file index within a batch ending in {(5,9),(4,8),(5,9)} -> test

Note: the reference min-max reads the target cell's complete lifetime
distribution, which is part of their protocol; this module reproduces it
faithfully for the like-for-like comparison. The strict mode
(normalize='source') is for strict leave-one-cell-out: normalisation statistics
are fitted on the source cells only.

Evaluation units (11): six XJTU batches + three TJU batches + HUST + MIT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1] / "external" / "PINN4SOH" / "data"

FEATURE_COLS = [
    "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
    "CC Q", "CC charge time", "voltage slope", "voltage entropy",
    "current mean", "current std", "current kurtosis", "current skewness",
    "CV Q", "CV charge time", "current slope", "current entropy",
]

XJTU_BATCHES = ["2C", "3C", "R2.5", "R3", "RW", "satellite"]
TJU_BATCHES = ["Dataset_1_NCA_battery", "Dataset_2_NCM_battery",
               "Dataset_3_NCM_NCA_battery"]
TJU_TEST_MOD = {0: (5, 9), 1: (4, 8), 2: (5, 9)}
TJU_NOMINAL = {0: 3.5, 1: 3.5, 2: 2.5}
HUST_TEST_IDS = {"1-4", "1-8", "2-4", "2-8", "3-4", "3-8", "4-4", "4-8",
                 "5-4", "5-7", "6-4", "6-8", "7-4", "7-8", "8-4", "8-8",
                 "9-4", "9-8", "10-4", "10-8"}

# the 11 evaluation units: (unit_id, dataset, batch_key)
FIT_UNITS = (
    [(f"XJTU-{b}", "XJTU", b) for b in XJTU_BATCHES]
    + [(f"TJU-{i+1}", "TJU", i) for i in range(3)]
    + [("HUST", "HUST", None), ("MIT", "MIT", None)]
)


@dataclass
class CellData:
    """One cell: raw (unnormalised) features after 3-sigma filtering, plus SOH."""

    cell_id: str
    X: np.ndarray          # (n_cycles, 16) raw features after filtering
    y: np.ndarray          # (n_cycles,) SOH = capacity / nominal
    is_test: bool = False


@dataclass
class FitUnit:
    unit_id: str
    dataset: str
    cells: List[CellData] = field(default_factory=list)

    @property
    def train_cells(self):
        return [c for c in self.cells if not c.is_test]

    @property
    def test_cells(self):
        return [c for c in self.cells if c.is_test]


def _three_sigma_filter(df: pd.DataFrame) -> pd.DataFrame:
    """The reference per-cell 3-sigma filter (all columns, capacity included)."""
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    out_index: set = set()
    for col in df.columns:
        s = df[col]
        rule = (s.mean() - 3 * s.std() > s) | (s.mean() + 3 * s.std() < s)
        out_index.update(np.arange(len(s))[rule])
    return df.drop(sorted(out_index), axis=0).reset_index(drop=True)


def _load_cell_csv(path: Path, nominal: float) -> CellData:
    df = pd.read_csv(path)
    df = _three_sigma_filter(df)
    X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = (df["capacity"] / nominal).to_numpy(dtype=np.float64)
    return CellData(cell_id=path.stem, X=X, y=y)


def load_fit_unit(dataset: str, batch_key=None) -> FitUnit:
    """Load every cell of one evaluation unit, with the official train/test flag."""
    cells: List[CellData] = []
    if dataset == "XJTU":
        root = DATA_ROOT / "XJTU data"
        for f in sorted(root.glob("*.csv")):
            # reference rule: `batch in file` (satellite matches Sim_satellite_*)
            if batch_key not in f.name:
                continue
            cell = _load_cell_csv(f, nominal=2.0)
            # reference rule: file name containing '4' or '8' -> test
            # (so 3C_battery-14 is a test cell too)
            cell.is_test = ("4" in f.name) or ("8" in f.name)
            cells.append(cell)
        return FitUnit(unit_id=f"XJTU-{batch_key}", dataset="XJTU", cells=cells)

    if dataset == "TJU":
        b = int(batch_key)
        root = DATA_ROOT / "TJU data" / TJU_BATCHES[b]
        files = sorted(root.glob("*.csv"))
        # the reference uses os.listdir order and enumerates files 1-based;
        # listdir order is not stable, so the 1-based index is taken over sorted
        # file names to keep it deterministic
        for i, f in enumerate(files):
            cell = _load_cell_csv(f, nominal=TJU_NOMINAL[b])
            cell.is_test = (i + 1) % 10 in TJU_TEST_MOD[b]
            cells.append(cell)
        return FitUnit(unit_id=f"TJU-{b+1}", dataset="TJU", cells=cells)

    if dataset == "HUST":
        root = DATA_ROOT / "HUST data"
        for f in sorted(root.glob("*.csv")):
            cell = _load_cell_csv(f, nominal=1.1)
            cell.is_test = f.stem in HUST_TEST_IDS
            cells.append(cell)
        return FitUnit(unit_id="HUST", dataset="HUST", cells=cells)

    if dataset == "MIT":
        root = DATA_ROOT / "MIT data"
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            for f in sorted(sub.glob("*.csv")):
                cell = _load_cell_csv(f, nominal=1.1)
                cell_num = int(f.stem.split("-")[-1])
                cell.is_test = cell_num % 5 == 0
                cells.append(cell)
        return FitUnit(unit_id="MIT", dataset="MIT", cells=cells)

    raise ValueError(f"unknown dataset {dataset!r}")


def normalize_cells(
    cells: List[CellData], method: str = "author_minmax",
    source_cells: Optional[List[CellData]] = None,
) -> Dict[str, np.ndarray]:
    """Return {cell_id: normalised features}.

    author_minmax: the reference protocol -- per-cell full-lifetime min-max to
                   [-1, 1] (it reads the cell's complete lifetime distribution,
                   which is part of that protocol)
    source:        strict mode -- one global min-max fitted on the pooled
                   source_cells; target cells are only transformed and their own
                   distribution is never read
    """
    out = {}
    if method == "author_minmax":
        for c in cells:
            mn, mx = c.X.min(axis=0), c.X.max(axis=0)
            rng = np.where(mx - mn == 0, 1.0, mx - mn)
            out[c.cell_id] = 2 * (c.X - mn) / rng - 1
    elif method == "source":
        assert source_cells, "source mode requires source_cells"
        pool = np.vstack([c.X for c in source_cells])
        mn, mx = pool.min(axis=0), pool.max(axis=0)
        rng = np.where(mx - mn == 0, 1.0, mx - mn)
        for c in cells:
            out[c.cell_id] = 2 * (c.X - mn) / rng - 1
    else:
        raise ValueError(f"unknown normalisation {method!r}")
    return out

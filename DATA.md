# Obtaining the datasets

No battery data is redistributed in this repository. All experiments read the
third-party open-access releases below; place them under `external/` exactly as
shown.

## 1. Primary benchmark: the four processed libraries (XJTU / TJU / HUST / MIT)

Released with the reference physics-informed work (PINN4SOH, Wang et al.,
Nature Communications 2024), preprocessed features and official splits used
here unchanged:

```bash
git clone https://github.com/wang-fujin/PINN4SOH external/PINN4SOH
```

Expected layout: `external/PINN4SOH/data/{XJTU data,TJU data,HUST data,MIT data}`.

## 2. Companion deep-learning benchmark (Table 8)

```bash
git clone https://github.com/wang-fujin/SOHbenchmark external/SOHbenchmark
```

Used by `bexp30`/`bexp31` via
`external/SOHbenchmark/data/XJTU/handcraft_features/batch-*_features.xlsx`.

## 3. Independent verification library MATR-CLO (Table 9)

The closed-loop-optimisation batch of Attia et al. (Nature 2020), from
https://data.matr.io/1/ (project "Closed-loop optimization of fast-charging
protocols"). Place the batch file at
`external/unexposed/MATR_CLO_2019-01-24.mat` and convert it once into
`external/unexposed/matr_clo_unit.npz`; the conversion is the `load_cells`
path documented in `experiments/bexp41_matrclo_protocol.py`.

## Notes

- The unit tests: 72 in total; 13 involve the dataset releases (5 assert
  against the official splits and fail loudly when the data is missing; 8 more
  are marked `needs_data` and skip cleanly with a reason); the remaining 59 run
  without any data.
- Re-running experiments is optional: every manuscript number is re-derivable
  from the committed archives in `results/battery/` (use `--report`).
- Raw-release identity: `experiments/raw_data_manifest.py` aligns the raw
  releases (XJTU / MIT) with the processed files, emits explicit per-row
  mappings and a release gate, and writes
  `results/battery/raw_data_manifest.json`; the curve-feature extraction
  (`bexp62`) and the simple-baseline experiments (`bexp63`-`bexp65`) consume it.

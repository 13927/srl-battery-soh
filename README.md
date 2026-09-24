# Deployable Self-Referenced Features for Cross-Cell Battery SOH Estimation

Code and result archives for the manuscript *"Deployable Self-Referenced
Features for Cross-Cell Battery State-of-Health Estimation: Randomized Pairing
Controls and Benchmarking"* (submitted to *Energies*).

The self-referenced lift (SRL) augments per-cycle statistical features with the
offset from the same cell's first recorded cycle, `x_t - x_1`, making the
comparison against the manufacturing origin explicit and computable in service.
Every number in the manuscript can be re-derived from the archives in
`results/battery/`, and every experiment can be re-run from the public
datasets (see `DATA.md`).

This repository holds both the initial submission and the materials added during
the first revision: the corrected strict-online reader, the experiments
`bexp50`-`bexp65` (baseline ladders, the strict-reader re-run, cost measurements,
curve-feature extraction, the simple curve-based references, and the availability
checks), the updated result archives, and the raw-data manifests. Superseded
archives are kept next to their replacements as the revision's historical record;
the script docstrings carry the corresponding version notes.

## Layout

| Path | Content |
|---|---|
| `battery_lab/` | Data adapters (official splits; two readers, `author_offline` and `strict_online`), feature construction, evaluation protocols |
| `experiments/bexp*.py` | The 60 experiment scripts of the study (`bexp1`-`bexp65`; some numbers combine two experiments); 24 of the original scripts carry decision criteria (34 in total), frozen a priori in the script docstring (marked `frozen criteria (committed before the run)`); the experiments added in the revision state their criteria in the docstring, and Section 2.7 of the manuscript grades their evidence |
| `experiments/raw_data_manifest.py` | Identity-alignment ledger between the raw releases (XJTU / MIT) and the processed files, with explicit row mapping and a release gate |
| `experiments/make_paper_figures.py` | Generates Figures 1-4 of the manuscript from the archives |
| `experiments/dataset_summary.py` | Generates the archive behind Appendix A (Table A1) |
| `results/battery/*.json`, `*.jsonl` | Per-seed result archives behind every table and figure, including the revision's corrected and superseded versions and the experiments whose criteria were falsified |
| `residual_lab/` | Shared model/metric helpers used by some experiments |
| `tests/` | Unit tests, including the feature-legality tests (no future rows, no cross-cell leakage) |

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# 1. fetch the public datasets (not redistributed here) -- see DATA.md
# 2. run the tests (5 assert against the official splits and fail without the
#    datasets; 8 more are marked needs_data and skip cleanly when the CSVs are
#    missing)
.venv/bin/python -m pytest tests/ -q
# 3. re-run any experiment, or print its archived report, e.g.:
.venv/bin/python experiments/bexp49_matrclo_permutation.py --report
.venv/bin/python experiments/make_paper_figures.py
```

Every experiment script is idempotent: completed cells are read from its JSON
archive in `results/battery/`, so `--report` reproduces the manuscript numbers
without recomputation, and deleting an archive re-runs it from the raw data.
All experiments run on CPU; the frozen TabPFN v3 weights are read from a local
directory (`TABPFN_WEIGHTS_DIR`), and the strict-reader GPU variant (`bexp57`)
also supports CUDA.

## Mapping tables to archives

| Manuscript | Archive |
|---|---|
| Table 2, Figure 1 | `bexp42_cycle_convention.json`, `bexp52_importance.json` |
| Table 3 | `bexp33_2x2.json` |
| Table 4 | `bexp24_ablation.json` (family ablation); `bexp28_rsr.json` backs the Section 2.2 slimming-probe numbers |
| Table 5, Figure 2 | `bexp36_significance.json`, `bexp36_raw_baseline.json` |
| Table 6 (baseline ladder) | `bexp50_linear.jsonl` (pipeline A rows), `bexp56_strict_rerun.jsonl` (pipeline B rows, strict-online reader) |
| Table 7 (unit-level decomposition) | `bexp14_gbdt.jsonl`, `bexp50_linear.jsonl`, `bexp36_significance.json` |
| Table 8 | `bexp30_sohbenchmark.json`, `bexp31_our_method_bench_scenario.json` |
| Table 9 | `bexp41_matrclo_protocol.json`, `bexp49_matrclo_permutation.json` |
| Table 10 (Pipeline B, corrected reader) | `bexp56_strict_rerun_summary.json`, `bexp60_strict_tabpfn_cpu_summary.json` |
| Table 11 (simple baselines on raw curves) | `bexp63_b2_curve_scalar.jsonl`; the C reference discussed alongside is `bexp64_physical_ref.jsonl` |
| Table 12, Figure 3a | `bexp44_seeds20.json`, `bexp48_raw20.json`, `bexp48b_instrument20.json`, `bexp27_randomization.json`, `bexp43_tabpfn_perm.json` |
| Table 13 (archived single-run importance) | `p6_importance.json` |
| Table 14, Figure 3b | `bexp52_importance.json` (ten seeds) |
| Table 15 | `bexp34_missing_anchor.json`; Section 3.11, Figure 4a/b: `bexp45_learnable_ref.json`, `bexp45_stage2.json` |
| Section 4.5, Figure 4c | `bexp46_distill.json`, `bexp47_inject.json` |
| Section 2.4 context ladder | `bexp7_context_ladder.jsonl` |
| Appendix A, Table A1 | `dataset_summary.json` (Table A2, the ageing conditions, comes from the public dataset metadata; see the appendix note) |

## License

MIT -- see `LICENSE`. The archived JSON results are derived from third-party
open-access datasets; the datasets themselves are governed by their original
licenses (see `DATA.md`).

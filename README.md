# Deployable Self-Referenced Features for Cross-Cell Battery SOH Estimation

Code and result archives for the manuscript *"Deployable Self-Referenced
Features for Cross-Cell Battery State-of-Health Estimation: Causal Validation
and Benchmarking"* (submitted to *Energies*).

The self-referenced lift (SRL) augments per-cycle statistical features with the
offset from the same cell's first recorded cycle, `x_t - x_1`, making the
comparison against the manufacturing origin explicit and computable in service.
Every number in the manuscript can be re-derived from the archives in
`results/battery/`, and every experiment can be re-run from the public
datasets (see `DATA.md`).

## Layout

| Path | Content |
|---|---|
| `battery_lab/` | Data adapters (official splits), feature construction, evaluation protocols |
| `experiments/bexp*.py` | The 44 experiments of the study; 24 of them carry decision criteria (34 in total), frozen a priori in the script docstring (marked `冻结判据`, "frozen criteria"), the rest are descriptive or asset-generating |
| `experiments/make_paper_figures.py` | Generates Figures 1-4 of the manuscript from the archives |
| `experiments/dataset_summary.py` | Generates the archive behind Appendix A (Table A1) |
| `results/battery/*.json`, `bexp7_context_ladder.jsonl` | Per-seed result archives behind every table and the context-size ladder of Section 2.4, including the experiments whose criteria were falsified |
| `residual_lab/` | Shared model/metric helpers used by some experiments |
| `tests/` | Unit tests, including the feature-legality tests (no future rows, no cross-cell leakage) |

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# 1. fetch the public datasets (not redistributed here) -- see DATA.md
# 2. run the tests (5 of 44 require the datasets and fail without them)
.venv/bin/python -m pytest tests/ -q
# 3. re-run any experiment, or print its archived report, e.g.:
.venv/bin/python experiments/bexp49_matrclo_permutation.py --report
.venv/bin/python experiments/make_paper_figures.py
```

Every experiment script is idempotent: completed cells are read from its JSON
archive in `results/battery/`, so `--report` reproduces the manuscript numbers
without recomputation, and deleting an archive re-runs it from the raw data.
All experiments run on CPU (GBDT ~3 s per evaluation unit; the frozen TabPFN
takes 10-100 s per unit).

## Mapping tables to archives

| Manuscript | Archive |
|---|---|
| Table 2, Figure 1 | `bexp42_cycle_convention.json`, `p6_importance.json` |
| Table 3 | `bexp33_2x2.json` |
| Table 4 | `bexp24_ablation.json` (family ablation); `bexp28_rsr.json` backs the Section 2.2 slimming-probe numbers |
| Table 5, Figure 2 | `bexp36_significance.json`, `bexp36_raw_baseline.json` |
| Table 6 | `bexp30_sohbenchmark.json`, `bexp31_our_method_bench_scenario.json` |
| Table 7 | `bexp41_matrclo_protocol.json`, `bexp49_matrclo_permutation.json` |
| Table 8 | `bexp35_strict_mode.json` |
| Table 9, Figure 3a | `bexp44_seeds20.json`, `bexp48_raw20.json`, `bexp48b_instrument20.json`, `bexp27_randomization.json`, `bexp43_tabpfn_perm.json` |
| Table 10, Figure 3b | `p6_importance.json` |
| Table 11 | `bexp34_missing_anchor.json` |
| Section 3.9, Figure 4a/b | `bexp45_learnable_ref.json`, `bexp45_stage2.json` |
| Section 4.4, Figure 4c | `bexp46_distill.json`, `bexp47_inject.json` |
| Appendix A, Table A1 | `dataset_summary.json` |

## License

MIT -- see `LICENSE`. The archived JSON results are derived from third-party
open-access datasets; the datasets themselves are governed by their original
licenses (see `DATA.md`).

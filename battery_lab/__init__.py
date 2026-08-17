"""battery_lab: feature-view experiments for battery state-of-health estimation.

Three lines of evidence:
  1. Head-to-head replication on the four libraries (history-feature TabPFN
     against the published PINN4SOH per-experiment results)
  2. The interaction effect (history features help TabPFN, hurt an MLP)
  3. The view-selection protocol (nested selection inside the source cells plus
     frozen verification on an unexposed library)

Data source: external/PINN4SOH/data (the reference authors' preprocessed CSVs,
one file per cell, 16 statistics plus capacity, one row per cycle).
"""

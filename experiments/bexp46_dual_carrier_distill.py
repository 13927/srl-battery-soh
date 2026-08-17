"""bexp46: dual-carrier teacher-student distillation (the method-C experiment).

Motivation: the paper uses carrier duality only to explain a phenomenon (swapping
the cycle-column convention swings the error between -49.6 and +107.8 per cent),
not to build a method. This experiment tests whether the carrier information that
is available offline only can be distilled into a fully online-computable student.

Teacher (offline-only carrier):
    author_minmax normalisation + cycle_author_minmax (per-cell full-lifetime
    life-fraction) + the full history stack -> 81 dimensions. The cycle column
    reads the cell's complete lifetime length and is not computable at inference.
Student (online-computable):
    source normalisation (statistics from training cells only) + cycle_scaled
    (/200) + from_first -> 33 dimensions. Every column is computable while a cell
    is in service.

Distillation:
    student target = (1-lambda)*y + lambda*t_oof
    t_oof = the teacher's out-of-fold prediction on the training cells (GroupKFold
    by cell_id). Out-of-fold is required: an in-sample prediction would collapse
    to the label itself and carry no information to distil.

Legality (spell it out or it looks like leakage):
    the full trajectories of the training cells are in hand anyway -- the offline
    constraint applies only to the target cell at inference. Using life-fraction
    for the teacher during training is entirely legal; the student needs no future
    information about the target cell at inference. This is the learning-using-
    privileged-information (LUPI) paradigm, not data leakage.

================ frozen criteria (committed before the run) ================
Main lambda = 0.5. lambda in {0.3,0.7} is reported for sensitivity only and must
not be used to pick per unit.
Metric: macro-averaged per-cell RMSE (as Table 5 of the paper), GBDT 5 seeds,
paired subsampling.

C1 (main): distilled (lambda=0.5) beats online_only in >= 7 of 11 units, and the
         median improvement exceeds the seed noise floor of 11.3 per cent
C2 (mechanism): on units with headroom (teacher better than online_only by more
         than 11.3 per cent), distilled closes the online->teacher gap by a median
         of >= 30 per cent.
         Note: bexp42 already showed the teacher is not universally better; units
         without headroom are excluded from C2, but distilled must not be
         significantly worse on them.
C3 (causal consistency): mispaired is not better than distilled
         (the mispaired arm: the from_first anchor is taken from another cell in
         the same unit, lambda fixed at 0.5)

Refutation branch: if C1 fails -> the verdict is "carrier duality is explanatory
    but not exploitable", recorded in doc/battery/phase_method_probe.md and the
    Limitations; the paper is submitted as is, and no existing claim is changed
    because this experiment failed.
=======================================================
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import CONTEXT_CAP, fit_predict_gbdt, subsample_context
from battery_lab.temporal_features import (add_history_features,
                                           cycle_author_minmax, cycle_scaled)
from bexp27_randomization import derangement

OUT = ROOT / "results/battery/bexp46_distill.json"
SEEDS = [0, 1, 2, 3, 4]
LAMBDAS = [0.3, 0.5, 0.7]
LAMBDA_MAIN = 0.5
NOISE_FLOOR = 11.3          # frozen: median seed-to-seed variation (per cent)
DERANGE_SEED = 20260811


def teacher_matrix(cells, Xn_author):
    """Offline carrier: [16 statistics, cycle_author_minmax, 4x16 history] = 81
    dimensions."""
    xs, ys, ids = [], [], []
    for c in cells:
        Z = Xn_author[c.cell_id]
        xs.append(np.hstack([Z, cycle_author_minmax(len(Z)), add_history_features(Z)]))
        ys.append(c.y)
        ids.extend([c.cell_id] * len(c.y))
    return np.vstack(xs), np.concatenate(ys), np.array(ids)


def student_matrix(cells, Xn_source, ref_map=None):
    """Online carrier: [16 statistics, cycle_scaled, from_first] = 33 dimensions.

    When ref_map is not None the mispaired arm is built: the anchor is taken from
    another cell's first cycle, so the width and the arithmetic are unchanged and
    only the identity of the reference is broken.
    """
    xs, ys, ids = [], [], []
    for c in cells:
        Z = Xn_source[c.cell_id]
        anchor = Z[:1] if ref_map is None else Xn_source[ref_map[c.cell_id]][:1]
        xs.append(np.hstack([Z, cycle_scaled(len(Z)), Z - anchor]))
        ys.append(c.y)
        ids.extend([c.cell_id] * len(c.y))
    return np.vstack(xs), np.concatenate(ys), np.array(ids)


def cell_macro_rmse(y, pred, ids):
    return float(np.mean([np.sqrt(np.mean((y[ids == c] - pred[ids == c]) ** 2))
                          for c in np.unique(ids)]))


def teacher_oof(X_tr, y_tr, ids_tr, seed):
    """The teacher's out-of-fold prediction on the training cells (grouped by
    cell, so a fold never sees its own cell)."""
    from sklearn.model_selection import GroupKFold

    n_cells = len(np.unique(ids_tr))
    gkf = GroupKFold(n_splits=min(5, n_cells))
    oof = np.zeros_like(y_tr)
    for tr, va in gkf.split(X_tr, y_tr, groups=ids_tr):
        oof[va] = fit_predict_gbdt(X_tr[tr], y_tr[tr], X_tr[va], seed)
    return oof


def run_unit(unit):
    """Return the per-cell RMSE of every arm on this unit (averaged over seeds)."""
    cells = unit.train_cells + unit.test_cells
    Xn_author = normalize_cells(cells, "author_minmax")
    Xn_source = normalize_cells(cells, "source", source_cells=unit.train_cells)
    ref_map = derangement([c.cell_id for c in unit.cells], DERANGE_SEED)

    # the row order is aligned across all arms, for paired subsampling
    Xt_tr, y_tr, ids_tr = teacher_matrix(unit.train_cells, Xn_author)
    Xt_te, y_te, ids_te = teacher_matrix(unit.test_cells, Xn_author)
    Xs_tr, _, _ = student_matrix(unit.train_cells, Xn_source)
    Xs_te, _, _ = student_matrix(unit.test_cells, Xn_source)
    Xm_tr, _, _ = student_matrix(unit.train_cells, Xn_source, ref_map)
    Xm_te, _, _ = student_matrix(unit.test_cells, Xn_source, ref_map)

    arms = {k: [] for k in ["teacher", "online_only", "mispaired"]}
    arms.update({f"distilled{lam}": [] for lam in LAMBDAS})

    for s in SEEDS:
        oof = teacher_oof(Xt_tr, y_tr, ids_tr, s)
        # the same row indices are used by every arm -> a paired comparison
        idx = np.arange(len(y_tr))
        if len(idx) > CONTEXT_CAP:
            idx = np.random.default_rng(s).choice(len(idx), CONTEXT_CAP, replace=False)

        def score(X_tr_arm, X_te_arm, target):
            pred = fit_predict_gbdt(X_tr_arm[idx], target[idx], X_te_arm, s)
            return cell_macro_rmse(y_te, pred, ids_te)

        arms["teacher"].append(score(Xt_tr, Xt_te, y_tr))
        arms["online_only"].append(score(Xs_tr, Xs_te, y_tr))
        for lam in LAMBDAS:
            arms[f"distilled{lam}"].append(
                score(Xs_tr, Xs_te, (1 - lam) * y_tr + lam * oof))
        arms["mispaired"].append(
            score(Xm_tr, Xm_te, (1 - LAMBDA_MAIN) * y_tr + LAMBDA_MAIN * oof))

    return {k: {"rmse": float(np.mean(v)), "seed_vals": v} for k, v in arms.items()}


def main():
    state = json.load(OUT.open()) if OUT.exists() else {}
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in [u for u, _, _ in FIT_UNITS]:
        if uid in state:
            continue
        t0 = time.time()
        state[uid] = run_unit(load_fit_unit(*umap[uid]))
        json.dump(state, OUT.open("w"), indent=1)
        r = state[uid]
        d = (r[f"distilled{LAMBDA_MAIN}"]["rmse"] - r["online_only"]["rmse"]) \
            / r["online_only"]["rmse"] * 100
        print(f"{uid:<16} online={r['online_only']['rmse']:.5f} "
              f"distill={r[f'distilled{LAMBDA_MAIN}']['rmse']:.5f} ({d:+.1f}%) "
              f"teacher={r['teacher']['rmse']:.5f} [{time.time()-t0:.0f}s]", flush=True)
    report(state)


def report(state=None):
    state = state or json.load(OUT.open())
    main_key = f"distilled{LAMBDA_MAIN}"
    print(f"\n{'='*88}\ndual-carrier distillation (GBDT, {len(SEEDS)} seeds, "
          f"main lambda={LAMBDA_MAIN})  metric: macro per-cell RMSE")
    print(f"{'unit':<16}{'online':>10}{'distill':>10}{'delta%':>8}"
          f"{'teacher':>10}{'head%':>8}{'closed%':>8}{'mispair':>10}")
    wins, deltas, closes, mis_better = 0, [], [], 0
    n = 0
    for uid, _, _ in FIT_UNITS:
        if uid not in state:
            continue
        r = state[uid]
        on, di, te = (r["online_only"]["rmse"], r[main_key]["rmse"],
                      r["teacher"]["rmse"])
        mi = r["mispaired"]["rmse"]
        d = (di - on) / on * 100
        head = (on - te) / on * 100                 # teacher headroom over the online baseline
        close = (on - di) / (on - te) * 100 if head > NOISE_FLOOR else float("nan")
        n += 1
        wins += int(di < on)
        deltas.append(d)
        if not np.isnan(close):
            closes.append(close)
        mis_better += int(mi < di)
        print(f"{uid:<16}{on:>10.5f}{di:>10.5f}{d:>+8.1f}{te:>10.5f}"
              f"{head:>+8.1f}{close:>8.1f}{mi:>10.5f}"
              if not np.isnan(close) else
              f"{uid:<16}{on:>10.5f}{di:>10.5f}{d:>+8.1f}{te:>10.5f}"
              f"{head:>+8.1f}{'  none':>8}{mi:>10.5f}")

    med_d = float(np.median(deltas)) if deltas else float("nan")
    c1 = wins >= 7 and med_d < -NOISE_FLOOR
    med_c = float(np.median(closes)) if closes else float("nan")
    c2 = bool(closes) and med_c >= 30
    c3 = mis_better <= n // 2

    # per-unit Mann-Whitney + Holm, the same method as the significance section
    from scipy.stats import mannwhitneyu
    raw_p, dsign = [], []
    for uid, _, _ in FIT_UNITS:
        if uid not in state:
            continue
        r = state[uid]
        raw_p.append(float(mannwhitneyu(r[main_key]["seed_vals"],
                                        r["online_only"]["seed_vals"],
                                        alternative="two-sided").pvalue))
        dsign.append(r[main_key]["rmse"] - r["online_only"]["rmse"])
    order = sorted(range(len(raw_p)), key=lambda i: raw_p[i])
    holm, run_max = [0.0] * len(raw_p), 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, raw_p[i] * (len(raw_p) - rank))
        holm[i] = min(1.0, run_max)
    sig_better = sum(ph < 0.05 and d < 0 for ph, d in zip(holm, dsign))
    sig_worse = sum(ph < 0.05 and d > 0 for ph, d in zip(holm, dsign))
    print(f"\nMann-Whitney + Holm（distilled λ={LAMBDA_MAIN} vs online_only, "
          f"{len(SEEDS)} vs {len(SEEDS)} seeds)")
    for uid, p_r, p_h in zip([u for u, _, _ in FIT_UNITS if u in state],
                             raw_p, holm):
        print(f"  {uid:<16} p_raw={p_r:.4f}  p_holm={p_h:.4f}")
    print(f"  after Holm: significantly better {sig_better}/{len(raw_p)}, "
          f"significantly worse {sig_worse}/{len(raw_p)}")

    print(f"\n{'-'*88}")
    print(f"C1 units where distilled beats online = {wins}/{n} (criterion >=7); "
          f"median delta% = {med_d:+.1f}% (criterion < -{NOISE_FLOOR}) -> "
          + ("holds" if c1 else "falsified"))
    print(f"C2 units with headroom {len(closes)}/{n}, median gap closed = "
          f"{med_c:.1f}% (criterion >=30) -> " + ("holds" if c2 else "falsified")
          if closes else f"C2 no unit with headroom >{NOISE_FLOOR}% -> not applicable "
          f"(the teacher has nothing to transfer)")
    print(f"C3 units where mispaired beats distilled = {mis_better}/{n} "
          f"(criterion <= half) -> "
          + ("holds" if c3 else "falsified -- causal consistency fails"))

    print(f"\nlambda sensitivity (reported only, not used to select)")
    for lam in LAMBDAS:
        ds = [(state[u][f"distilled{lam}"]["rmse"] - state[u]["online_only"]["rmse"])
              / state[u]["online_only"]["rmse"] * 100
              for u, _, _ in FIT_UNITS if u in state]
        w = sum(x < 0 for x in ds)
        print(f"  lambda={lam}: median delta% {np.median(ds):+.1f}%  "
              f"beats online {w}/{len(ds)}")

    verdict = ("pass -> proceed to stage 2 (more seeds / other backbones / their deep baselines)"
               if c1 and c3 else
               "refuted -> carrier duality is explanatory but not exploitable; submit the paper as is")
    print(f"\nverdict: {verdict}")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif "--smoke" in sys.argv:
        umap = {u: (d, b) for u, d, b in FIT_UNITS}
        uid = "XJTU-2C"
        unit = load_fit_unit(*umap[uid])
        cells = unit.train_cells + unit.test_cells
        Xn_s = normalize_cells(cells, "source", source_cells=unit.train_cells)
        Xs, _, _ = student_matrix(unit.train_cells, Xn_s)
        Xt, y, ids = teacher_matrix(unit.train_cells,
                                    normalize_cells(cells, "author_minmax"))
        print(f"student matrix {Xs.shape} (should be 33d)  "
              f"teacher matrix {Xt.shape} (should be 81d)")
        # the student input must contain no full-lifetime quantity: the cycle
        # column must be the arithmetic sequence t/200
        c0 = unit.train_cells[0]
        col = Xs[:len(c0.y), 16]
        ok = np.allclose(np.diff(col), 1 / 200.0)
        print(f"student cycle column is the fixed scale t/200: "
              + ("yes" if ok else "no")
              + f" (first={col[0]:.5f} last={col[-1]:.5f})")
        # the first from_first row must be exactly zero
        print(f"student from_first first row all zero: "
              f"{'✓' if np.allclose(Xs[0, 17:], 0) else '✗'}")
        oof = teacher_oof(Xt, y, ids, 0)
        print(f"teacher OOF prediction RMSE={np.sqrt(np.mean((y-oof)**2)):.5f} "
              f"(should be clearly above 0, otherwise in-sample leakage occurred)")
        t0 = time.time()
        r = run_unit(unit)
        print(f"\n{uid} all arms on one unit took {time.time()-t0:.0f}s")
        for k, v in r.items():
            print(f"  {k:<16} {v['rmse']:.5f}")
    else:
        main()

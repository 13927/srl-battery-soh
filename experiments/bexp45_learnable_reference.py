"""bexp45: a learnable reference layer (the method-A experiment).

Motivation: the paper shows that the choice of reference is a first-class design
variable (the best reference flips with the normalisation convention, -6.7% <->
+4.5%), but the method itself hard-codes the anchor to the first cycle. This
experiment asks whether letting the model learn which early cycles to anchor on
can beat the hard-coded first cycle.

Reference layer:
    score each of the cell's first k cycles, softmax to weights, combine into an
    anchor
        s_i = v^T tanh(W z_i)      i = 1..k
        alpha = softmax(s)
        r = sum_i alpha_i * z_i
    features = [ z_t , c_t , z_t - r ]

Online legality: only the first k cycles are used (k=10 fixed, far below any
cell's life), so they are always available in service.

================ design note (fixed after the smoke test, then frozen) ================
Division of labour: the gradient only learns the reference, GBDT judges the
accuracy. Two measured facts motivate this:

(1) only 4 of the 11 units have >= 20 training cells (TJU-1 53, TJU-2 44, HUST 57,
    MIT 102); the other 7 have only 6-12, so a 20% per-cell validation split holds
    just 1-2 cells and early stopping amounts to random model selection
    (measured: on XJTU-2C the val loss is best at epoch 1 while the train loss
    drops to 7.5e-5, overfitting at once).
    -> the reference layer is trained on a fixed epoch budget, with no
       validation-based model selection.
(2) the MLP head is not only worse on units with few cells but points the other
    way: the project's own sklearn-MLP baseline on XJTU-2C gives raw=0.039 /
    fixed_srl=0.096 (the lift makes it worse by +144%), XJTU-3C +216%; whereas
    GBDT on the same data is +0.4% / -14.2%. On units with enough cells the two
    agree (TJU-1 MLP -29.4% vs GBDT -26.0%).
    -> judging a reference construction with a backbone that is itself harmed by
       the reference family is uninformative. The accuracy comparison must return
       to the model families of the paper (GBDT / TabPFN).

Therefore: first learn alpha on the training cells by gradient (fixed budget, no
early stopping), then feed the learned anchor r into GBDT and compare per unit
against the hard-coded first-cycle SRL, over all 11 units, 5 seeds. The
end-to-end MLP-head numbers are reported only on the 4 cell-rich units, as a
secondary result.

================ frozen criteria (committed before the run) ================
Metric: macro per-cell RMSE, GBDT 5 seeds. Noise floor 11.3% (the project's frozen
value). The comparison must be against fixed_srl; beating raw alone is not a pass.

A1 (accuracy type A-i): learnable_ref beats fixed_srl in >= 7 of 11 units
    and the median delta% <= -11.3%
    -> a learnable reference brings a real accuracy gain, the method holds, a
       separate second paper
A2 (automation type A-ii): |median delta%| < 11.3% (within the noise floor)
    and the median weight alpha gives the first cycle > 0.6
    -> the model discovered the right reference on its own, corroborating the
       paper's causal conclusion; the contribution is the removal of manual design
       and its interpretability, not accuracy; folded into the paper's Section 3.9
A3 (refutation A-iii): median delta% >= +11.3%
    -> the reference must be given by domain knowledge, written as a boundary
       condition; the paper is submitted as is

A2' (self-anchoring type, added after the smoke test, alongside A2):
    |median delta%| < 11.3%
    and alpha is not concentrated on the first cycle (alpha_1 <= 0.6)
    and the learned anchor departs clearly from the first cycle (max|r - z_1| > 0.5
    on most units)
    -> reading: the active ingredient is "anchoring to the cell's own early state",
       not "the first cycle itself". This runs with the paper's mispairing control
       (swapping in another cell breaks it, but which of one's own early cycles is
       used is not sensitive), so it strengthens the causal claim rather than being
       a new accuracy claim.
       Placement: one short paragraph in Section 3.9, changing no existing number.

If neither A2 nor A2' holds beyond the three tiers, treat it conservatively as
A-iii and make no accuracy claim. Report whichever tier the result falls into; the
criteria may not be changed after the fact.

The two descriptive statistics of alpha (argmax position, mass of the first 3
cycles) are reported only, not used as criteria.

================ stage 2 power hardening (added after the verdict, committed before the run) ================
Trigger: A2' holds (it did). Following stage-2 item 1 of the spec, "for a passing
criterion, raise seeds to 20 (matching the paper's power standard) and add a second
backbone".

Why it is necessary: Section 3.9 now carries a null-hypothesis claim ("accuracy is
statistically indistinguishable from the hard-coded first cycle"). A null claim
needs more power, not less, and the decisive experiment had only 3 reference seeds
x 5 GBDT seeds. There is precedent in the project: raising the three arms from 5 to
20 seeds in bexp44 flipped the unit-level win count from 9/11 to 8/11. The same
risk applies here.

Design: reuse the same learned references (3 reference seeds) with the backbone
changed to
  - GBDT at 20 seeds (matching the bexp44 power standard)
  - TabPFN at 3 seeds (matching the bexp43 TabPFN convention; the reference seed is
    fixed at 0 to keep the cost down)
Statistics: per-unit Mann-Whitney U (learnable vs fixed_srl over the seed samples)
plus Holm correction, the same method as the significance section.

================ frozen criteria (committed before the run) ================
S1 (maintained): under 20-seed GBDT, |median delta%| is still < 11.3% and the
    number of units with a Holm-corrected significant difference is <= 2/11 ->
    the "statistically indistinguishable" wording of Section 3.9 stands
S2 (overturned to accuracy type): median delta% <= -11.3% and lower in >= 7/11
    units -> Section 3.9 must be rewritten as an accuracy claim, and the case for a
    separate paper reassessed
S3 (overturned to refutation): median delta% >= +11.3%, or the number of
    Holm-corrected significantly worse units is >= 4/11
    -> Section 3.9 must be rewritten as a boundary condition ("a learnable
       reference is worse than the domain-given first cycle")
The second backbone is only a consistency check: if the TabPFN side disagrees in
    direction with the GBDT side, Section 3.9 must state "this conclusion is not
    consistent across the two backbones" and not report only the favourable side.
=======================================================
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import CONTEXT_CAP, fit_predict_gbdt, subsample_context
from battery_lab.temporal_features import cycle_author_minmax

OUT = ROOT / "results/battery/bexp45_learnable_ref.json"
GBDT_SEEDS = [0, 1, 2, 3, 4]
REF_SEEDS = [0, 1, 2]              # reference-layer training seeds (alpha averaged over the three)
K_EARLY = 10
NOISE_FLOOR = 11.3
REF_EPOCHS, BATCH, LR = 100, 256, 1e-3          # fixed budget, no early stopping
MIN_CELLS_FOR_HEAD = 20                          # min training cells for the end-to-end MLP head
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class RefLearner(nn.Module):
    """Reference layer plus a light head. The head only supplies a training signal
    to the reference layer; it takes no part in the accuracy verdict."""

    def __init__(self, d):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(d, 32), nn.Tanh(), nn.Linear(32, 1))
        self.head = nn.Sequential(nn.Linear(2 * d + 1, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def alpha(self, early):                       # early: (B, k, d)
        return torch.softmax(self.scorer(early).squeeze(-1), dim=1)

    def forward(self, z, c, early):
        r = torch.einsum("bk,bkd->bd", self.alpha(early), early)
        return self.head(torch.cat([z, c, z - r], dim=1)).squeeze(-1), r


def pack_unit(unit, k):
    """Return (z, c, early, y, ids) for (train, test); z/c/early are standardised
    with training-row statistics."""
    Xn = normalize_cells(unit.train_cells + unit.test_cells, "author_minmax")

    def pack(cells):
        zs, cs, es, ys, ids = [], [], [], [], []
        for cell in cells:
            Z = Xn[cell.cell_id]
            n = len(Z)
            zs.append(Z)
            cs.append(cycle_author_minmax(n))
            es.append(np.repeat(Z[:k][None], n, axis=0))
            ys.append(cell.y)
            ids.extend([cell.cell_id] * n)
        return (np.vstack(zs), np.vstack(cs), np.concatenate(es, axis=0),
                np.concatenate(ys), np.array(ids))

    tr, te = pack(unit.train_cells), pack(unit.test_cells)
    mu, sd = tr[0].mean(0), tr[0].std(0) + 1e-8
    mc, sc = tr[1].mean(0), tr[1].std(0) + 1e-8
    std = lambda p: ((p[0] - mu) / sd, (p[1] - mc) / sc, (p[2] - mu) / sd, p[3], p[4])
    return std(tr), std(te)


def learn_reference(tr, te, seed):
    """Train the reference layer on a fixed budget; return (mean alpha, train
    anchor, test anchor, final train MSE)."""
    torch.manual_seed(seed)
    z, c, e, y, _ = tr
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    Z, C, E, Y = T(z), T(c), T(e), T(y)
    model = RefLearner(z.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    n = len(Y)
    for _ in range(REF_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            pred, _ = model(Z[b], C[b], E[b])
            lossf(pred, Y[b]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        _, r_tr = model(Z, C, E)
        a_tr = model.alpha(E).mean(0)
        pred, r_te = model(T(te[0]), T(te[1]), T(te[2]))
        final = float(lossf(model(Z, C, E)[0], Y))
    return (a_tr.cpu().numpy(), r_tr.cpu().numpy(), r_te.cpu().numpy(), final,
            float(np.mean([np.sqrt(np.mean((te[3][te[4] == cid]
                                           - pred.cpu().numpy()[te[4] == cid]) ** 2))
                           for cid in np.unique(te[4])])))


def gbdt_score(X_tr, y_tr, X_te, y_te, ids_te, seed):
    Xc, yc = subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)
    pred = fit_predict_gbdt(Xc, yc, X_te, seed)
    return float(np.mean([np.sqrt(np.mean((y_te[ids_te == c] - pred[ids_te == c]) ** 2))
                          for c in np.unique(ids_te)]))


def run_unit(unit):
    k = min(K_EARLY, min(len(c.y) for c in unit.cells))
    tr, te = pack_unit(unit, k)
    z, c, e, y, _ = tr
    z2, c2, e2, y2, ids2 = te

    alphas, head_rmse, r_pairs, devs = [], [], [], []
    for s in REF_SEEDS:
        a, r_tr, r_te, final, hr = learn_reference(tr, te, s)
        alphas.append(a)
        head_rmse.append(hr)
        r_pairs.append((r_tr, r_te))
        devs.append(float(np.abs(r_tr - e[:, 0, :]).max()))   # how far the anchor departs from the first cycle

    feats = {
        "raw": (np.hstack([z, c]), np.hstack([z2, c2])),
        "fixed_srl": (np.hstack([z, c, z - e[:, 0, :]]),
                      np.hstack([z2, c2, z2 - e2[:, 0, :]])),
    }
    out = {"k_early": k, "n_train_cells": len(unit.train_cells),
           "anchor_dev_from_first": float(np.mean(devs)),
           "alpha_mean": np.mean(alphas, axis=0).tolist(),
           "alpha_per_seed": [a.tolist() for a in alphas],
           "head_rmse": {"rmse": float(np.mean(head_rmse)), "seed_vals": head_rmse,
                         "eligible": len(unit.train_cells) >= MIN_CELLS_FOR_HEAD}}

    for name, (Xa, Xb) in feats.items():
        vals = [gbdt_score(Xa, y, Xb, y2, ids2, s) for s in GBDT_SEEDS]
        out[name] = {"rmse": float(np.mean(vals)), "seed_vals": vals}

    # learnable reference: one anchor set per reference seed, each with the GBDT
    # seeds, all averaged
    vals = []
    for r_tr, r_te in r_pairs:
        Xa = np.hstack([z, c, z - r_tr])
        Xb = np.hstack([z2, c2, z2 - r_te])
        vals += [gbdt_score(Xa, y, Xb, y2, ids2, s) for s in GBDT_SEEDS]
    out["learnable_ref"] = {"rmse": float(np.mean(vals)), "seed_vals": vals}
    return out


def main():
    state = json.load(OUT.open()) if OUT.exists() else {}
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    print(f"device = {DEV}   reference layer, fixed budget {REF_EPOCHS} epochs, no early stopping")
    for uid in [u for u, _, _ in FIT_UNITS]:
        if uid in state:
            continue
        t0 = time.time()
        state[uid] = run_unit(load_fit_unit(*umap[uid]))
        json.dump(state, OUT.open("w"), indent=1)
        r = state[uid]
        d = (r["learnable_ref"]["rmse"] - r["fixed_srl"]["rmse"]) / r["fixed_srl"]["rmse"] * 100
        print(f"{uid:<16} raw={r['raw']['rmse']:.5f} fixed={r['fixed_srl']['rmse']:.5f} "
              f"learn={r['learnable_ref']['rmse']:.5f} ({d:+.1f}%) "
              f"α₁={r['alpha_mean'][0]:.2f} [{time.time()-t0:.0f}s]", flush=True)
    report(state)


def report(state=None):
    state = state or json.load(OUT.open())
    print(f"\n{'='*92}\nlearnable reference layer "
          f"({len(REF_SEEDS)} reference seeds x {len(GBDT_SEEDS)} GBDT seeds, "
          f"k={K_EARLY})  metric: macro per-cell RMSE")
    print(f"{'unit':<16}{'raw':>10}{'fixed_srl':>11}{'learnable':>11}"
          f"{'d vs fixed':>12}{'a(cyc1)':>10}{'a top3':>9}{'a argmax':>9}"
          f"{'|r-z₁|':>9}")
    wins, deltas, a1s, devs, n = 0, [], [], [], 0
    for uid, _, _ in FIT_UNITS:
        if uid not in state:
            continue
        r = state[uid]
        fx, lr_ = r["fixed_srl"]["rmse"], r["learnable_ref"]["rmse"]
        d = (lr_ - fx) / fx * 100
        a = r["alpha_mean"]
        n += 1
        wins += int(lr_ < fx)
        deltas.append(d)
        a1s.append(a[0])
        dev = r.get("anchor_dev_from_first", float("nan"))
        devs.append(dev)
        print(f"{uid:<16}{r['raw']['rmse']:>10.5f}{fx:>11.5f}{lr_:>11.5f}"
              f"{d:>+12.1f}{a[0]:>10.2f}{sum(a[:3]):>9.2f}"
              f"{int(np.argmax(a))+1:>8}{dev:>9.2f}")

    med_d = float(np.median(deltas))
    med_a1 = float(np.median(a1s))
    med_dev = float(np.nanmedian(devs)) if devs else float("nan")
    a_i = wins >= 7 and med_d <= -NOISE_FLOOR
    a_iii = med_d >= NOISE_FLOOR
    in_floor = (not a_i) and (not a_iii)
    a_ii = in_floor and med_a1 > 0.6
    a_ii_p = in_floor and med_a1 <= 0.6 and med_dev > 0.5

    print(f"\n{'-'*92}")
    print(f"units where learnable beats fixed_srl = {wins}/{n};  "
          f"median delta% = {med_d:+.1f}%;  "
          f"median alpha(cycle 1) = {med_a1:.2f}  (uniform would be {1/K_EARLY:.2f})")
    print(f"A1 (A-i accuracy, >=7/11 and delta<=-{NOISE_FLOOR}%): "
          + ("holds" if a_i else "no"))
    print(f"A2 (A-ii automation, |delta|<{NOISE_FLOOR}% and alpha1>0.6): "
          + ("holds" if a_ii else "no"))
    print(f"A3 (A-iii refutation, delta>=+{NOISE_FLOOR}%): "
          + ("holds" if a_iii else "no"))
    print(f"A2' (self-anchoring, |delta|<{NOISE_FLOOR}% and alpha1<=0.6 and "
          f"median|r-z1|>0.5): " + ("holds" if a_ii_p else "no")
          + f"  (median|r-z1|={med_dev:.2f})")

    elig = [(u, state[u]) for u, _, _ in FIT_UNITS
            if u in state and state[u]["head_rmse"]["eligible"]]
    if elig:
        print(f"\nsecondary result: end-to-end MLP head (only the {len(elig)} units "
              f"with >={MIN_CELLS_FOR_HEAD} training cells; the rest lack a "
              f"validation set and are structurally out of scope)")
        for u, r in elig:
            print(f"  {u:<16} end-to-end={r['head_rmse']['rmse']:.5f}  "
                  f"(same unit, GBDT + learnable reference={r['learnable_ref']['rmse']:.5f})")

    verdict = ("A-i accuracy -> a separate second paper" if a_i else
               "A-ii automation (alpha converges to cycle 1) -> folded into Section 3.9" if a_ii else
               "A-ii' self-anchoring (the anchor departs from the first cycle but the "
               "effect is the same) -> strengthens the causal claim, one short "
               "paragraph in Section 3.9, no accuracy claim" if a_ii_p else
               "A-iii refutation -> the reference must be domain-given, submit as is" if a_iii else
               "beyond the three tiers -> treat conservatively as A-iii, no accuracy claim")
    print(f"\nverdict: {verdict}")


# ============================ stage 2: power hardening ============================
OUT2 = ROOT / "results/battery/bexp45_stage2.json"
GBDT_SEEDS20 = list(range(20))
TAB_SEEDS = [0, 1, 2]


def run_unit_stage2(unit, backbone):
    """The same learned references, re-evaluated with another backbone and more
    seeds."""
    from battery_lab.protocols import fit_predict_tabpfn

    k = min(K_EARLY, min(len(c.y) for c in unit.cells))
    tr, te = pack_unit(unit, k)
    z, c, e, y, _ = tr
    z2, c2, e2, y2, ids2 = te

    if backbone == "gbdt":
        fit, seeds, ref_seeds = fit_predict_gbdt, GBDT_SEEDS20, REF_SEEDS
    else:
        fit, seeds, ref_seeds = fit_predict_tabpfn, TAB_SEEDS, [0]

    def score(X_tr, X_te, seed):
        Xc, yc = subsample_context(X_tr, y, CONTEXT_CAP, seed)
        pred = fit(Xc, yc, X_te, seed)
        return float(np.mean([np.sqrt(np.mean((y2[ids2 == cid] - pred[ids2 == cid]) ** 2))
                              for cid in np.unique(ids2)]))

    out = {"backbone": backbone, "n_ref_seeds": len(ref_seeds)}
    for name, Xa, Xb in [
            ("raw", np.hstack([z, c]), np.hstack([z2, c2])),
            ("fixed_srl", np.hstack([z, c, z - e[:, 0, :]]),
             np.hstack([z2, c2, z2 - e2[:, 0, :]]))]:
        vals = [score(Xa, Xb, s) for s in seeds]
        out[name] = {"rmse": float(np.mean(vals)), "seed_vals": vals}

    vals = []
    for rs in ref_seeds:
        _, r_tr, r_te, _, _ = learn_reference(tr, te, rs)
        Xa = np.hstack([z, c, z - r_tr])
        Xb = np.hstack([z2, c2, z2 - r_te])
        vals += [score(Xa, Xb, s) for s in seeds]
    out["learnable_ref"] = {"rmse": float(np.mean(vals)), "seed_vals": vals}
    return out


def main_stage2(backbone):
    state = json.load(OUT2.open()) if OUT2.exists() else {}
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    n_seeds = len(GBDT_SEEDS20) if backbone == "gbdt" else len(TAB_SEEDS)
    print(f"stage 2 power hardening: backbone={backbone}, {n_seeds} seeds, device={DEV}")
    for uid in [u for u, _, _ in FIT_UNITS]:
        key = f"{uid}|{backbone}"
        if key in state:
            continue
        t0 = time.time()
        state[key] = run_unit_stage2(load_fit_unit(*umap[uid]), backbone)
        json.dump(state, OUT2.open("w"), indent=1)
        r = state[key]
        d = (r["learnable_ref"]["rmse"] - r["fixed_srl"]["rmse"]) / r["fixed_srl"]["rmse"] * 100
        print(f"{uid:<16} fixed={r['fixed_srl']['rmse']:.5f} "
              f"learn={r['learnable_ref']['rmse']:.5f} ({d:+.1f}%) "
              f"[{time.time()-t0:.0f}s]", flush=True)
    report_stage2(state)


def report_stage2(state=None):
    from scipy.stats import mannwhitneyu

    state = state or json.load(OUT2.open())
    for backbone in ("gbdt", "tabpfn"):
        rows = [(u, state[f"{u}|{backbone}"]) for u, _, _ in FIT_UNITS
                if f"{u}|{backbone}" in state]
        if not rows:
            continue
        n_seeds = len(rows[0][1]["fixed_srl"]["seed_vals"])
        print(f"\n{'='*88}\nstage 2: {backbone} {n_seeds} seeds  "
              f"learnable_ref vs fixed_srl（Mann-Whitney + Holm）")
        print(f"{'unit':<16}{'fixed':>10}{'learnable':>11}{'delta%':>9}"
              f"{'p_raw':>10}{'p_holm':>10}")
        raw_p, deltas, wins = [], [], 0
        for u, r in rows:
            fx, lr_ = r["fixed_srl"]["rmse"], r["learnable_ref"]["rmse"]
            d = (lr_ - fx) / fx * 100
            p = float(mannwhitneyu(r["learnable_ref"]["seed_vals"],
                                   r["fixed_srl"]["seed_vals"],
                                   alternative="two-sided").pvalue)
            raw_p.append(p)
            deltas.append(d)
            wins += int(lr_ < fx)
        order = sorted(range(len(raw_p)), key=lambda i: raw_p[i])
        holm = [0.0] * len(raw_p)
        run_max = 0.0
        for rank, i in enumerate(order):
            run_max = max(run_max, raw_p[i] * (len(raw_p) - rank))
            holm[i] = min(1.0, run_max)
        for (u, r), d, p, ph in zip(rows, deltas, raw_p, holm):
            print(f"{u:<16}{r['fixed_srl']['rmse']:>10.5f}"
                  f"{r['learnable_ref']['rmse']:>11.5f}{d:>+9.1f}{p:>10.4f}{ph:>10.4f}")
        med = float(np.median(deltas))
        sig_worse = sum(ph < 0.05 and d > 0 for ph, d in zip(holm, deltas))
        sig_any = sum(ph < 0.05 for ph in holm)
        print(f"\nmedian delta% = {med:+.1f}%   units lower = {wins}/{len(rows)}   "
              f"Holm significant = {sig_any}/{len(rows)} (of which worse {sig_worse})")
        if backbone == "gbdt":
            s1 = abs(med) < NOISE_FLOOR and sig_any <= 2
            s2 = med <= -NOISE_FLOOR and wins >= 7
            s3 = med >= NOISE_FLOOR or sig_worse >= 4
            print(f"S1 maintained (|delta|<{NOISE_FLOOR}% and Holm significant<=2/11): "
                  + ("holds -- Section 3.9 wording stands" if s1 else "no"))
            print("S2 overturned to accuracy type: "
                  + ("holds -- Section 3.9 must be rewritten" if s2 else "no"))
            print("S3 overturned to refutation: "
                  + ("holds -- Section 3.9 must be rewritten" if s3 else "no"))


if __name__ == "__main__":
    if "--stage2" in sys.argv:
        bk = "tabpfn" if "--tabpfn" in sys.argv else "gbdt"
        main_stage2(bk)
    elif "--report2" in sys.argv:
        report_stage2()
    elif "--report" in sys.argv:
        report()
    elif "--smoke" in sys.argv:
        umap = {u: (d, b) for u, d, b in FIT_UNITS}
        for uid in ["XJTU-2C", "TJU-1"]:
            unit = load_fit_unit(*umap[uid])
            k = min(K_EARLY, min(len(c.y) for c in unit.cells))
            tr, te = pack_unit(unit, k)
            print(f"\n{uid}: k={k} training cells={len(unit.train_cells)} "
                  f"z={tr[0].shape} early={tr[2].shape}")
            Xf = np.hstack([tr[0], tr[1], tr[0] - tr[2][:, 0, :]])
            print(f"  fixed_srl first-row diff all zero: "
                  + ("yes" if np.allclose(Xf[0, 17:], 0) else "no"))
            print(f"  early window constant per cell: "
                  f"{'✓' if np.allclose(tr[2][0], tr[2][len(unit.train_cells[0].y)-1]) else '✗'}")
            t0 = time.time()
            a, r_tr, r_te, final, hr = learn_reference(tr, te, 0)
            print(f"  reference layer trained in {time.time()-t0:.0f}s  "
                  f"final train MSE={final:.2e}  end-to-end rmse={hr:.5f}")
            print(f"  alpha = {[round(float(x), 3) for x in a]}  (sum={a.sum():.3f}, "
                  f"argmax at cycle {int(np.argmax(a))+1})")
            print(f"  does the anchor collapse to the first cycle: max|r - z_1| = "
                  f"{np.abs(r_tr - tr[2][:, 0, :]).max():.4f}")
            t0 = time.time()
            out = run_unit(unit)
            print(f"  all arms on one unit {time.time()-t0:.0f}s -> raw={out['raw']['rmse']:.5f} "
                  f"fixed={out['fixed_srl']['rmse']:.5f} "
                  f"learn={out['learnable_ref']['rmse']:.5f}")
    else:
        main()

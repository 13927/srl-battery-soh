"""bexp6: manuscript assets -- summary tables and figures.

Outputs:
  doc/battery/figures/interaction_matrix.png    heat map of the interaction
                                               (2 backbones x 11 units)
  doc/battery/figures/forest_history_tabpfn.png per-unit forest plot, history vs
                                               raw
  doc/battery/tables/*.md                       Markdown tables
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from battery_lab.data_adapters import FIT_UNITS

R = Path("results/battery")
FIG = Path("doc/battery/figures"); FIG.mkdir(parents=True, exist_ok=True)
TAB = Path("doc/battery/tables"); TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = ["Arial Unicode MS", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def load_bexp1():
    rows = [json.loads(l) for l in (R / "bexp1_author_protocol.jsonl").read_text().splitlines()]
    agg = {}
    for uid, _, _ in FIT_UNITS:
        for bb in ("tabpfn", "mlp"):
            m = {}
            for view in ("raw", "history"):
                v = [r["rmse_cell_macro"] for r in rows if r["unit"] == uid
                     and r["view"] == view and r["backbone"] == bb]
                m[view] = float(np.mean(v)) if v else np.nan
            agg[(uid, bb)] = (m["history"] - m["raw"]) / m["raw"] * 100
    return agg


def fig_interaction(agg):
    units = [u for u, _, _ in FIT_UNITS]
    data = np.array([[agg[(u, "tabpfn")], agg[(u, "mlp")]] for u in units])
    fig, ax = plt.subplots(figsize=(6, 7))
    vmax = 60
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["TabPFN", "MLP"])
    ax.set_yticks(range(len(units))); ax.set_yticklabels(units, fontsize=8)
    for i in range(len(units)):
        for j in range(2):
            v = data[i, j]
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 30 else "black")
    ax.set_title("history vs raw, relative difference in per cent "
                 "(negative favours history)\n"
                 "interaction: helps TabPFN on most units, hurts the MLP "
                 "everywhere")
    fig.colorbar(im, ax=ax, label="Δ RMSE %")
    fig.tight_layout()
    fig.savefig(FIG / "interaction_matrix.png", dpi=150)
    plt.close(fig)


def fig_forest():
    rows = [json.loads(l) for l in (R / "bexp1_author_protocol.jsonl").read_text().splitlines()]
    units = [u for u, _, _ in FIT_UNITS]
    means, los, his = [], [], []
    for uid in units:
        pairs = {}
        for r in rows:
            if r["unit"] == uid and r["backbone"] == "tabpfn":
                pairs.setdefault(r["view"], []).append(r["rmse_cell_macro"])
        rd = np.array(pairs["history"]) - np.array(pairs["raw"])
        rng = np.random.default_rng(0)
        boots = [rd[rng.integers(0, len(rd), len(rd))].mean() for _ in range(2000)]
        means.append(rd.mean() * 1000)
        los.append(np.percentile(boots, 2.5) * 1000)
        his.append(np.percentile(boots, 97.5) * 1000)
    y = np.arange(len(units))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.errorbar(means, y, xerr=[np.array(means) - np.array(los),
                                np.array(his) - np.array(means)],
                fmt="o", capsize=3, color="tab:blue")
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(units, fontsize=8)
    ax.set_xlabel("history - raw, macro RMSE x1000 "
                  "(negative favours history, 5 seeds)")
    ax.set_title("per-unit effect of the history view on TabPFN "
                 "(reference protocol)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "forest_history_tabpfn.png", dpi=150)
    plt.close(fig)


def table_four_lib():
    rows = [json.loads(l) for l in (R / "bexp1_author_protocol.jsonl").read_text().splitlines()]
    user_ref = {"HUST": 0.009070, "MIT": 0.007372, "TJU": 0.009676, "XJTU": 0.007530}
    lines = ["# Four-library replication (TabPFN + history vs the published "
             "PINN4SOH results)\n",
             "| library | reproduced TabPFN + history | previously reported | "
             "difference |", "|---|---|---|---|"]
    for lib, ref in user_ref.items():
        rs = [r for r in rows if r["unit"].startswith(lib)
              and r["view"] == "history" and r["backbone"] == "tabpfn"]
        if lib == "XJTU":
            um = {}
            for r in rs:
                um.setdefault(r["unit"], []).append(r["rmse_cell_macro"])
            val = float(np.mean([np.mean(v) for v in um.values()]))
        elif lib == "TJU":
            us = {}
            for r in rs:
                us.setdefault(r["unit"], []).append((r["rmse_overall"], r["n_test_rows"]))
            vals = [np.mean([x[0] for x in l]) for l in us.values()]
            ws = [l[0][1] for l in us.values()]
            val = float(np.average(vals, weights=ws))
        else:
            val = float(np.mean([r["rmse_overall"] for r in rs]))
        lines.append(f"| {lib} | {val:.6f} | {ref:.6f} | {(val-ref)/ref*100:+.2f}% |")
    (TAB / "four_lib_alignment.md").write_text("\n".join(lines) + "\n")


def main():
    agg = load_bexp1()
    fig_interaction(agg)
    fig_forest()
    table_four_lib()
    print("written:")
    for p in sorted(FIG.glob("*.png")) + sorted(TAB.glob("*.md")):
        print(" ", p)


if __name__ == "__main__":
    main()

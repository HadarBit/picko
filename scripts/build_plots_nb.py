#!/usr/bin/env python3
"""Build notebooks/research/nb_plots.ipynb — a LOCAL, GPU-free notebook that reads
the saved results JSON (breadth / depth / separation) and renders paper-ready,
consistently-styled figures: Breadth curve, Depth bars, Separation bars, and all
confusion matrices in one organized subplot grid.

Run:  python scripts/build_plots_nb.py
"""
import json, os

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "notebooks", "research")


def md(s): return {"cell_type": "markdown", "metadata": {}, "id": None, "source": s}
def co(s): return {"cell_type": "code", "metadata": {}, "id": None, "execution_count": None, "outputs": [], "source": s}


cells = [
 md("""# PICKO Research · Figures — paper-ready plots from saved results

Runs **locally, no GPU** — reads the results JSON each notebook saved and re-draws
publication-quality, consistently-styled figures. Point the three paths below at your
JSON files (local downloads, or the Drive `picko_out/nb1|nb2|nb3` folders)."""),
 md("## 1 · Style & configuration\\nOne shared style (Okabe-Ito colourblind-safe palette, thin spines, subtle grid) is applied to every figure so the set reads as one system."),
 co('''import json, os
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "regular",
    "axes.labelsize": 12, "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    "axes.grid": False, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3, "legend.frameon": False, "legend.fontsize": 10,
})
BLUE, ORANGE, GREEN, VERM, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#8a8a8a"

# <- edit to your files
NB1_RESULTS = "breadth_results.json"
NB2_RESULTS = "depth_results.json"
NB3_RESULTS = "separation_results.json"
SAVE_DIR    = "."

def finish(ax, yaxis_grid=True, xaxis_grid=False):
    sns.despine(ax=ax)
    if yaxis_grid: ax.grid(axis="y", color="#cccccc", lw=0.6, alpha=0.6)
    if xaxis_grid: ax.grid(axis="x", color="#cccccc", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)'''),
 md("## 2 · Breadth — selection accuracy vs. number of tools offered"),
 co('''runs = pd.DataFrame(json.load(open(NB1_RESULTS)))
agg = (runs.groupby("k").agg(mean=("selection_acc", "mean"), std=("selection_acc", "std"),
                             n_visible=("n_visible", "median")).reset_index())
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.fill_between(agg["k"], agg["mean"]-agg["std"], agg["mean"]+agg["std"], color=BLUE, alpha=0.12)
ax.plot(agg["k"], agg["mean"], "-", color=BLUE, lw=1.8, zorder=3)
ax.scatter(runs["k"], runs["selection_acc"], s=16, color=BLUE, alpha=0.20, zorder=1, linewidths=0)
ax.errorbar(agg["k"], agg["mean"], yerr=agg["std"], fmt="o", color=BLUE, ms=6, capsize=3, lw=1.2,
            zorder=4, label="selection accuracy (mean ± s.d.)")
wall = agg[agg["n_visible"] < agg["k"]]
if len(wall):
    kw, vw = int(wall["k"].iloc[0]), int(wall["n_visible"].iloc[0])
    ax.axvline(kw, color=VERM, ls=(0, (4, 3)), lw=1.3)
    ax.text(kw-0.6, 0.28, f"encoder truncation\\n(~{vw} of {kw} tools visible)",
            color=VERM, fontsize=9, ha="right", va="center")
ax.set_ylim(0, 1.0); ax.set_xlabel("Number of tools offered at inference, $k$")
ax.set_ylabel("Tool-selection accuracy")
ax.set_title("Breadth: selection accuracy vs. number of tools offered")
ax.legend(loc="lower left"); finish(ax)
fig.tight_layout(); fig.savefig(os.path.join(SAVE_DIR, "fig_breadth.png")); plt.show()'''),
 md("## 3 · Depth — parameter extraction vs. parameter count\\n`param_f1` is undefined for 0-parameter tools (no parameters to score), so it is shown as **n/a** rather than a misleading 0; `args_exact_acc` is the meaningful metric there."),
 co('''depth = pd.DataFrame(json.load(open(NB2_RESULTS)))
ORDER = ["0", "1", "2-3", "4+"]
per_iter = (depth.groupby(["iteration", "param_bucket"])[["args_exact_acc", "param_f1"]]
            .mean().reset_index())
agg = (per_iter.groupby("param_bucket")
       .agg(args_mean=("args_exact_acc", "mean"), args_std=("args_exact_acc", "std"),
            pf1_mean=("param_f1", "mean"), pf1_std=("param_f1", "std"),
            n_iter=("iteration", "nunique")).reindex([b for b in ORDER if b in per_iter["param_bucket"].values]))
if "0" in agg.index: agg.loc["0", ["pf1_mean", "pf1_std"]] = np.nan   # param_f1 undefined at 0 params

x = np.arange(len(agg)); w = 0.38
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.bar(x-w/2, agg["args_mean"], w, yerr=agg["args_std"].fillna(0), capsize=3, color=BLUE,
       edgecolor="white", lw=0.5, label="args_exact_acc (all-or-nothing)", error_kw=dict(lw=1, ecolor="#444"))
ax.bar(x+w/2, agg["pf1_mean"].fillna(0), w, yerr=agg["pf1_std"].fillna(0), capsize=3, color=ORANGE,
       edgecolor="white", lw=0.5, label="param_f1 (partial credit)", error_kw=dict(lw=1, ecolor="#444"))
for _, r in per_iter.iterrows():
    if r["param_bucket"] in list(agg.index):
        ax.scatter(list(agg.index).index(r["param_bucket"])-w/2, r["args_exact_acc"],
                   s=14, color="#20364a", alpha=0.7, zorder=5, linewidths=0)
if "0" in agg.index:
    ax.text(list(agg.index).index("0")+w/2, 0.02, "n/a\\n(0 params)", ha="center", va="bottom",
            fontsize=8, color=GREY)
ax.set_xticks(x); ax.set_xticklabels(agg.index)
ax.set_ylim(0, 1.05); ax.set_xlabel("Number of parameters (bucket)")
ax.set_ylabel("Extraction accuracy")
ax.set_title(f"Depth: parameter extraction vs. parameter count ({int(agg['n_iter'].max())} iterations)")
ax.legend(loc="upper right"); finish(ax)
fig.tight_layout(); fig.savefig(os.path.join(SAVE_DIR, "fig_depth.png")); plt.show()'''),
 md("## 4 · Separation — load results & axis labels"),
 co('''sep = json.load(open(NB3_RESULTS))
per_group = pd.DataFrame(sep["per_group"]).sort_values("selection_acc")
conf = sep["confusion"]
AXIS = {"cross_source_search": "cross-source", "single_item_summary": "cross-source",
        "hf_search_variants": "within-source", "wikipedia_retrieve_vs_summarize": "within-source",
        "get_paper_content": "within-source", "arxiv_latex": "within-source"}
display(per_group.assign(axis=per_group["group"].map(AXIS)).round(3))'''),
 md("## 5 · Separation — bar chart"),
 co('''fig, ax = plt.subplots(figsize=(6.6, 4.2))
colors = [VERM if AXIS.get(g) == "within-source" else GREEN for g in per_group["group"]]
bars = ax.barh(per_group["group"], per_group["selection_acc"], color=colors, edgecolor="white", lw=0.5)
for b, v in zip(bars, per_group["selection_acc"]):
    ax.text(v+0.012, b.get_y()+b.get_height()/2, f"{v:.2f}", va="center", fontsize=10)
ax.set_xlim(0, 1.08); ax.set_xlabel("Tool-selection accuracy")
ax.set_title("Separation: disambiguation of look-alike tool groups")
ax.legend(handles=[Patch(color=GREEN, label="cross-source"),
                   Patch(color=VERM, label="within-source")], loc="lower right")
finish(ax, yaxis_grid=False, xaxis_grid=True)
fig.tight_layout(); fig.savefig(os.path.join(SAVE_DIR, "fig_separation_bars.png")); plt.show()'''),
 md("## 6 · Separation — confusion matrices in one grid\\nCounts annotated; shading is **row-normalised** (share of each true tool's queries): diagonal = correct rate, off-diagonal = where it leaks."),
 co('''def conf_matrix(cmap_dict):
    labels = sorted(set(cmap_dict) | {p for row in cmap_dict.values() for p in row})
    M = pd.DataFrame(0, index=labels, columns=labels)
    for r, row in cmap_dict.items():
        for p, c in row.items(): M.loc[r, p] = c
    return M

def short(name): return name.split("_", 1)[1] if "_" in name else name

order = per_group.sort_values("selection_acc", ascending=False)["group"].tolist()
ncol = 3; nrow = int(np.ceil(len(order)/ncol))
fig = plt.figure(figsize=(6.3*ncol, 5.4*nrow))
gs = gridspec.GridSpec(nrow, ncol, figure=fig, hspace=0.6, wspace=0.55)
for i, g in enumerate(order):
    ax = fig.add_subplot(gs[i//ncol, i%ncol])
    M = conf_matrix(conf[g]); acc = per_group.set_index("group").loc[g, "selection_acc"]
    Mn = M.div(M.sum(axis=1).replace(0, 1), axis=0)
    sns.heatmap(Mn, cmap="Blues", vmin=0, vmax=1, cbar=False, linewidths=0.5, linecolor="white",
                ax=ax, xticklabels=[short(c) for c in M.columns], yticklabels=[short(r) for r in M.index])
    for yi in range(M.shape[0]):
        for xi in range(M.shape[1]):
            ax.text(xi+0.5, yi+0.5, int(M.values[yi, xi]), ha="center", va="center",
                    fontsize=11, color="white" if Mn.values[yi, xi] > 0.5 else "#333")
    ax.set_title(f"{g}\\n{AXIS.get(g,'')} · acc = {acc:.2f}", fontsize=12)
    ax.set_xlabel("predicted", fontsize=10); ax.set_ylabel("reference", fontsize=10)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9)
    for lbl in ax.get_xticklabels(): lbl.set_ha("right")
fig.suptitle("Separation: confusion matrices (counts; shading = row-normalised share)", fontsize=15, y=0.995)
fig.savefig(os.path.join(SAVE_DIR, "fig_confusion_grid.png")); plt.show()'''),
]


def write():
    for j, c in enumerate(cells):
        c["id"] = f"c{j:02d}"
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = os.path.join(OUTDIR, "nb_plots.ipynb")
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path, "·", len(cells), "cells")


if __name__ == "__main__":
    write()

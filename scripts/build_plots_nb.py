#!/usr/bin/env python3
"""Build notebooks/research/nb_plots.ipynb — a LOCAL, GPU-free notebook that reads
the saved results JSON (breadth_results.json / separation_results.json) and renders
publication-quality figures: the Breadth curve, the Separation bar chart, and all
confusion matrices in one organized subplot grid.

Run:  python scripts/build_plots_nb.py
"""
import json, os

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "notebooks", "research")


def md(s): return {"cell_type": "markdown", "metadata": {}, "id": None, "source": s}
def co(s): return {"cell_type": "code", "metadata": {}, "id": None, "execution_count": None, "outputs": [], "source": s}


cells = [
 md("""# PICKO Research · Plots — pretty figures from saved results

Runs **locally, no GPU** — reads the results JSON each notebook saved and re-draws
publication-quality figures. Point `NB1_RESULTS` / `NB3_RESULTS` at your JSON files
(local copies or the Drive `picko_out/nb1`, `picko_out/nb3` folders)."""),
 md("## 1 · Setup"),
 co('''import json, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style="white", context="talk")
plt.rcParams.update({"axes.titleweight": "semibold", "figure.dpi": 120,
                     "savefig.dpi": 160, "font.size": 11})

# <- edit these to your files (local downloads or the Drive picko_out subfolders)
NB1_RESULTS = "breadth_results.json"
NB3_RESULTS = "separation_results.json"
SAVE_DIR    = "."          # where PNGs are written
BLUE = "#3b6fb0"'''),
 md("## 2 · Breadth curve (mean ± std over random tool subsets)"),
 co('''runs = pd.DataFrame(json.load(open(NB1_RESULTS)))
agg = (runs.groupby("k")
       .agg(mean=("selection_acc", "mean"), std=("selection_acc", "std"),
            n_visible=("n_visible", "median")).reset_index())

fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(agg["k"], agg["mean"]-agg["std"], agg["mean"]+agg["std"], color=BLUE, alpha=0.15)
ax.plot(agg["k"], agg["mean"], "-", color=BLUE, lw=2, zorder=3)
ax.scatter(runs["k"], runs["selection_acc"], s=22, color=BLUE, alpha=0.18, zorder=1)  # each repeat
ax.errorbar(agg["k"], agg["mean"], yerr=agg["std"], fmt="o", color=BLUE, ms=8,
            capsize=4, lw=1.5, zorder=4, label="selection_acc (mean ± std)")
wall = agg[agg["n_visible"] < agg["k"]]
if len(wall):
    kw, vw = int(wall["k"].iloc[0]), int(wall["n_visible"].iloc[0])
    ax.axvline(kw, color="#c0504d", ls=":", lw=1.6)
    ax.text(kw-0.4, 0.30, f"truncation wall\\n~{vw} of {kw} tools visible",
            color="#c0504d", fontsize=10, ha="right", va="center")
ax.set_ylim(0, 1.0); ax.set_xlabel("# tools offered at inference (k)")
ax.set_ylabel("tool-selection accuracy")
ax.set_title("Breadth — selection accuracy vs. number of tools offered")
ax.legend(loc="lower left", frameon=False)
sns.despine(ax=ax); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(SAVE_DIR, "pretty_breadth.png")); plt.show()'''),
 md("## 3 · Separation — load results & axis labels"),
 co('''sep = json.load(open(NB3_RESULTS))
per_group = pd.DataFrame(sep["per_group"]).sort_values("selection_acc")
conf = sep["confusion"]
AXIS = {"cross_source_search": "cross-source", "single_item_summary": "cross-source",
        "hf_search_variants": "within-source", "wikipedia_retrieve_vs_summarize": "within-source",
        "get_paper_content": "within-source", "arxiv_latex": "within-source"}
display(per_group.assign(axis=per_group["group"].map(AXIS)).round(3))'''),
 md("## 4 · Separation bar chart (green = cross-source, red = within-source)"),
 co('''fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#c0504d" if AXIS.get(g) == "within-source" else "#4C8045" for g in per_group["group"]]
bars = ax.barh(per_group["group"], per_group["selection_acc"], color=colors)
for b, v in zip(bars, per_group["selection_acc"]):
    ax.text(v+0.01, b.get_y()+b.get_height()/2, f"{v:.2f}", va="center", fontsize=11)
ax.set_xlim(0, 1.05); ax.set_xlabel("tool-selection accuracy")
ax.set_title("Separation — hardest look-alike groups (lower = more confused)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#4C8045", label="cross-source"),
                   Patch(color="#c0504d", label="within-source")], loc="lower right", frameon=False)
sns.despine(ax=ax); ax.grid(axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(SAVE_DIR, "pretty_separation_bars.png")); plt.show()'''),
 md("## 5 · Confusion matrices — one organized grid\\nCount annotated; cell shading is **row-normalized** (share of each true tool's queries), so the diagonal = correct rate and off-diagonal = where it leaks."),
 co('''def conf_matrix(cmap_dict):
    labels = sorted(set(cmap_dict) | {p for row in cmap_dict.values() for p in row})
    M = pd.DataFrame(0, index=labels, columns=labels)
    for r, row in cmap_dict.items():
        for p, c in row.items(): M.loc[r, p] = c
    return M

def short(name):  # compact tick labels: drop the family prefix
    return name.split("_", 1)[1] if "_" in name else name

order = per_group.sort_values("selection_acc", ascending=False)["group"].tolist()
n = len(order); ncol = 3; nrow = int(np.ceil(n/ncol))
fig = plt.figure(figsize=(6.3*ncol, 5.5*nrow))
gs = gridspec.GridSpec(nrow, ncol, figure=fig, hspace=0.55, wspace=0.55)
for i, g in enumerate(order):
    ax = fig.add_subplot(gs[i // ncol, i % ncol])
    M = conf_matrix(conf[g])
    acc = per_group.set_index("group").loc[g, "selection_acc"]
    Mn = M.div(M.sum(axis=1).replace(0, 1), axis=0)
    sns.heatmap(Mn, cmap="Blues", vmin=0, vmax=1, cbar=False, linewidths=0.5,
                linecolor="white", ax=ax,
                xticklabels=[short(c) for c in M.columns], yticklabels=[short(r) for r in M.index])
    for yi in range(M.shape[0]):          # manual counts, contrast-safe on dark cells
        for xi in range(M.shape[1]):
            ax.text(xi+0.5, yi+0.5, int(M.values[yi, xi]), ha="center", va="center",
                    fontsize=11, color="white" if Mn.values[yi, xi] > 0.5 else "#333")
    ax.set_title(f"{g}\\n{AXIS.get(g,'')} · acc={acc:.2f}", fontsize=12)
    ax.set_xlabel("predicted", fontsize=10); ax.set_ylabel("reference", fontsize=10)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9)
    for lbl in ax.get_xticklabels(): lbl.set_ha("right")
fig.suptitle("Separation — confusion matrices (count annotated, row-normalized shading)",
             fontsize=16, fontweight="semibold", y=0.99)
fig.savefig(os.path.join(SAVE_DIR, "pretty_confusion_grid.png"), bbox_inches="tight"); plt.show()'''),
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

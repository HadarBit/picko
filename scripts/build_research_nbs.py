#!/usr/bin/env python3
"""Build the 3 PICKO research notebooks (Breadth / Depth / Separation) in
notebooks/research/ with shared, restart-safe init cells.

Run:  python scripts/build_research_nbs.py   (regenerates all three notebooks)

Design goals baked into the generated cells:
  * Colab-aligned to Google Drive `MyDrive/picko/` — data in, checkpoints+results out
    (so a runtime restart loses nothing), tuned for an L4 GPU.
  * Run-and-forget: every sweep is resumable (finished sizes/iterations/groups are
    skipped from Drive) and fault-tolerant (one failure is logged and the loop goes on).
  * Observability: timestamped `log()` lines to screen AND a durable Drive run.log,
    per-step timing, checkpoint-write confirmation, and a one-glance env/GPU header.
"""
import json
import os

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "notebooks", "research")
os.makedirs(OUTDIR, exist_ok=True)


def md(s): return {"cell_type": "markdown", "metadata": {}, "id": None, "source": s}
def co(s): return {"cell_type": "code", "metadata": {}, "id": None, "execution_count": None, "outputs": [], "source": s}


# ---------- shared init cells (identical across the 3 notebooks) ----------
BOOTSTRAP_MD = md("""## 0 · Colab quick-start (GPU) — run & forget, restart-safe

**On Colab first: Runtime → Change runtime type → GPU (L4 recommended; T4/A100 also fine).**
This cell clones the repo, pins the exact JAX/Flax, mounts Drive, and points **both** the data (in) and
the checkpoints+results (out) at your **`MyDrive/picko/`** folder — so a runtime restart loses nothing.

**Prerequisite (one-time):** `picko_balanced.jsonl` must be in `MyDrive/picko/`. **Running locally?** This
cell is a no-op — skip to cell 1.""")

BOOTSTRAP = co('''# --- Colab bootstrap (safe to re-run; no-op locally) ---
import os, sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.exists("/content/picko"):
        !git clone -b hadar-work https://github.com/HadarBit/picko.git /content/picko
    %pip install -q "jax[cuda12]==0.10.2" "jaxlib==0.10.2" "flax==0.12.8"
    sys.path.insert(0, "/content/picko")
    from google.colab import drive; drive.mount("/content/drive")
    import shutil
    DRIVE = "/content/drive/MyDrive/picko"                      # <- everything lives here
    os.environ["PICKO_OUT_DIR"] = f"{DRIVE}/picko_out"          # checkpoints + results (durable)
    os.environ["PICKO_LOG"]     = f"{DRIVE}/picko_out/run.log"  # durable log across restarts
    os.makedirs(os.environ["PICKO_OUT_DIR"], exist_ok=True)
    dst = "/content/picko/data/picko_balanced.jsonl"
    if not os.path.exists(dst):
        cands = [f"{DRIVE}/picko_balanced.jsonl", "/content/drive/MyDrive/picko_balanced.jsonl"]
        src = next((c for c in cands if os.path.exists(c)), None)
        if src is None:
            have = os.listdir(DRIVE) if os.path.isdir(DRIVE) else "(MyDrive/picko not found)"
            raise FileNotFoundError(
                "picko_balanced.jsonl not found. Upload it to MyDrive/picko/. "
                f"Currently in {DRIVE}: {have}")
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(src, dst)
        print("copied data from", src)
    import jax
    print("GPU:");
    !nvidia-smi -L
    print("jax devices:", jax.devices())
    _plat = jax.devices()[0].platform
    assert _plat == "gpu", (
        f"JAX is running on '{_plat}', NOT the GPU — every finetune/eval will be ~30x slower "
        "(hours instead of minutes). FIX: Runtime > Change runtime type > GPU (L4), then "
        "Runtime > Restart session, and re-run this cell. If a GPU IS selected but this still "
        "fails, the CUDA plugin didn't load — re-run the %pip line above, then restart.")
    print("bootstrap OK · GPU active · data =", dst, "· OUT_DIR =", os.environ["PICKO_OUT_DIR"])
else:
    print("Not on Colab — running locally (CPU).")''')

SETUP_MD = md("## 1 · Setup & data overview")

SETUP = co('''# ensure the repo root is importable (works from notebooks/research/, Colab, etc.)
import os, sys
_here = os.path.abspath(os.getcwd())
for _ in range(6):
    if os.path.exists(os.path.join(_here, "scripts", "picko_research.py")): break
    _here = os.path.dirname(_here)
if os.path.isdir("/content/picko"): _here = "/content/picko"
if _here not in sys.path: sys.path.insert(0, _here)

from scripts.picko_research import *
import json, time
import pandas as pd, numpy as np, matplotlib.pyplot as plt
try:
    import seaborn as sns; sns.set_theme(style="whitegrid")
except Exception:
    sns = None
from tqdm.auto import tqdm

cat, tok, raw, FOCUS, OUT_DIR = load_context()
env_report(OUT_DIR)   # jax devices + is OUT_DIR durable (Drive)?''')

DF1_MD = md("### The 40 focus tools\\nOne row per tool, with its family, category and **parameter count / bucket**.")
DF1 = co('display(tools_dataframe(cat, FOCUS))')

DF2_MD = md("### All examples for these 40 tools\\nOne row per training example (query → gold tool), tagged with the gold tool's **param bucket**.")
DF2 = co('''ex_df = examples_dataframe(cat, raw, FOCUS)
print("examples:", ex_df.shape[0], "| per param bucket:", ex_df["param_bucket"].value_counts().to_dict())
display(ex_df.head(10))''')


def init_cells(title_md):
    return [md(title_md), BOOTSTRAP_MD, BOOTSTRAP, SETUP_MD, SETUP, DF1_MD, DF1, DF2_MD, DF2]


# ======================================================================
# NB1 — Breadth (amount)
# ======================================================================
nb1 = init_cells("""# PICKO Research · NB1 — **Breadth**: how many tools before selection breaks?

We finetune **one** model on the 40 focus tools and then, on its **held-out test set**, offer each query a
random subset of `k` tools (its gold tool + `k-1` distractors) and measure tool-selection accuracy. Training
is fixed, so the curve isolates a single variable — *how many tools are offered at inference* — averaged
over `N_REPEATS` random subsets per `k` (mean ± std). The 1024-token encoder truncates long compact lists,
so past ~20 tools some are never seen; that ceiling is the result, annotated with `n_visible`.""")
nb1 += [
 md("""## 2 · Train / test split

Both the training subprocess and this notebook call the **same deterministic** `per_tool_split`
(`seed=42`, 10 test + 10 val per tool). The model trains **only on the train split**; the sweep below runs
**only on the held-out test split**, so no test query is ever seen in training."""),
 md("## 3 · Configure"),
 co('''NB_DIR = os.path.join(OUT_DIR, "nb1"); os.makedirs(NB_DIR, exist_ok=True)   # this notebook's outputs
BREADTH_SIZES  = [3, 5, 10, 20, 30, 40]   # tools offered per query at inference
N_REPEATS      = 8                        # random subsets averaged per size (mean +/- std)
CAP_PER_TOOL   = 120                       # examples/tool -> 100 train / 10 val / 10 test
EPOCHS         = 1
EVAL_SUBSAMPLE = 100                       # test queries per (size, repeat), sampled across all tools; None = full
MAX_GEN_LEN    = 64
BATCH_SIZE     = 8                         # lower to 4 on OOM, raise to 16 if headroom
RUN_TRAIN      = True
FORCE_RETRAIN  = False
print("focus tools:", len(FOCUS), "| sizes:", BREADTH_SIZES, "| repeats/size:", N_REPEATS, "| out:", NB_DIR)'''),
 md("## 4 · Train the model (40 focus tools, compact)\\nTrained once to Drive and reused; the returned `test` set is the held-out split used by the sweep."),
 co('''FOCUS40 = finetune_and_eval(cat, raw, tok, FOCUS, "breadth_focus40", NB_DIR,
                            cap=CAP_PER_TOOL, epochs=EPOCHS, compact=True, offer_all=len(FOCUS),
                            eval_subsample=None, run_train=RUN_TRAIN, force_retrain=FORCE_RETRAIN,
                            max_gen_len=MAX_GEN_LEN, batch_size=BATCH_SIZE)
m40, p40, tk40 = FOCUS40["bundle"]
TEST = FOCUS40["test"]                     # held-out test queries (never trained on)
log(f"model ready · held-out test queries={len(TEST)}")'''),
 md("## 5 · Sweep: offer k tools per query, repeat, average\\nInference only, on the held-out test set. Resumable: finished `(k, repeat)` pairs persist to `nb1/breadth_results.json`."),
 co('''import contextlib, io
RES = os.path.join(NB_DIR, "breadth_results.json")
rows = json.load(open(RES)) if (os.path.exists(RES) and not FORCE_RETRAIN) else []
done = {(r["k"], r["repeat"]) for r in rows}
if done: log(f"resumed {len(done)} finished (k,repeat) run(s)")

def gold_of(ex):
    calls = json.loads(ex.get("answers", "[]"))
    return next((c["name"] for c in calls if isinstance(c, dict) and c.get("name")), None)

t_all = time.time()
for k in BREADTH_SIZES:
    for rep in range(N_REPEATS):
        if (k, rep) in done and not FORCE_RETRAIN: continue
        try:
            test = TEST[:EVAL_SUBSAMPLE] if EVAL_SUBSAMPLE else TEST
            offered = [offer_subset(cat, gold_of(e), FOCUS, k, seed=k*100000+rep*1000+i)
                       for i, e in enumerate(test)]
            with contextlib.redirect_stdout(io.StringIO()):
                preds = predict(m40, p40, tk40, test, tools_override=offered, max_gen_len=MAX_GEN_LEN)
            met = evaluate(test, preds, family_of=family_of)
            vis = int(np.median([n_visible(e["query"], json.loads(o), tok) for e, o in zip(test, offered)]))
            rows = [r for r in rows if not (r["k"] == k and r["repeat"] == rep)] + [{
                "k": k, "repeat": rep, "selection_acc": met["selection_acc"],
                "name_f1": met["name_f1"], "parse_rate": met["parse_rate"],
                "n_visible": vis, "n_test": met["n"]}]
            done.add((k, rep)); json.dump(rows, open(RES, "w"), indent=2)
            log(f"k={k} rep={rep}: selection={met['selection_acc']:.3f} visible={vis}/{k}")
        except Exception as e:
            log(f"k={k} rep={rep}: FAILED ({type(e).__name__}: {e})")

log(f"ALL RUNS DONE in {time.time()-t_all:.0f}s · results={RES}")
runs = pd.DataFrame(rows)
breadth = (runs.groupby("k")
           .agg(selection_mean=("selection_acc", "mean"), selection_std=("selection_acc", "std"),
                name_f1_mean=("name_f1", "mean"), parse_rate=("parse_rate", "mean"),
                n_visible=("n_visible", "median"), n_repeats=("repeat", "nunique"),
                n_test=("n_test", "sum")).reset_index())
breadth["selection_std"] = breadth["selection_std"].fillna(0)
display(breadth.round(3))'''),
 md("## 6 · The Breadth curve"),
 co('''fig, ax = plt.subplots(figsize=(8,4.5))
ax.errorbar(breadth["k"], breadth["selection_mean"], yerr=breadth["selection_std"],
            fmt="o-", color="#4C72B0", capsize=4, label="selection_acc (mean +/- std)")
ax.scatter(runs["k"], runs["selection_acc"], s=12, color="#4C72B0", alpha=0.25, zorder=1)
wall = breadth[breadth["n_visible"] < breadth["k"]]
if len(wall):
    kw = int(wall["k"].iloc[0]); vw = int(wall["n_visible"].iloc[0])
    ax.axvline(kw, color="#C44E52", ls=":", lw=1.5)
    ax.text(kw, 0.06, f" truncation wall\\n (~{vw} of {kw} tools visible)", color="#C44E52", fontsize=9, va="bottom")
ax.set_xlabel("# tools offered at inference (k)"); ax.set_ylabel("tool-selection accuracy")
ax.set_ylim(0,1.02); ax.set_title(f"Breadth: selection vs #tools offered ({int(breadth['n_repeats'].max())} subsets/size)")
ax.legend()
plt.tight_layout(); save_fig("breadth_curve", out_dir=NB_DIR); plt.show()'''),
 md("""## 7 · Read-out

One model, probed on held-out queries with random subsets of `k` tools, so the curve reflects the *offered*
tool count alone — the error bars show the spread a single sample per size would hide. Selection stays high
for small `k` and falls as `k` grows past the red line, where the compact offered list overflows the
1024-token encoder and the extra tools are truncated away. **Takeaway:** one PICKO instance is bounded by
the context window it can offer, not by what it was trained on — beyond the wall, shard the tool set."""),
]

# ======================================================================
# NB2 — Depth (parameters), repeated stratified sampling
# ======================================================================
nb2 = init_cells("""# PICKO Research · NB2 — **Depth**: is parameter extraction harder with more parameters?

40 full-schema tools can't all be offered at once (token limit), and a single model gives one
unreplicated score per tool. Instead we **repeatedly sample a small, bucket-balanced set** (tools from
every param-count bucket, sized to fit the encoder), finetune, and measure argument extraction — over
several iterations — so each bucket gets many measurements and we can show **error bars**.

*Run & forget:* each iteration trains once to Drive; a restart **skips finished iterations** and keeps
the collected per-tool rows in `picko_out/depth_results.json`.""")
nb2 += [
 md("## 2 · Configure the repeated sampling\\nEach iteration draws `TOOLS_PER_BUCKET` tools from **each** bucket (0 / 1 / 2-3 / 4+) into one small model."),
 co('''NB_DIR = os.path.join(OUT_DIR, "nb2"); os.makedirs(NB_DIR, exist_ok=True)   # this notebook's outputs
N_ITER          = 5     # <- number of independent (tool-sample + finetune) iterations
TOOLS_PER_BUCKET = 2     # tools drawn from each param bucket per iteration
CAP_PER_TOOL     = 120   # examples/tool -> 100 train / 10 val / 10 test
EPOCHS           = 1
EVAL_SUBSAMPLE   = 40
BATCH_SIZE       = 8     # lower to 4 on OOM, raise to 16 if headroom
RUN_TRAIN        = True
FORCE_RETRAIN    = False
print("param buckets available:", tools_dataframe(cat, FOCUS)["param_bucket"].value_counts().to_dict(), "| out:", NB_DIR)'''),
 md("## 3 · Run the iterations\\n*Resumable:* finished iterations are skipped; per-tool rows persist to `nb2/depth_results.json` after each iteration."),
 co('''RES = os.path.join(NB_DIR, "depth_results.json")
rows = json.load(open(RES)) if (os.path.exists(RES) and not FORCE_RETRAIN) else []
done_iters = {r["iteration"] for r in rows}
if done_iters: log(f"loaded {len(done_iters)} finished iteration(s) from {RES}")

t_all = time.time()
for i in range(N_ITER):
    ckpt = os.path.join(NB_DIR, f"picko_depth_iter{i}_best.pkl")
    if (i in done_iters) and os.path.exists(ckpt) and not FORCE_RETRAIN:
        log(f"iter {i}: skip (already done)"); continue
    try:
        names = sample_stratified(cat, FOCUS, TOOLS_PER_BUCKET, seed=i)
        log(f"=== start iter {i} · tools={names} ===")
        R = finetune_and_eval(cat, raw, tok, names, f"depth_iter{i}", NB_DIR,
                              cap=CAP_PER_TOOL, epochs=EPOCHS, compact=False, token_aware=True,
                              eval_subsample=EVAL_SUBSAMPLE, run_train=RUN_TRAIN,
                              force_retrain=FORCE_RETRAIN, batch_size=BATCH_SIZE)
        new = []
        for tool, s in R["metrics"]["per_tool"].items():
            _, tot = cat.params_of(tool)
            new.append({"iteration": i, "tool": tool, "total_params": tot,
                        "param_bucket": param_bucket(tot), "n": s["n"],
                        "selection_acc": s["selection_acc"],
                        "args_exact_acc": s["args_exact_acc"], "param_f1": s["param_f1"]})
        rows = [r for r in rows if r["iteration"] != i] + new
        done_iters.add(i)
        json.dump(rows, open(RES, "w"), indent=2)   # persist each iteration
        log(f"=== done iter {i}: {len(new)} tools measured ===")
    except Exception as e:
        log(f"iter {i}: FAILED ({type(e).__name__}: {e}) — skipping; re-run to resume")

log(f"ALL ITERATIONS DONE in {time.time()-t_all:.0f}s · results={RES}")
depth = pd.DataFrame(rows)
print("collected", len(depth), "per-tool measurements across", depth["iteration"].nunique(), "iterations")
display(depth.head(12))'''),
 md("## 4 · Extraction accuracy per parameter bucket (mean ± std)"),
 co('''# per-iteration bucket means first (paired within iteration), then mean/std across iterations
per_iter = (depth.groupby(["iteration","param_bucket"])[["args_exact_acc","param_f1"]]
            .mean().reset_index())
agg = (per_iter.groupby("param_bucket")
       .agg(args_mean=("args_exact_acc","mean"), args_std=("args_exact_acc","std"),
            pf1_mean=("param_f1","mean"), pf1_std=("param_f1","std"),
            n_iter=("iteration","nunique"))
       .reindex([b for b in PARAM_BUCKET_ORDER if b in per_iter["param_bucket"].values]))
display(agg.round(3))

x = np.arange(len(agg)); w = 0.38
fig, ax = plt.subplots(figsize=(8,4.5))
ax.bar(x-w/2, agg["args_mean"], w, yerr=agg["args_std"].fillna(0), capsize=4, color="#4C72B0", label="args_exact_acc")
ax.bar(x+w/2, agg["pf1_mean"], w, yerr=agg["pf1_std"].fillna(0), capsize=4, color="#DD8452", label="param_f1")
# overlay each iteration's bucket mean as points
for _, r in per_iter.iterrows():
    xi = list(agg.index).index(r["param_bucket"]) if r["param_bucket"] in list(agg.index) else None
    if xi is not None: ax.scatter(xi-w/2, r["args_exact_acc"], color="#243b57", s=14, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(agg.index); ax.set_ylim(0,1)
ax.set_xlabel("# parameters (bucket)"); ax.set_ylabel("accuracy")
ax.set_title(f"Depth: parameter extraction vs #params ({int(agg['n_iter'].max())} iterations)"); ax.legend()
plt.tight_layout(); save_fig("depth_buckets", out_dir=NB_DIR); plt.show()'''),
 md("## 5 · Per-tool scatter (all iterations)"),
 co('''tool_mean = depth.groupby(["tool","total_params"])["args_exact_acc"].mean().reset_index()
plt.figure(figsize=(7.5,4.5))
plt.scatter(tool_mean["total_params"], tool_mean["args_exact_acc"], s=55, color="#4C72B0")
for _, r in tool_mean.iterrows():
    plt.annotate(r["tool"].split("_")[0], (r["total_params"], r["args_exact_acc"]), fontsize=7)
plt.xlabel("# parameters in tool"); plt.ylabel("mean args_exact_acc")
plt.title("Depth: per-tool extraction vs parameter count")
plt.tight_layout(); save_fig("depth_scatter", out_dir=NB_DIR); plt.show()'''),
 md("""## 6 · Read-out

Argument extraction is near-solved for **0–1 parameter** tools and **degrades for multi-parameter (4+)**
tools — the error bars show it's a consistent effect across independent tool samples, not one unlucky
model. This is where a small specialist model needs the most help (and where finetuning gains most)."""),
]

# ======================================================================
# NB3 — Separation (ambiguous)
# ======================================================================
nb3 = init_cells("""# PICKO Research · NB3 — **Separation**: can it tell look-alike tools apart?

We finetune one model on the 40 focus tools and probe curated groups of near-identical tools (same action
across sources, or same source across actions). Each group's queries are drawn from the model's **held-out
test set** and offered the full group in one shot, so we measure pure disambiguation + which tool it
confuses for which.""")
nb3 += [
 md("""## 2 · Train / test split

Both the training subprocess and this notebook call the **same deterministic** `per_tool_split`
(`seed=42`, 10 test + 10 val per tool). The model trains **only on the train split**; the group probes
below run **only on the held-out test split**, so no test query is ever seen in training."""),
 md("""## 3 · The ambiguous groups

Each group is a set of look-alike tools on one of two axes: **cross-source** (same action, different
source — the source word disambiguates) or **within-source** (same source, subtly different action)."""),
 co('''AXIS = {"cross_source_search": "cross-source", "single_item_summary": "cross-source",
        "hf_search_variants": "within-source", "wikipedia_retrieve_vs_summarize": "within-source",
        "get_paper_content": "within-source", "arxiv_latex": "within-source"}
grp_rows = []
for g, tools in SIMILAR_GROUPS.items():
    fams = sorted({family_of(t) for t in tools})
    grp_rows.append({"group": g, "axis": AXIS.get(g, ""), "n_tools": len(tools),
                     "families": ",".join(fams), "tools": ", ".join(tools)})
groups_df = pd.DataFrame(grp_rows).sort_values("axis")
display(groups_df)'''),
 md("### Sample queries per tool\\nThe actual requests the model must tell apart — one example query per tool, grouped."),
 co('''def gold_of(ex):
    calls = json.loads(ex.get("answers", "[]"))
    return next((c["name"] for c in calls if isinstance(c, dict) and c.get("name")), None)

sample_q = {}
for e in raw:
    g = gold_of(e)
    if g and g not in sample_q: sample_q[g] = e["query"]

ex_rows = [{"group": g, "axis": AXIS.get(g, ""), "tool": t, "example_query": sample_q.get(t, "")[:160]}
           for g, tools in SIMILAR_GROUPS.items() for t in tools]
display(pd.DataFrame(ex_rows))'''),
 md("## 4 · Train the model (40 focus tools)\\nTrained once to Drive and reused; the returned `test` set is the held-out split the group probes run on."),
 co('''NB_DIR = os.path.join(OUT_DIR, "nb3"); os.makedirs(NB_DIR, exist_ok=True)   # this notebook's outputs
CAP_PER_TOOL, EPOCHS, BATCH_SIZE = 120, 1, 8   # examples/tool -> 100 train / 10 val / 10 test; BATCH_SIZE: lower to 4 on OOM
RUN_TRAIN, FORCE_RETRAIN = True, False
FOCUS40 = finetune_and_eval(cat, raw, tok, FOCUS, "focus40", NB_DIR,
                            cap=CAP_PER_TOOL, epochs=EPOCHS, compact=False, token_aware=True,
                            eval_subsample=None, run_train=RUN_TRAIN,
                            force_retrain=FORCE_RETRAIN, batch_size=BATCH_SIZE)
m40, p40, tk40 = FOCUS40["bundle"]
TEST = FOCUS40["test"]                     # held-out test queries (never trained on)
log(f"focus40 selection_acc={FOCUS40['metrics']['selection_acc']:.3f} · held-out test={len(TEST)}")'''),
 md("## 5 · Per-group disambiguation\\nFor each group we take the held-out queries whose gold tool is in the group and offer the full group. Resumable: finished groups persist to `nb3/separation_results.json`."),
 co('''import contextlib, io
RES = os.path.join(NB_DIR, "separation_results.json")
prev = json.load(open(RES)) if (os.path.exists(RES) and not FORCE_RETRAIN) else {"per_group": [], "confusion": {}}
sep_by = {r["group"]: r for r in prev.get("per_group", [])}
group_conf = prev.get("confusion", {})
if sep_by: log(f"resumed {len(sep_by)} finished group(s)")

def gold_of(ex):
    calls = json.loads(ex.get("answers", "[]"))
    return next((c["name"] for c in calls if isinstance(c, dict) and c.get("name")), None)

t_all = time.time()
for gname, gtools in SIMILAR_GROUPS.items():
    if gname in sep_by and gname in group_conf and not FORCE_RETRAIN:
        log(f"{gname}: skip (done)"); continue
    try:
        gset = set(gtools)
        gtest = [e for e in TEST if gold_of(e) in gset]                      # held-out queries for this group
        offered = [offer_subset(cat, gold_of(e), gtools, len(gtools), seed=0, compact=False) for e in gtest]
        with contextlib.redirect_stdout(io.StringIO()):
            gpreds = predict(m40, p40, tk40, gtest, tools_override=offered)
        gm = evaluate(gtest, gpreds, family_of=family_of)
        sep_by[gname] = {"group": gname, "n_tools": len(gtools), "n": gm["n"],
                         "selection_acc": gm["selection_acc"], "name_f1": gm["name_f1"]}
        group_conf[gname] = confusion(gtest, gpreds)
        json.dump({"per_group": list(sep_by.values()), "confusion": group_conf}, open(RES, "w"), indent=2)
        log(f"{gname}: selection={gm['selection_acc']:.3f} (n={gm['n']})")
    except Exception as e:
        log(f"{gname}: FAILED ({type(e).__name__}: {e})")

log(f"ALL GROUPS DONE in {time.time()-t_all:.0f}s · results={RES}")
if not sep_by:
    raise RuntimeError("No group succeeded — see the FAILED lines above.")
separation = pd.DataFrame(list(sep_by.values())).sort_values("selection_acc")
display(separation)

plt.figure(figsize=(8,4)); plt.barh(separation["group"], separation["selection_acc"], color="#4C72B0")
plt.xlim(0,1); plt.xlabel("tool-selection accuracy"); plt.title("Separation: hardest look-alike groups (lower = more confused)")
plt.tight_layout(); save_fig("separation_groups", out_dir=NB_DIR); plt.show()'''),
 md("## 6 · Confusion heatmaps (who gets mistaken for whom)"),
 co('''for gname, conf in group_conf.items():
    labels = sorted(set(conf) | {p for row in conf.values() for p in row})
    M = pd.DataFrame(0, index=sorted(conf), columns=labels)
    for r, row in conf.items():
        for p, n in row.items(): M.loc[r, p] = n
    plt.figure(figsize=(0.9*len(labels)+2, 0.5*len(M)+1.5))
    if sns: sns.heatmap(M, annot=True, fmt="d", cmap="Blues", cbar=False)
    else:
        plt.imshow(M.values, cmap="Blues"); plt.xticks(range(len(labels)), labels, rotation=90); plt.yticks(range(len(M)), M.index)
    plt.title(f"Separation · {gname}"); plt.xlabel("predicted"); plt.ylabel("reference")
    plt.tight_layout(); save_fig(f"separation_confusion_{gname}", out_dir=NB_DIR); plt.show()'''),
 md("""## 7 · Read-out

Residual selection errors concentrate inside these look-alike groups. The lowest-accuracy group is the
frontier for a tool-picker; the heatmaps show whether confusions are symmetric (two tools mutually
confused) or a sink (everything collapses to one generic tool)."""),
]


def write(cells, name):
    for j, c in enumerate(cells):
        c["id"] = f"c{j:02d}"
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "PICKO (.venv)", "language": "python", "name": "picko"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = os.path.join(OUTDIR, name)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path, "·", len(cells), "cells")


if __name__ == "__main__":
    write(nb1, "nb1_breadth_amount.ipynb")
    write(nb2, "nb2_depth_parameters.ipynb")
    write(nb3, "nb3_separation_ambiguous.ipynb")

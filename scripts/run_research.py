#!/usr/bin/env python3
"""Headless PICKO research runner — the unattended equivalent of Notebook 3.

Runs every finetune (Baseline · Breadth · Depth · Separation) and writes:
  - checkpoints/picko_<tag>_best.pkl     (one per run, stable names)
  - data/research_results.json           (all metrics/tables for plotting)

Designed to survive an overnight run. Launch it so macOS won't sleep mid-run
(stay on AC power):

    caffeinate -is nohup .venv/bin/python -u scripts/run_research.py \
        > research.log 2>&1 &

  caffeinate -i  = no idle sleep · -s = no system sleep on AC.
Then watch progress with:  tail -f research.log
Re-running resumes: finished tags with a checkpoint are skipped unless --retrain.

Afterwards, open Notebook 3 with RUN_TRAIN=False (it reloads these checkpoints),
or read data/research_results.json directly.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from scripts.tool_catalog import Catalog, family_of
from scripts.research_sets import (focus_names, BREADTH_SIZES, nested_sets,
                                    SIMILAR_GROUPS, param_bucket, PARAM_BUCKET_ORDER)
from scripts.picko_eval import (load_model, predict, evaluate, confusion,
                                base_checkpoint, n_visible)
from needle.training.finetune import _per_tool_split
from needle.dataset.dataset import get_tokenizer

NEEDLE = shutil.which("needle") or os.path.join(os.path.dirname(sys.executable), "needle")


def finetune_and_eval(cat, raw, tok, names, tag, *, cap, epochs, retrain,
                      compact=False, offer_all=None, token_aware=False,
                      eval_subsample=None):
    kw = dict(cap_per_tool=cap, compact=compact, seed=0)
    if offer_all is not None:
        kw["offer_all_max"] = offer_all
    if token_aware:
        kw["tokenizer"] = tok
    data = cat.restrict_dataset(raw, names, **kw)
    path = os.path.join(ROOT, "data", f"picko_{tag}.jsonl")
    with open(path, "w") as f:
        for e in data:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    ckpt = os.path.join(ROOT, "checkpoints", f"picko_{tag}_best.pkl")
    if retrain or not os.path.exists(ckpt):
        print(f"[{tag}] finetuning on {len(data)} examples ({len(names)} tools)…", flush=True)
        subprocess.run([NEEDLE, "finetune", path, "--epochs", str(epochs),
                        "--batch-size", "32"], cwd=ROOT, check=True)
        newest = max(glob.glob(os.path.join(ROOT, "checkpoints", "needle_finetuned_*_best.pkl")),
                     key=os.path.getmtime)
        shutil.copy(newest, ckpt)
    else:
        print(f"[{tag}] reusing existing checkpoint (use --retrain to redo)", flush=True)

    _, _, test = _per_tool_split(data)
    if eval_subsample:
        test = test[:eval_subsample]
    m, p, tk = load_model(ckpt)
    print(f"[{tag}] evaluating on {len(test)} test examples…", flush=True)
    preds = predict(m, p, tk, test)
    metrics = evaluate(test, preds, family_of=family_of)
    return {"names": names, "ckpt": ckpt, "bundle": (m, p, tk),
            "test": test, "preds": preds, "metrics": metrics, "data": data}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-per-tool", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eval-subsample", type=int, default=None)
    ap.add_argument("--retrain", action="store_true",
                    help="Retrain even if a checkpoint already exists.")
    args = ap.parse_args()

    cat = Catalog()
    tok = get_tokenizer()
    raw = [json.loads(l) for l in open(os.path.join(ROOT, "data", "picko_balanced.jsonl")) if l.strip()]
    FOCUS = focus_names(cat)
    common = dict(cap=args.cap_per_tool, epochs=args.epochs, retrain=args.retrain,
                  eval_subsample=args.eval_subsample)
    results = {"config": {"cap_per_tool": args.cap_per_tool, "epochs": args.epochs,
                          "focus": FOCUS}}

    # 1) Baseline — 3 diverse tools (one per family where possible)
    print("\n=== Baseline (3 tools) ===", flush=True)
    baseline_tools = ["arxiv_search_papers", "pubmed_search_articles", "wikipedia_search_wikipedia"]
    R = finetune_and_eval(cat, raw, tok, baseline_tools, "baseline3",
                          compact=False, offer_all=3, **common)
    bm, pm, tkm = load_model(base_checkpoint())
    base_preds = predict(bm, pm, tkm, R["test"])
    results["baseline"] = {"tools": baseline_tools,
                           "base": evaluate(R["test"], base_preds, family_of=family_of),
                           "finetuned": R["metrics"]}

    # 2) Breadth — nested sizes, compact, selection only
    print("\n=== Breadth (tool-set size) ===", flush=True)
    SETS = nested_sets(FOCUS, BREADTH_SIZES, seed=0)
    breadth = []
    for k in BREADTH_SIZES:
        Rk = finetune_and_eval(cat, raw, tok, SETS[k], f"breadth_k{k}",
                               compact=True, offer_all=k, **common)
        vis = int(np.median([n_visible(e["query"], json.loads(e["tools"]), tok)
                             for e in Rk["test"]]))
        breadth.append({"k": k, "selection_acc": Rk["metrics"]["selection_acc"],
                        "name_f1": Rk["metrics"]["name_f1"], "n_visible": vis})
        print(f"  k={k}: selection={Rk['metrics']['selection_acc']:.3f} visible={vis}/{k}", flush=True)
    results["breadth"] = breadth

    # 3) Depth — all 40, full schemas, per parameter-count bucket
    print("\n=== Depth (parameter complexity) ===", flush=True)
    DEPTH = finetune_and_eval(cat, raw, tok, FOCUS, "depth40",
                              compact=False, token_aware=True, **common)
    df = cat.as_dataframe().set_index("tool")
    depth_tools = []
    for name, s in DEPTH["metrics"]["per_tool"].items():
        tot = int(df.loc[name, "total_params"]) if name in df.index else 0
        depth_tools.append({"tool": name, "total_params": tot, "bucket": param_bucket(tot),
                            "n": s["n"], "selection_acc": s["selection_acc"],
                            "args_exact_acc": s["args_exact_acc"], "param_f1": s["param_f1"]})
    results["depth"] = {"per_tool": depth_tools, "overall": {
        k: DEPTH["metrics"][k] for k in ["selection_acc", "name_f1", "args_exact_acc",
                                         "param_f1", "call_exact"]}}

    # 4) Separation — curated look-alike groups on the 40-tool model
    print("\n=== Separation (disambiguation) ===", flush=True)
    m40, p40, tk40 = DEPTH["bundle"]
    separation = []
    for gname, gtools in SIMILAR_GROUPS.items():
        gset = cat.restrict_dataset(raw, gtools, offer_all_max=len(gtools),
                                    cap_per_tool=args.cap_per_tool, seed=0)
        _, _, gtest = _per_tool_split(gset)
        if args.eval_subsample:
            gtest = gtest[:args.eval_subsample]
        gpreds = predict(m40, p40, tk40, gtest)
        gm = evaluate(gtest, gpreds, family_of=family_of)
        separation.append({"group": gname, "n_tools": len(gtools), "n": gm["n"],
                           "selection_acc": gm["selection_acc"], "name_f1": gm["name_f1"],
                           "confusion": confusion(gtest, gpreds)})
        print(f"  {gname}: selection={gm['selection_acc']:.3f}", flush=True)
    results["separation"] = separation

    out = os.path.join(ROOT, "data", "research_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDONE — wrote {out}", flush=True)


if __name__ == "__main__":
    main()

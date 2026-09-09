#!/usr/bin/env python3
"""Recompute base-vs-finetuned tool-call metrics on the SAME held-out test set
that `needle finetune` used (its per-tool split is deterministic, seed=42).

Usage:
    python scripts/eval_picko.py data/picko_subset.jsonl \
        checkpoints/needle_finetuned_..._best.pkl
"""
import json
import sys

from needle.model.run import load_checkpoint
from needle.model.architecture import SimpleAttentionNetwork
from needle.dataset.dataset import get_tokenizer
from needle.training.finetune import _per_tool_split, _quick_tool_eval, _resolve_checkpoint


def _fmt(tag, m):
    print(f"\n== {tag} ==")
    print(f"  call_f1={m['call_f1']:.3f}  exact={m['exact_match']:.3f}  "
          f"name_f1={m['name_f1']:.3f}  args_acc={m['args_acc']:.3f}  "
          f"parse={m['parse_rate']:.3f}  n={m['n']}")
    for t, d in sorted(m["per_tool"].items()):
        print(f"    {t:28} {d['correct']}/{d['total']}")


def _load(path):
    params, config = load_checkpoint(path)
    return SimpleAttentionNetwork(config), params


def main(data_path, ft_ckpt):
    examples = [json.loads(l) for l in open(data_path) if l.strip()]
    _, _, test = _per_tool_split(examples)  # identical seed=42 split
    print(f"Held-out test examples: {len(test)}")
    tok = get_tokenizer()

    base_ckpt = _resolve_checkpoint(None)  # downloads base if needed
    print(f"Base checkpoint:      {base_ckpt}")
    print(f"Finetuned checkpoint: {ft_ckpt}")

    bm, bp = _load(base_ckpt)
    base = _quick_tool_eval(bm, bp, tok, test)
    _fmt("BASE (un-finetuned)", base)

    fm, fp = _load(ft_ckpt)
    ft = _quick_tool_eval(fm, fp, tok, test)
    _fmt("FINETUNED (PICKO)", ft)

    print("\n== DELTA (finetuned - base) ==")
    for k in ("call_f1", "exact_match", "name_f1", "args_acc"):
        print(f"  {k:12} {base[k]:.3f} -> {ft[k]:.3f}   ({ft[k]-base[k]:+.3f})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python scripts/eval_picko.py <data.jsonl> <finetuned_best.pkl>")
    main(sys.argv[1], sys.argv[2])

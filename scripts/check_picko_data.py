#!/usr/bin/env python3
"""Validate a PICKO training JSONL before running `needle finetune`.

Checks each line is well-formed for the needle format:
  * parses as JSON with `query`, `tools`, `answers`
  * `tools` and `answers` are JSON-encoded STRINGS (not nested objects)
  * every answer call `name` exists in that line's `tools`
  * per-tool answer counts (warns on <120, the needle threshold)

Usage:
    python scripts/check_picko_data.py data/picko_subset.jsonl
"""
import json
import sys
from collections import Counter


def main(path):
    n = 0
    bad = 0
    tool_counts = Counter()
    empty = 0
    problems = []

    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                ex = json.loads(line)
            except ValueError as e:
                bad += 1
                problems.append(f"L{lineno}: not JSON ({e})")
                continue
            if not all(k in ex for k in ("query", "tools", "answers")):
                bad += 1
                problems.append(f"L{lineno}: missing query/tools/answers")
                continue
            if not isinstance(ex["tools"], str) or not isinstance(ex["answers"], str):
                bad += 1
                problems.append(f"L{lineno}: tools/answers must be JSON-encoded strings")
                continue
            try:
                tools = json.loads(ex["tools"])
                answers = json.loads(ex["answers"])
            except ValueError as e:
                bad += 1
                problems.append(f"L{lineno}: inner JSON invalid ({e})")
                continue
            tool_names = {t.get("name") for t in tools}
            if not answers:
                empty += 1
            for call in answers:
                name = call.get("name")
                if name not in tool_names:
                    bad += 1
                    problems.append(f"L{lineno}: answer '{name}' not in offered tools")
                elif name:
                    tool_counts[name] += 1

    print(f"Lines: {n}   valid: {n - bad}   bad: {bad}   abstention(empty answers): {empty}")
    print("\nPer-tool answer counts:")
    for name, c in tool_counts.most_common():
        flag = "  ⚠ <120" if c < 120 else ""
        print(f"  {name:28} {c}{flag}")

    low = [t for t, c in tool_counts.items() if c < 120]
    if problems[:20]:
        print("\nFirst problems:")
        for p in problems[:20]:
            print("  -", p)

    if bad:
        print(f"\n✗ {bad} malformed line(s) — fix before finetuning.")
        return 1
    if low:
        print(f"\n⚠ {len(low)} tool(s) under 120 examples: {sorted(low)}")
        print("  Top up with: python scripts/generate_picko_data.py --focus <cluster> ...")
        return 0
    print("\n✓ Data looks good — ready for `needle finetune`.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/check_picko_data.py <data.jsonl>")
    sys.exit(main(sys.argv[1]))

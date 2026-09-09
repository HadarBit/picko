#!/usr/bin/env python3
"""Load and slice the PICKO scientific tool catalog.

Source of truth: `full_tools_53tools_11products.json` (75 tools, 11 families).
Grouping/metadata: `tools_metadata.csv` (family -> category, arg counts).

A "family" (aka product/source) is identified by the tool-name prefix — e.g.
`arxiv_search_papers` -> `arxiv`, `semantic_scholar_search_papers` -> `semantic_scholar`.
Families map up to 4 higher-level categories via the CSV.

Typical use:
    from scripts.tool_catalog import Catalog
    cat = Catalog()
    cat.list_families()                      # {family: {count, category, ...}}
    tools, pools = cat.select_tools(k=11)     # the 11-per-family D3 set
    tools, pools = cat.select_tools(families=["arxiv", "pubmed"])
    tools, pools = cat.select_tools(categories=["Academic & Literature Search"])
`pools` is a list-of-lists (one inner list per family) — the shape
`generate_picko_data._apply_patches` expects.
"""
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_JSON = os.path.join(HERE, "full_tools_53tools_11products.json")
ELEVEN_JSON = os.path.join(HERE, "11tools_11products.json")
METADATA_CSV = os.path.join(HERE, "tools_metadata.csv")

# Known family prefixes (longest-match first so `semantic_scholar` beats `semantic`).
KNOWN_FAMILIES = [
    "semantic_scholar", "pubmed", "arxiv", "wikipedia", "kaggle", "crossref",
    "openalex", "unpaywall", "google", "github", "hf",
]


def family_of(tool_name):
    """Map a tool name to its family prefix (longest known prefix wins)."""
    for fam in sorted(KNOWN_FAMILIES, key=len, reverse=True):
        if tool_name == fam or tool_name.startswith(fam + "_"):
            return fam
    return tool_name.split("_")[0]  # fallback


def _first_int(s):
    m = re.search(r"\d+", s or "")
    return int(m.group()) if m else 0


class Catalog:
    def __init__(self, master=MASTER_JSON, metadata=METADATA_CSV):
        with open(master) as f:
            self.tools = json.load(f)
        self.by_name = {t["name"]: t for t in self.tools}

        # family -> category / source label, from the CSV's representative rows
        self.family_category = {}
        self.family_source = {}
        if os.path.exists(metadata):
            with open(metadata, newline="") as f:
                for row in csv.DictReader(f):
                    # Messy CSV: stray commas make DictReader overflow into a
                    # list under key None — keep only real string columns.
                    row = {(k or "").strip(): (v or "").strip()
                           for k, v in row.items()
                           if isinstance(k, str) and isinstance(v, str)}
                    tname = row.get("tool name", "")
                    if not tname:
                        continue
                    fam = family_of(tname)
                    self.family_category[fam] = row.get("category", "")
                    self.family_source[fam] = row.get("source name", "")

    # ---- introspection ----
    def family_of(self, tool_name):
        return family_of(tool_name)

    def category_of(self, family):
        return self.family_category.get(family, "Uncategorized")

    def params_of(self, tool_name):
        params = self.by_name[tool_name].get("parameters", {}) or {}
        total = len(params)
        required = sum(1 for p in params.values()
                       if isinstance(p, dict) and p.get("required"))
        return required, total

    def list_families(self):
        """{family: {count, category, source, tools:[...]}}, sorted by size desc."""
        fams = {}
        for t in self.tools:
            fam = family_of(t["name"])
            d = fams.setdefault(fam, {"count": 0, "tools": [],
                                      "category": self.category_of(fam),
                                      "source": self.family_source.get(fam, fam)})
            d["count"] += 1
            d["tools"].append(t["name"])
        return dict(sorted(fams.items(), key=lambda kv: -kv[1]["count"]))

    def as_dataframe(self):
        """One row per tool: name, family, category, required/total params."""
        import pandas as pd
        rows = []
        for t in self.tools:
            fam = family_of(t["name"])
            req, tot = self.params_of(t["name"])
            rows.append({"tool": t["name"], "family": fam,
                         "category": self.category_of(fam),
                         "required_params": req, "total_params": tot})
        return pd.DataFrame(rows)

    # ---- selection ----
    def select_tools(self, families=None, categories=None, names=None, k=None,
                     one_per_family=False):
        """Return (flat_tools, pools) for a chosen slice.

        - names:          explicit tool names.
        - families:       whole families by prefix.
        - categories:     whole categories (via CSV mapping).
        - one_per_family: pick the representative tool per family (the 11-set).
        - k:              cap the number of tools (after the above filters).
        `pools` groups the chosen tools by family (list of lists).
        """
        if names:
            chosen = [self.by_name[n] for n in names if n in self.by_name]
        elif one_per_family:
            with open(ELEVEN_JSON) as f:
                reps = json.load(f)
            chosen = [self.by_name.get(t["name"], t) for t in reps]
        else:
            chosen = list(self.tools)
            if families:
                fset = set(families)
                chosen = [t for t in chosen if family_of(t["name"]) in fset]
            if categories:
                cset = set(categories)
                chosen = [t for t in chosen
                          if self.category_of(family_of(t["name"])) in cset]

        if k is not None:
            chosen = chosen[:k]

        # group into pools by family, preserving order
        pools_map = {}
        for t in chosen:
            pools_map.setdefault(family_of(t["name"]), []).append(t)
        return chosen, list(pools_map.values())


if __name__ == "__main__":
    cat = Catalog()
    print(f"{len(cat.tools)} tools, {len(cat.list_families())} families")
    for fam, d in cat.list_families().items():
        print(f"  {fam:18} n={d['count']:2}  [{d['category']}]")
    flat, pools = cat.select_tools(one_per_family=True)
    print(f"\none-per-family set: {len(flat)} tools -> {[t['name'] for t in flat]}")

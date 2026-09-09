#!/usr/bin/env python3
"""Build the PICKO quick-start scientific tool subset from the full catalog.

Reads `full_tools_first_draft.json` (40 scientific tool definitions), fixes the
`descrptiion` typo, selects an 8-tool subset spanning four clusters, and writes:

  data/scientific_pools.json   {pool_name: [tool, ...]} grouped by cluster
  data/picko_subset_tools.json flat JSON array of the 8 tools (for `needle run`)

The clustering matters: same-cluster tools (e.g. the four "*_search_*" tools)
get shown together to the generator so PICKO must *disambiguate* between
semantically similar tools (project dimension D3). The subset also spans a
1 -> 6 required/optional parameter range to stress parameter extraction (D2).

Usage:
    python scripts/make_scientific_pools.py
"""
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG = os.path.join(HERE, "full_tools_first_draft.json")
OUT_DIR = os.path.join(HERE, "data")

# Quick-start subset, grouped by cluster. 4 "search" tools across clusters give
# the disambiguation stress (D3); the mix of param counts gives extraction (D2).
POOLS = {
    "pubmed": ["pubmed_search_articles", "pubmed_download_article"],
    "arxiv": ["arxiv_search_papers", "arxiv_get_abstract"],
    "wikipedia": ["wikipedia_search_wikipedia", "wikipedia_extract_key_facts"],
    "huggingface": ["hf_paper_search", "hf_model_search"],
}


def _fix_typos(obj):
    """Recursively rename the known `descrptiion` typo key to `description`."""
    if isinstance(obj, dict):
        fixed = {}
        for k, v in obj.items():
            if k == "descrptiion":
                k = "description"
            fixed[k] = _fix_typos(v)
        return fixed
    if isinstance(obj, list):
        return [_fix_typos(x) for x in obj]
    return obj


def main():
    with open(CATALOG) as f:
        catalog = _fix_typos(json.load(f))
    by_name = {t["name"]: t for t in catalog}

    pools = {}
    flat = []
    missing = []
    for pool_name, names in POOLS.items():
        pools[pool_name] = []
        for n in names:
            if n not in by_name:
                missing.append(n)
                continue
            pools[pool_name].append(by_name[n])
            flat.append(by_name[n])
    if missing:
        raise SystemExit(f"Tools not found in catalog: {missing}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "scientific_pools.json"), "w") as f:
        json.dump(pools, f, indent=2)
    # Compact single-line array, matching the `tools` field encoding needle uses.
    with open(os.path.join(OUT_DIR, "picko_subset_tools.json"), "w") as f:
        json.dump(flat, f, separators=(",", ":"), ensure_ascii=False)

    print(f"Wrote {len(flat)} tools across {len(pools)} pools:")
    for pool_name, tools in pools.items():
        for t in tools:
            n_req = sum(1 for p in t.get("parameters", {}).values()
                        if isinstance(p, dict) and p.get("required"))
            n_tot = len(t.get("parameters", {}))
            print(f"  [{pool_name:11}] {t['name']:28} {n_req} required / {n_tot} params")


if __name__ == "__main__":
    main()

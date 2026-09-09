# PICKO — Project Guide

PICKO fine-tunes **Needle** (Cactus's 26M-param JAX/Flax encoder-decoder tool-picker) to select the
right **scientific** tool and extract its arguments from a single prompt. This file is the map of the
PICKO-specific files we added on top of the upstream Needle repo (whose own docs are in `README.md`).

Research dimensions we measure:
- **D1** — tool-set size (how many tools before accuracy degrades)
- **D2** — parameter extraction (getting arguments right)
- **D3** — disambiguation of semantically similar tools (e.g. arxiv vs pubmed vs semantic scholar)

---

## 1. Tool catalogs & metadata

All three JSON files share the same schema — a flat list of
`{ "name", "description", "parameters": { <arg>: {"type","description","required"} } }` — and differ
only in *which* tools they contain.

| File | Brief |
|---|---|
| `full_tools_first_draft.json` | The **original** catalog. **40 tools**, 4 families (arxiv 14, hf 12, wikipedia 10, pubmed 4). Used for the first 8-tool experiment. Has a `descrptiion` typo (auto-fixed on load). Subset of the master below. |
| `full_tools_53tools_11products.json` | The **master catalog** (source of truth). **75 tools** across **11 families/products** (arxiv, hf, kaggle, wikipedia, github, openalex, semantic_scholar, pubmed, crossref, google, unpaywall). *(Filename says "53" but it holds 75.)* |
| `11tools_11products.json` | **One representative tool per family** (11 tools). All similar "search"-style tools → the **gold set for D3** disambiguation, and a small bucket for D1. |
| `tools_metadata.csv` | Summary table, one row per the 11 representative tools: `category, source name, tool name, required args, total args`. Groups the 11 families into **4 categories** (Academic & Literature Search · Data & ML Repositories · General Knowledge & Web · Scholarly Metadata). |

**Relationship:** `first_draft` (40) ⊂ `master` (75); `11tools` (11) is a one-per-family slice of the master; the CSV is metadata over those 11.

---

## 2. Generated data & training files (`data/`, git-ignored)

Each line is one training example: `{"query", "tools", "answers"}` where `tools`/`answers` are
JSON-encoded strings. Needle needs **≥120 examples per tool**.

| File | Brief |
|---|---|
| `data/picko_gen.jsonl` | **The raw working pool.** Every generated example is *appended* here (currently ~8.6k lines, 50/75 tools at ≥120). Notebook 1's `OUTPUT_JSONL`. |
| `data/picko_balanced_50.jsonl` | Clean, finetune-ready: the **50 ready tools × exactly 120** + negatives (6,200 lines). Use this today. |
| `data/picko_balanced.jsonl` | An earlier balanced snapshot (superseded by `picko_balanced_50.jsonl`). |
| `data/picko_subset.jsonl` | The original **8-tool** dataset (1,935 lines) — the first, already-finetuned experiment. |
| `data/scientific_pools.json`, `data/picko_subset_tools.json` | The 8-tool subset definitions used by the first experiment. |

---

## 3. Scripts (`scripts/`)

Reusable modules; the notebooks are thin wrappers over these.

| Script | Brief |
|---|---|
| `tool_catalog.py` | `Catalog` — load the 75-tool master + CSV; `family_of`, `category_of`, `select_tools(families=/categories=/one_per_family=/k=)`, `as_dataframe()`. |
| `generate_picko_data.py` | Gemini data synthesis. Key pieces: `robust_generate` (coverage-driven, retry/backoff), **`force_tool`** (forces ≥N examples for one tool by offering only it), **`balance_dataset`** (trim to exactly N/tool), a **13-RPM rate limiter** (`set_rate_limit`) and **fail-fast quota handling** (`QuotaExhausted`). Needs `GEMINI_API_KEY`. |
| `check_picko_data.py` | Validate a JSONL: format correct, answer names in `tools`, per-tool counts (warns <120). |
| `picko_eval.py` | The evaluator: **separates tool-selection from parameter-extraction**, per-tool & per-family, plus D3 `confusion()` and D1 `build_tools_override()`. Reuses Needle's batched decoder. |
| `eval_picko.py` | CLI to recompute base-vs-finetuned metrics on the deterministic (seed-42) test split from a saved checkpoint. |
| `make_scientific_pools.py` | Builds the original 8-tool subset + pools (fixes the `descrptiion` typo). |

---

## 4. Notebooks (`notebooks/`)

### `01_generate_data.ipynb` — Generate training data with Gemini
Produce a balanced `{query, tools, answers}` dataset for a chosen tool set, with live progress bars.

1. **Setup** — load `GEMINI_API_KEY` from `.env`, imports.
2. **Pick tools** — `SELECTION` is the single source of truth. `dict()` = **all 75**; `dict(one_per_family=True)` = the 11-rep D3 set; `dict(families=[…])` / `categories=[…]` / `names=[…]`.
3. **Configure** — `TARGET_PER_TOOL`, `FINAL_PER_TOOL`, `WORKERS`, **`RPM=13`** (stays under the 15-RPM free-tier limit), output paths. Applies the rate limiter.
4. **Broad pass (optional)** — offers the whole set and lets Gemini choose; fast coverage of popular tools + negatives. Set `DO_BROAD=False` to skip.
5. **Force every tool to ≥ target ⭐** — for each short tool, offers **only that tool** so Gemini must call it (no starved niche tools). Stops cleanly if the quota is exhausted (progress is saved — just rerun later).
6. **Balance to exactly N/tool** — writes the finetune-ready `data/picko_balanced.jsonl`, scoped to your selection.
7. **Validate** + **8. Visualize** coverage.

### `02_finetune_and_analyze.ipynb` — Finetune & the research story
Explore families, finetune (or reuse a checkpoint), and measure results, telling the D1/D2/D3 story.

1. **Setup** · **2. Explore families** — tables + charts of the 11 families / 4 categories / param-count distribution.
2. **Select subset** — choose tools to analyze (`DATA_JSONL` = a file from Notebook 1).
3. **Finetune (reuse-on-demand)** — loads an existing `checkpoints/needle_finetuned_*_best.pkl`; set `RUN_FINETUNE=True` to retrain (~40 min CPU).
4. **Run inference** — greedy-decode base vs finetuned on the held-out test set (cached).
5. **Core result (D2)** — base-vs-finetuned table separating **tool selection** vs **parameter extraction**, per-tool & per-family, + a "param-extraction vs #params" plot.
6. **D3** — confusion heatmap over the offered tools (which tool gets picked when wrong).
7. **D1** — accuracy vs #tools offered: a cheap eval-time distractor sweep (5/10/20/50) + an optional full-retrain sweep.
8. **Story** — narrative cells tying it together for the final write-up.

---

## 5. Checkpoints (`checkpoints/`, git-ignored)

| File | Brief |
|---|---|
| `needle.pkl` | Base Needle weights, auto-downloaded from HF `Cactus-Compute/needle`. |
| `needle_finetuned_20260820120014_*_best.pkl` | The finetuned **PICKO** model (8-tool experiment). `*_33/66/99.pkl` are per-epoch snapshots; `_best.pkl` is the one to use. |

---

## 6. Metrics glossary (used in Notebook 2 / `picko_eval.py`)

| Metric | Means |
|---|---|
| **selection_acc / name_f1** | **Tool-function selection** — did it pick the right *tool*? |
| **args_exact_acc** | **Parameter extraction** — given the right tool, is the whole argument dict exactly right? |
| **param_f1** | Finer parameter extraction — per-argument precision/recall (partial credit). |
| **call_exact** | Full call correct (name **and** args). |
| **abstain_acc** | On "no tool fits" cases, how often it correctly returns nothing. |

**Headline result (8-tool run):** finetuning moved parameter extraction **41% → 65%** and call F1
**36% → 60%**, while tool selection was already high (name F1 88% → 92%) — the win is in D2.

---

## 7. Quick start

```bash
source ./setup                 # venv + Needle + JAX (once)
export GEMINI_API_KEY=...       # kept in .env
```

Then either drive the notebooks (`jupyter notebook notebooks/`), or the CLI:

```bash
# generate (coverage-driven, rate-limited) then validate
python scripts/generate_picko_data.py --workers 3 --target-per-tool 130
python scripts/check_picko_data.py data/picko_gen.jsonl

# finetune + auto base-vs-finetuned eval
needle finetune data/picko_balanced_50.jsonl --epochs 3 --batch-size 32

# recompute separated metrics from a checkpoint
python scripts/eval_picko.py data/picko_balanced_50.jsonl checkpoints/needle_finetuned_*_best.pkl
```

> **Note:** run finetune via the `needle` console script, **not** `python -m needle.cli` (no `__main__` guard). CPU eval is slow (~4 tok/s).

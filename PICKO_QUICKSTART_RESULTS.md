# PICKO — Quick-Start Finetune: What We Did & Results

**Project:** PICKO — a specialized tool-picker for scientific AI agents, built by finetuning
**Needle** (Cactus's 26M-parameter JAX/Flax encoder-decoder; Simple Attention Network, no FFN)
to select the right scientific tool and extract its arguments from a single prompt.

**This document** records the first end-to-end quick-start run: data generation → finetuning →
evaluation, on an 8-tool subset. It probes the project's research dimensions **D2** (parameter
extraction) and **D3** (disambiguation of similar tools); scaling the tool count for **D1** is the
next step.

---

## 1. The problem we had to solve first

The file we started from, `full_tools_first_draft.json`, is a **tool catalog** — 40 scientific
tool *definitions* (`{name, description, parameters}`), **not** training data. Needle's finetuner
consumes a **JSONL** where each line is a training example:

```json
{"query": "...", "tools": "<JSON-encoded tool array>", "answers": "<JSON-encoded call array>"}
```

with **≥120 examples per tool** (100 train / 10 val / 10 test). So the core work was
**synthesizing** query→answer examples over our tools.

---

## 2. The 8-tool subset

Chosen from the 40-tool catalog (4 clusters), to stress D2 and D3 in a small, fast run:

| Cluster | Tool | Required / total params | Role |
|---|---|---|---|
| PubMed | `pubmed_search_articles` | 1 / 3 | literature search (D3) |
| PubMed | `pubmed_download_article` | 1 / 3 | fetch by ID (D2) |
| ArXiv | `arxiv_search_papers` | 1 / 6 | literature search (D3), many params (D2) |
| ArXiv | `arxiv_get_abstract` | 1 / 1 | single-param baseline (D2) |
| Wikipedia | `wikipedia_search_wikipedia` | 1 / 2 | literature/info search (D3) |
| Wikipedia | `wikipedia_extract_key_facts` | 1 / 3 | fact extraction (D2) |
| HuggingFace | `hf_paper_search` | 1 / 3 | paper search (D3) |
| HuggingFace | `hf_model_search` | 0 / 6 | model search, all-optional params (D2) |

**Why these:** four near-synonymous `*_search_*` tools force the model to *disambiguate* rather than
keyword-match a tool name (**D3**); the 1→6 parameter spread stresses *argument extraction* (**D2**).
The catalog's `descrptiion` typo (in `pubmed_search_articles`) is fixed during subsetting.

---

## 3. The pipeline (scripts added, all under `scripts/`)

| Step | Script | What it does |
|---|---|---|
| 1 | `make_scientific_pools.py` | Picks the 8 tools, fixes the typo, groups them into clusters → `data/scientific_pools.json`, `data/picko_subset_tools.json`. |
| 2 | `generate_picko_data.py` | Synthesizes training data via **Gemini**, reusing Needle's own generator (`needle/dataset/generate.py`) but monkeypatched onto our tools. |
| 3 | `check_picko_data.py` | Validates JSONL format + per-tool counts (≥120). |
| 4 | `needle finetune` | Finetunes + auto-evaluates base vs finetuned. |
| — | `eval_picko.py` | Recomputes base-vs-finetuned metrics on the deterministic test split (used to recover results). |

### How data generation was adapted (step 2)

Needle's generator ships tuned for **on-device assistant** tools (timers, smart home). We pointed it
at our scientific tools by patching its module globals — no fork of the upstream file:

- `ALL_POOLS` → our 4 scientific clusters.
- `_synthesize_tools` → **disabled** (never invent off-catalog tools).
- `_OVERLAP_PAIRS` → our own similar `*_search_*` tools, so batches routinely offer 2+
  near-synonymous tools → forces **D3** disambiguation.
- `CALL_TYPES` → **single-shot heavy** (PICKO scope = one prompt → one call): mostly `single`,
  plus abstention negatives (`none`/`near_miss`) and light robustness (`indirect`/`disfluent`/`garbled`).
  Multi-call types dropped.
- `_CONTEXT_SEEDS` / `SCENARIOS` → **science personas & tasks** (PhD student doing a lit review, a
  bioinformatician, "find recent papers on a method"…) instead of "lock the back door".
- `LANGUAGES` → English only.

The script is **coverage-driven**: it generates at low concurrency (3 workers) with retry/backoff
(to avoid Gemini 429 rate limits — the first 8-worker attempt lost 397/414 batches to throttling),
runs a broad phase, then **auto-tops-up** any tool still under target by focusing on its cluster.

### Generated dataset

- **1,935 examples total**, 0 malformed, **451 abstention negatives** (empty `answers`, teaching the
  model to decline when no tool fits).
- Per-tool positive counts (all ≥ target 140):

  | Tool | # |
  |---|---|
  | arxiv_search_papers | 243 |
  | pubmed_search_articles | 213 |
  | hf_paper_search | 212 |
  | pubmed_download_article | 177 |
  | arxiv_get_abstract | 173 |
  | wikipedia_extract_key_facts | 161 |
  | wikipedia_search_wikipedia | 160 |
  | hf_model_search | 145 |

Output: `data/picko_subset.jsonl`.

---

## 4. Finetuning

```bash
needle finetune data/picko_subset.jsonl --epochs 3 --batch-size 32
```

- Split (deterministic, seed 42): **1,755 train / 90 val / 90 test** (per-tool 10 held out for test).
- Base weights auto-downloaded from HF `Cactus-Compute/needle`.
- Baked-in objective weighting: tool **name** 2×, argument **values** 4×, argument **keys** 1.5×.
- Training reduced text-loss to ~**0.05** (perplexity ~1.05) — the model learned the JSON call format well.
- Best checkpoint: `checkpoints/needle_finetuned_20260820120014_*_best.pkl` (~52 MB).
- Hardware: **CPU-only Mac** (JAX CPU backend). Slow (~4 tok/s at eval, ~70 s/train-step), but fine
  for this scale.

---

## 5. Results — base vs finetuned (80-example held-out test set)

Metrics (needle's F1 methodology): **call F1** = full call (name+args) correctness; **name F1** = tool
name only; **exact match** = whole prediction correct; **args accuracy** = argument correctness;
per-tool = correct/10.

### Aggregate

| Metric | Base (un-finetuned) | Finetuned (PICKO) | Δ |
|---|---|---|---|
| call F1 | 36.0% | **60.0%** | **+24.0** |
| exact match | 36.2% | **60.0%** | **+23.7** |
| name F1 (tool selection) | 88.2% | **92.5%** | +4.3 |
| args accuracy (param extraction) | 40.8% | **64.9%** | **+24.1** |

**Reading it:** the gains are overwhelmingly in **parameter extraction** (+24 pts), not tool
selection (already high at 88%, +4 pts). Finetuning a tiny model on a few hundred examples/tool
teaches it to emit *correctly parameterized* calls — exactly the PICKO/D2 thesis.

### Per-tool (correct / 10) — BASE

| Tool | Base |
|---|---|
| arxiv_get_abstract | 10/10 |
| pubmed_download_article | 8/10 |
| hf_paper_search | 5/10 |
| wikipedia_search_wikipedia | 3/10 |
| pubmed_search_articles | 2/10 |
| wikipedia_extract_key_facts | 1/10 |
| arxiv_search_papers | 0/10 |
| hf_model_search | 0/10 |

### Reading the baseline (the PICKO thesis, visible before any finetuning)

- **name F1 (88%) ≫ call F1 (36%)**: the base model usually names a *plausible* tool but botches the
  **arguments** (args-acc 41%). Closing that gap is exactly **D2**.
- **Bimodal per-tool**: near-perfect on simple/single-param tools (`arxiv_get_abstract` 10/10) but
  **0/10** on the multi-param search tools (`arxiv_search_papers`, `hf_model_search`) — the hard D2/D3 cases.

### Per-tool (finetuned) — selection vs parameter extraction

| Tool | selection | args-exact | param F1 |
|---|---|---|---|
| arxiv_get_abstract | 1.00 | 1.00 | 1.00 |
| pubmed_search_articles | 0.90 | 0.78 | 0.94 |
| pubmed_download_article | 1.00 | 0.70 | 0.92 |
| wikipedia_extract_key_facts | 0.90 | 0.89 | 0.97 |
| hf_model_search | 1.00 | 0.50 | 0.70 |
| arxiv_search_papers | 0.90 | 0.44 | 0.83 |
| hf_paper_search | 0.90 | 0.44 | 0.79 |
| wikipedia_search_wikipedia | 0.80 | 0.38 | 0.80 |

The multi-parameter tools (`arxiv_search_papers`, `hf_model_search`) are *selected* almost perfectly
but remain the hardest for full-argument extraction — the D2 frontier. `wikipedia_search_wikipedia`
also shows the lowest selection (0.80), a D3 signal (confused with `wikipedia_extract_key_facts`).

> Recovered via `scripts/eval_picko.py` / `scripts/picko_eval.py` on the identical seed-42 test split
> after the original run's temp logs were cleared (checkpoint intact); numbers match the original run.

---

## 6. Files & how to reproduce

```bash
# 0. env (once)
source ./setup            # or: python -m venv .venv && pip install -e . && pip install -U jax
export GEMINI_API_KEY=...  # from https://aistudio.google.com/apikey  (kept in .env)

# 1. build the 8-tool subset
python scripts/make_scientific_pools.py

# 2. generate training data (coverage-driven; ~few min)
python scripts/generate_picko_data.py --workers 3 --target-per-tool 140

# 3. validate
python scripts/check_picko_data.py data/picko_subset.jsonl

# 4. finetune (+ auto base-vs-finetuned eval)
needle finetune data/picko_subset.jsonl --epochs 3 --batch-size 32

# 5. (re)compute metrics from a saved checkpoint
python scripts/eval_picko.py data/picko_subset.jsonl checkpoints/needle_finetuned_*_best.pkl
```

Key artifacts:
- `data/picko_subset.jsonl` — the training set (1,935 examples).
- `data/scientific_pools.json`, `data/picko_subset_tools.json` — the 8-tool subset.
- `checkpoints/needle_finetuned_*_best.pkl` — the PICKO model.

### Notebooks (for the final research)

- **`notebooks/01_generate_data.ipynb`** — pick tools from the catalog, generate `{query,tools,answers}`
  via Gemini with a live progress bar, validate, and chart coverage.
- **`notebooks/02_finetune_and_analyze.ipynb`** — explore the **11 tool families / 4 categories**,
  select a subset, finetune (reuse-on-demand), and produce the separated **tool-selection vs
  parameter-extraction** metrics with a D2 complexity plot, a **D3 confusion heatmap**, and a **D1**
  tool-set-size curve — plus a written research-story section.

Both are backed by reusable modules: `scripts/tool_catalog.py` (catalog/family/category selection) and
`scripts/picko_eval.py` (the separated-metrics evaluator). The richer catalog lives in
`full_tools_53tools_11products.json` (75 tools, 11 families) with `11tools_11products.json` as the
one-per-family D3 set.

---

## 7. Next steps

- **D1 (tool-set size):** widen the subset to 10 / 20 / 40 tools (edit the `POOLS` dict in
  `make_scientific_pools.py`) and watch accuracy degrade.
- **Abstention:** we already generate `none`/`near_miss` negatives; measure how reliably PICKO
  declines when no listed tool fits.
- **Data quality:** the generated queries are English, single-shot; add harder disambiguation pairs
  and longer argument values to push D2/D3.
- **Compute:** move finetuning/eval off CPU (GPU/TPU) once scaling up — CPU eval is the bottleneck.

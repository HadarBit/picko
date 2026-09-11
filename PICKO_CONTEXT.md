# PICKO — Full Project Context

A single reference for everything in this project: purpose, data, code, findings, how to run, and
current state. Written for handoff / sharing with a teammate or a fresh session.

---

## 1. What PICKO is

**PICKO** fine-tunes **Needle** — Cactus's **26M-parameter JAX/Flax encoder–decoder tool-picker**
(a "Simple Attention Network", no FFN) — to select the right **scientific** tool and extract its
arguments **from a single prompt**. The research question: *how far can a tiny model go as a scientific
tool-picker?*

- **Authors:** Hadar Bitan & Hodaya Stern Zuckerman (Tom Hope's "LLMs for Science" course).
- **Repo:** https://github.com/HadarBit/picko · working branch **`hadar-work`**.
- **Stack:** JAX/Flax (not PyTorch/HF). CLI is the `needle` console script (`finetune`, `run`, `eval`, …).
- **Model I/O:** finetune consumes **JSONL** of `{"query", "tools", "answers"}` where `tools`/`answers`
  are **JSON-encoded strings**. Needs **≥120 examples/tool** (100 train / 10 val / 10 test).
  `needle finetune` auto-downloads base weights from HF `Cactus-Compute/needle` and prints base-vs-
  finetuned metrics.

### The three research dimensions (renamed for clarity)

| Old tag | Name | Question | Metric |
|---|---|---|---|
| D1 | **Breadth** (tool-set size) | How many tools can it choose among before it picks wrong? | **selection only** (`selection_acc`/`name_f1`) |
| D2 | **Depth** (parameter complexity) | Given the right tool, can it fill arguments — harder with more params? | `args_exact_acc`/`param_f1` **by #params** |
| D3 | **Separation** (disambiguation) | Can it tell near-identical tools apart? | selection + confusion **within look-alike groups** |

---

## 2. Tool catalogs & metadata

All catalog JSONs share one schema — a flat list of
`{"name","description","parameters":{<arg>:{"type","description","required"}}}` — and differ only in
*which* tools they hold.

| File | What |
|---|---|
| `full_tools_first_draft.json` | Original **40 tools**, 4 families (arxiv 14, hf 12, wikipedia 10, pubmed 4). Has a `descrptiion` typo (auto-fixed on load). Strict subset of the master. |
| `full_tools_53tools_11products.json` | **Master catalog** (source of truth): **75 tools**, **11 families** (arxiv 14, hf 12, kaggle 11, wikipedia 10, github 7, openalex 5, semantic_scholar 5, pubmed 4, crossref 4, google 2, unpaywall 1). *Filename says "53" but holds 75.* |
| `11tools_11products.json` | One representative tool per family (11) — the original D3/Separation gold set. |
| `tools_metadata.csv` | One row per the 11 reps: `category, source name, tool name, required args, total args`. Groups families into **4 categories**: Academic & Literature Search · Data & ML Repositories · General Knowledge & Web · Scholarly Metadata. (Messy CSV — parsed defensively.) |

**Relationship:** `first_draft` (40) ⊂ `master` (75); `11tools` (11) is a one-per-family slice.

---

## 3. Data files (`data/`, git-ignored)

Each line = one example `{"query","tools","answers"}`.

| File | What |
|---|---|
| `data/picko_balanced.jsonl` | **The working pool.** All **75 tools × exactly 120 + 120 negatives = 9,120 lines** (~26.6 MB). Built in Notebook 1 via Gemini generation → per-tool "force" → balance. **Everything downstream is scoped from this file.** |
| `data/picko_gen.jsonl` | Raw Gemini generation pool (superset, unbalanced). |
| `data/picko_subset.jsonl` | The original 8-tool experiment (1,935 lines). |
| `data/picko_subset_<N>tools.jsonl`, `data/picko_<tag>.jsonl` | Scoped subsets written at run time by the notebooks/runner. |
| `data/research_results.json` | All Breadth/Depth/Separation metrics (written by `run_research.py` / notebook 03). |

**Generation** (Notebook 1) uses the **Gemini API** (`gemini-3.1-flash-lite-preview`, free tier 15 RPM).
A 13-RPM global rate limiter + fail-fast quota handling live in `scripts/generate_picko_data.py`. The
key functions: `force_tool` (offer only one tool so Gemini must call it → guarantees coverage of niche
tools) and `balance_dataset` (trim to exactly N/tool). **`GEMINI_API_KEY` is in `.env`.**

---

## 4. Scripts (`scripts/`)

| Script | Role |
|---|---|
| `tool_catalog.py` | `Catalog` — loads the 75-tool master + CSV. `family_of`, `category_of`, `select_tools(families=/categories=/names=/one_per_family=/k=)`, `as_dataframe()`. **`restrict_dataset(...)`** (below) re-scopes a dataset to a study subset. |
| `research_sets.py` | Research config: `FOCUS_FAMILIES` (arxiv/hf/wikipedia/pubmed), `focus_names(cat)`, `BREADTH_SIZES=[3,5,10,20,30]`, `nested_sets()`, `SIMILAR_GROUPS` (6 curated look-alike groups), `param_bucket()`/`PARAM_BUCKET_ORDER`. |
| `picko_eval.py` | The evaluator. `load_model`, `predict` (constrained greedy decode, batched), **`evaluate`** (separates tool-selection from parameter-extraction; per-tool & per-family), **`confusion`** (Separation matrix), `build_tools_override` (offer gold + k−1 distractors), `tools_token_len`/`n_visible` (encoder-truncation annotation), `base_checkpoint()`. |
| `generate_picko_data.py` | Gemini synthesis: rate limiter, `force_tool`, `balance_dataset`, quota fail-fast. |
| `run_research.py` | **Headless runner** = the unattended twin of Notebook 3. Runs Baseline · Breadth · Depth · Separation; resumable; honors `PICKO_OUT_DIR`. |
| `check_picko_data.py`, `eval_picko.py`, `make_scientific_pools.py` | Data validation, checkpoint re-eval, original 8-tool pools. |

### `Catalog.restrict_dataset(examples, names, ...)` — the core transform
Generated examples offer `gold + random distractors from all 75 tools`. To study a subset cleanly we
(a) keep only rows whose gold tool ∈ `names`, and (b) **rewrite each row's offered `tools` to that
subset**. Knobs:
- `offer_all_max` — offer all subset tools when small, else `gold + sampled distractors`.
- `cap_per_tool` — subsample N positives/tool (faster training).
- `compact=True` — offer `{name, description}` only (drop `parameters`) — for **Breadth**.
- `tokenizer=…, max_tokens=900` — greedily drop distractors (never gold) until the offered JSON fits —
  used for **Depth** so nothing is silently truncated.

---

## 5. Notebooks (`notebooks/`)

- **`01_generate_data.ipynb`** — Gemini generation with progress bars: pick tools → configure (RPM=13) →
  broad pass → **force every tool ≥ target** → **balance to exactly N/tool** → validate → visualize.
  Produced `data/picko_balanced.jsonl`.
- **`02_finetune_and_analyze.ipynb`** — the subset **baseline** notebook: pick a `SUBSET`, re-scope
  `picko_balanced.jsonl` to it (cell 3a/3b), finetune (`RUN_TRAIN`/`RUN_FINETUNE`), and show base-vs-
  finetuned (selection vs parameter extraction) + a simple D1/D3.
- **`03_research_dimensions.ipynb`** — **the research deliverable.** Focus = Original 40 (arxiv, hf,
  wikipedia, pubmed). Sections:
  - **0 · Colab quick-start** (GPU): clone repo, pin JAX/Flax, mount Drive, run headless, resumable.
  - **1 · Setup** + **1b · View saved results** (self-contained, reads `research_results.json`).
  - **2 · Baseline** (3 diverse tools).
  - **3 · Breadth** — finetune one model **per nested size** (3⊂5⊂10⊂20⊂30), **compact** schemas,
    selection-only; curve of `selection_acc` vs k with an `n_visible` truncation annotation.
  - **4 · Depth** — one model on all 40 (full schemas), metrics **bucketed by parameter count** (0/1/2-3/4+).
  - **5 · Separation** — reuse the 40-tool model on the 6 `SIMILAR_GROUPS`; per-group selection + confusion heatmaps.
  - **6 · Story.**

Shared helper `finetune_and_eval(...)`: re-scope → write JSONL → `needle finetune` → copy checkpoint to
a stable `picko_<tag>_best.pkl` → decode → `evaluate`. Resumable (skips tags whose checkpoint exists).

---

## 6. Key technical findings

1. **The "~11-tool limit" is the context window, not model capacity.** Needle's inference encoder packs
   `[query, <tools>, all schemas]` and **truncates to 1024 tokens** (`needle/model/run.py`
   `_build_encoder_input`). Measured on this catalog: a **full** schema ≈ **124 tokens** (median 107,
   max 434) → only **~8–9 full tools fit**; **compact** (name+description) fits up to **~20** (30
   compact tools → only ~21 visible). Beyond that, later tools are silently cut and the model never
   sees them. → **Breadth trains a model per size with compact schemas and reads selection only.**
2. **Metrics must separate selection from parameters.** `name_f1`/`selection_acc` = which tool (Separation);
   `args_exact_acc`/`param_f1` = arguments (Depth); `call_exact`/`call_f1` = both.
3. **Compact-schema runs have meaningless parameter metrics by design** — the model can't see argument
   definitions, so `args_acc`/`call_f1` are ~0. Ignore them for Breadth.
4. **Constrained decoding** forces valid tool names, so `picko_eval` selection numbers read *higher*
   than `needle`'s raw internal eval. Report the notebook's numbers, not the training log's.

### Results so far
- **8-tool run** (`picko_subset.jsonl`): finetune moved param-extraction **41%→65%**, call F1 **36%→60%**,
  selection (name F1) **88%→92%** — win is parameter extraction.
- **11-per-family run**: selection **0.90→0.96**, args_exact **0.33→0.52**, call_exact **0.30→0.50**.
- **Breadth k=10 (lean cap=40, 1 epoch)**: selection ≈ **58%** (internal eval) — modest, config is lean.

---

## 7. How to run

### Locally (CPU, ~112 s/step)
```bash
source ./setup                       # venv + Needle + JAX (once)
export GEMINI_API_KEY=...            # in .env (only needed for data generation)
# headless research sweep (all ~6 finetunes); keep the Mac awake:
caffeinate -is nohup .venv/bin/python -u scripts/run_research.py --cap-per-tool 40 > research.log 2>&1 &
tail -f research.log
```
Or drive `notebooks/03_research_dimensions.ipynb` with the `picko` kernel, `RUN_TRAIN=True`.

### Google Colab (GPU — much faster)
Open the notebook from GitHub (branch `hadar-work`), **Runtime → GPU (T4)**, then run **Section 0**:
- **0a** clones the repo, `pip install`s **pinned `jax[cuda12]==0.10.2` + `jaxlib==0.10.2` + `flax==0.12.8`**,
  mounts Drive, sets `PICKO_OUT_DIR=/content/drive/MyDrive/picko_out`, copies the data file.
- **0b** runs `run_research.py` on the GPU (writes checkpoints + results to **Drive** → **restart-safe**;
  re-run to resume).
- **1b** renders all tables/plots from `research_results.json`.

Upload `data/picko_balanced.jsonl` (9,120 lines, ~26.6 MB) to Drive once — it's git-ignored so **not** in
the clone. The 26 MB Colab file-picker upload often truncates → prefer Drive.

---

## 8. Gotchas / lessons

- **Run finetune via the `needle` console script**, not `python -m needle.cli` (no `__main__` guard).
  In subprocess, resolve it as `shutil.which("needle") or <venv>/bin/needle` (PATH isn't guaranteed).
- **JAX/Flax versions are pinned:** working set is **jax 0.10.2 / jaxlib 0.10.2 / flax 0.12.8**. Colab's
  default (jax ≥0.11 with an old flax) crashes with `get_opaque_trace_state` / `jax.checkpoint(concrete=)`
  errors. Pin both.
- **GPU not used?** A step at ~148 s means it's still on CPU. Check `jax.devices()` shows a CUDA device
  and don't set `JAX_PLATFORMS=cpu` on Colab.
- **HF 403 on upload** ("can't create model under Cactus-Compute") after training is **harmless** — the
  checkpoint saved locally; needle just tried to push to the author's namespace.
- **Apple M2 GPU (Metal) is unusable** — `jax-metal` doesn't support JAX 0.10.2 and is experimental. CPU only on Mac.
- **Colab isn't truly "background"** on free tier (idle ~90 min, 12 h max). Drive persistence + the
  resumable runner make restarts safe; keep the tab open or use Colab Pro for long runs.
- **macOS sleep pauses (doesn't kill) a process**; `caffeinate -is` on AC power keeps it computing.

---

## 9. Current state & next steps

- ✅ `data/picko_balanced.jsonl` complete (75×120 + negatives). Helper chain validated end-to-end.
- ✅ Notebook 03 + `run_research.py` built, Drive-persistent & resumable; Colab GPU setup added.
- 🔬 **In progress:** first sweep at `cap=40` (quick). Then rerun at **`cap=120`** for presentable numbers.
- ⏭ **Pending push:** the new/updated files (`scripts/{run_research,research_sets,tool_catalog,picko_eval}.py`,
  `notebooks/02…`, `notebooks/03…`) are committed locally? — must be **pushed to `hadar-work`** for the
  Colab GitHub link to see them.
- ⏭ Fill in the Notebook 3 "Story" section with the final Breadth/Depth/Separation numbers.

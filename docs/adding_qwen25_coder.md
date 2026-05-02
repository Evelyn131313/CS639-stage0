# Adding Qwen2.5-Coder-7B-Instruct to Stage 0

## Overview

This document explains how to add `Qwen/Qwen2.5-Coder-7B-Instruct` as a third model
(`qwen25_coder_7b`) alongside the existing `qwen3_8b_think` and `qwen3_8b_nothink` runs.
The model has no thinking mode and uses its own weights, so it must be loaded and unloaded
separately from Qwen3-8B.

After these changes the pipeline produces **300 (problem, run_name) pairs** (3 models × 100
problems) instead of 200.

---

## Configuration

| Field | Value |
|---|---|
| Run name | `qwen25_coder_7b` |
| HuggingFace model ID | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Temperature | 0.7 |
| top\_p | 0.95 |
| top\_k | 20 |
| max\_tokens | 4096 |
| seed | 0 |
| Thinking mode | OFF |
| dtype | bfloat16 |
| gpu\_memory\_utilization | 0.85 |

---

## Files Changed

| File | Section / Location | What changes |
|---|---|---|
| `notebooks/stage0_generation_roleBC.ipynb` | Section 1 — Locked Configuration | Add `qwen25_coder_7b` to `MODELS` and `RUN_NAMES` |
| `notebooks/stage0_generation_roleBC.ipynb` | Section 5 — Model Configuration | Add entry to `MODEL_CONFIGS` |
| `notebooks/stage0_generation_roleBC.ipynb` | Section 7 — Kernel Generation | Load/unload coder model separately after Qwen3-8B |
| `scripts/merge_shards.py` | `RUN_NAMES` constant (line 53) | Add `"qwen25_coder_7b"` |
| `scripts/run_official_benchmark.py` | `RUN_NAMES` constant (line 55) | Add `"qwen25_coder_7b"` |
| `scripts/run_official_benchmark.py` | Steps 4–7 | Add Step 6 for coder eval; shift old Steps 6–7 to 7–8 |

---

## Step-by-Step Changes

### 1. Generation Notebook — Section 1

Find the **Locked Configuration** cell and add the new model name:

```python
# BEFORE
MODELS    = ['qwen3_8b_think', 'qwen3_8b_nothink']
RUN_NAMES = ['qwen3_8b_think', 'qwen3_8b_nothink']

# AFTER
MODELS    = ['qwen3_8b_think', 'qwen3_8b_nothink', 'qwen25_coder_7b']
RUN_NAMES = ['qwen3_8b_think', 'qwen3_8b_nothink', 'qwen25_coder_7b']
```

Section 4 (env\_info) already loops over `RUN_NAMES`, so no change needed there — it will
automatically create a `runs/qwen25_coder_7b/` directory and write `env_info.json`.

### 2. Generation Notebook — Section 5

Add a new entry to `MODEL_CONFIGS` after the existing `qwen3_8b_nothink` block:

```python
"qwen25_coder_7b": {
    "model_id":   "Qwen/Qwen2.5-Coder-7B-Instruct",
    "run_name":   "qwen25_coder_7b",
    "load_kwargs": {"dtype": "bfloat16", "gpu_memory_utilization": 0.85},
    "sampling_params": SamplingParams(
        temperature=0.7, top_p=0.95, top_k=20, max_tokens=4096, seed=0
    ),
    "thinking": False,
    "chat_template_kwargs": None,
},
```

### 3. Generation Notebook — Section 7

Because `Qwen2.5-Coder-7B-Instruct` has **different weights** from `Qwen3-8B`, it must be
loaded in a separate vLLM instance. Modify `generate_for_role` to accept an explicit list of
model tags, then load each model group separately:

```python
def generate_for_role(llm: LLM, model_tags: list[str]) -> None:
    for model_tag in model_tags:           # ← was: for model_tag in MODELS
        ...                                #   (rest of body unchanged)
```

Then replace the single load block at the bottom of the cell with:

```python
# ── Qwen3-8B: think + nothink (shared weights) ───────────────────────────
print("Loading Qwen3-8B (shared weights for think + nothink) ...")
llm = load_vllm_model("qwen3_8b_think")
generate_for_role(llm, model_tags=["qwen3_8b_think", "qwen3_8b_nothink"])
unload_vllm_model(llm)

# ── Qwen2.5-Coder-7B-Instruct ────────────────────────────────────────────
print("Loading Qwen2.5-Coder-7B-Instruct ...")
llm = load_vllm_model("qwen25_coder_7b")
generate_for_role(llm, model_tags=["qwen25_coder_7b"])
unload_vllm_model(llm)

print("\n✓ Generation complete.")
```

### 4. `scripts/merge_shards.py`

```python
# BEFORE (line 53)
RUN_NAMES  = ["qwen3_8b_think", "qwen3_8b_nothink"]

# AFTER
RUN_NAMES  = ["qwen3_8b_think", "qwen3_8b_nothink", "qwen25_coder_7b"]
```

`EXPECTED_PAIRS` is computed as `len(RUN_NAMES) * len(ALL_PROBLEMS)` so it automatically
becomes **300**. No other changes needed.

### 5. `scripts/run_official_benchmark.py`

```python
# BEFORE (line 55)
RUN_NAMES = ["qwen3_8b_think", "qwen3_8b_nothink"]

# AFTER
RUN_NAMES = ["qwen3_8b_think", "qwen3_8b_nothink", "qwen25_coder_7b"]
```

Also add `"eval_results_coder.csv"` to `REQUIRED_OUTPUTS`, and add a new **Step 6** between
the nothink eval and the analysis step:

```python
banner("Step 6 — Evaluate qwen25_coder_7b kernels")
run_cmd(
    [
        sys.executable, str(eval_script),
        "run_name=qwen25_coder_7b",
        f"level={LEVEL}",
        "dataset_src=huggingface",
        "backend=triton",
        f"num_correct_trials={args.num_correct_trials}",
        f"num_perf_trials={args.num_perf_trials}",
        f"runs_dir={runs_dir}",
    ],
    step="eval qwen25_coder_7b",
    dry_run=dry_run,
)
```

Then renumber the old Step 6 (analysis) → **Step 7** and old Step 7 (checklist) → **Step 8**.

---

## Running the New Experiment

### Roles B and C (Colab generation)

No change to which role runs which problem IDs. After updating the notebook:

1. Set `ROLE = "B"` or `"C"` in Section 0 as usual.
2. Run all cells top-to-bottom.
3. Section 7 will now run **three** generation passes — Qwen3-8B (think + nothink, shared
   load) then Qwen2.5-Coder-7B-Instruct (separate load).
4. The verification in Section 8 and the zip in Section 9 now include a third run directory.

> **Expected extra time:** ~5–10 min additional generation for the coder model per role.

### Role A — Merge

```bash
python scripts/merge_shards.py \
    --shard_b shards/roleB_problems_1_50 \
    --shard_c shards/roleC_problems_51_100 \
    --runs_dir runs
```

The merged `runs/` will now contain three subdirectories and the CSVs will have **300 rows**
each (3 models × 100 problems).

### Role A — Official Benchmark

```bash
python scripts/run_official_benchmark.py --runs_dir runs --levels 1
```

The script now runs eval for all three run names and produces `eval_results_coder.csv` in
addition to the existing think/nothink CSVs.

---

## Backward Compatibility

These changes are **additive only**:

- Existing `qwen3_8b_think` / `qwen3_8b_nothink` outputs are untouched.
- The merge script will reject shards that are missing the new run directory — if you are
  re-merging old shards that pre-date this change, generate the coder kernels first or
  temporarily revert `RUN_NAMES` in `merge_shards.py` for the legacy merge.

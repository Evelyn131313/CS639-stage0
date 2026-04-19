# CS 639 Stage 0 — Role A (Unified Environment / Merge / Official Evaluation)

This repository is the working directory for **Role A** of the **CS 639 Project Stage 0 (evaluation)**.

Role A is **not** responsible for generating large numbers of kernels. Instead, A owns the **shared foundation** for the entire Stage 0 effort: standing up the unified environment, merging the shards produced by B and C, and running the **single, authoritative benchmark** at the end. All final speedup numbers in this stage come exclusively from A's unified-session run.

---

## 1. Goals of This Stage

This stage covers **Stage 0: evaluation only** — no SFT training and no RLVR training.

The goal is to build a solid Triton-kernel-generation baseline for the base model (Qwen3-8B) on **KernelBench Level 1**, leaving Stage 1 / Stage 2 a directly reusable foundation consisting of:

- a unified environment
- a unified prompt / sampling configuration
- a unified `runs/` directory layout
- a unified evaluation method
- unified result tables and plots

### Stage-Wide Ground Rules

1. The official benchmark covers **KernelBench Level 1 only**.
2. The official model lineup is Qwen3-8B in two modes: `qwen3_8b_think` and `qwen3_8b_nothink`.
3. **Pilot first, then full run** (a small smoke test before committing to all 100 problems).
4. B and C generate kernels in parallel on their own Colab instances (B: 1–50, C: 51–100).
5. A owns the unified environment, the merge step, and the official evaluation.
6. The official speedup numbers come **only** from A's unified-session run; the timings B/C see in their own Colabs are for debugging only and never enter the main results table.

> Why this split: the team has no shared, fixed compute. Each member's Colab GPU, runtime, and session length is variable, so naively stitching three sets of timings into one table would be invalid. Stage 0 therefore uses **distributed generation (B/C in parallel) + a single unified official benchmark (owned by A)**.

---

## 2. Directory Layout

```
project_A/
├── KernelBench/                       # Upstream KernelBench repo (problem set + eval scaffolding)
│   ├── KernelBench/                   # Level 1/2/3 problem sources
│   ├── src/                           # Core eval code
│   ├── scripts/                       # Upstream eval scripts
│   ├── EVAL.md                        # Upstream eval notes
│   └── README.md                      # Upstream README
│
├── notebooks/
│   ├── open_source_triton_benchmark.ipynb   # Starting notebook (basis for the unified template)
│   └── stage0_generation_roleBC.ipynb       # Unified generation notebook used by B and C
│
├── scripts/                           # Stage 0 pipeline scripts owned by A
│   ├── run_smoke_test.py              # Pilot stage: merge + eval smoke test
│   ├── merge_shards.py                # Merge B/C uploads into the official runs/ dir
│   ├── run_official_benchmark.py      # Run the official benchmark in a single session
│   └── benchmark_analysis.py          # Produce the final summary tables and plots
│
├── shards/                            # Per-shard uploads from B and C (consumed by A's merge)
│   ├── roleB_problems_1_50/
│   └── roleC_problems_51_100/
│
├── runs/                              # Official runs/ produced by A's merge (single source of truth)
│   └── stage0_qwen3_8b/
│       ├── qwen3_8b_think/
│       └── qwen3_8b_nothink/
│
└── results/                           # Final summary tables, per-problem tables, and plots
```

---

## 3. Role A's Concrete Responsibilities

| # | Responsibility                                                                                                                   | Output                                |
| - | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| 1 | Take `notebooks/open_source_triton_benchmark.ipynb` and turn it into a **unified Colab notebook / script** for the team. | Shared generation notebook for B/C    |
| 2 | Lock down stage-wide configuration:`LEVELS = [1]`, fixed run names, fixed prompt construction.                                 | Constant config                       |
| 3 | Standardize environment metadata: GPU model, CUDA / torch / triton / transformers / vLLM versions, notebook runtime version.     | Environment metadata file             |
| 4 | Build the **shard merge script** that consolidates B's and C's uploads into the official `runs/` tree.                  | `scripts/merge_shards.py`           |
| 5 | In a single unified session, generate baseline timings, run `eval_from_generations.py`, and produce the benchmark analysis.    | `scripts/run_official_benchmark.py` |
| 6 | Produce the final Stage 0 summary tables and plots.                                                                              | Files under `results/`              |

### What A does **not** do

- A does **not** do large-scale shard generation.
- A does **not** backfill problems for B / C unless someone is severely behind.

---

## 4. Stage-Wide Configuration (do not change locally)

### Models and run names

| Model             | Run name             |
| ----------------- | -------------------- |
| Qwen3-8B-think    | `qwen3_8b_think`   |
| Qwen3-8B-no-think | `qwen3_8b_nothink` |

### Problem ranges

- Official benchmark: **KernelBench Level 1** only.
- B: problems 1–50
- C: problems 51–100

### Generation configuration

- **think mode**: Qwen3 chat template with thinking enabled.
- **no-think mode**: same model, same overall pipeline, with thinking disabled.
- Every generation must persist three artifacts: **raw / clean / extracted kernel**.

### Static-validity output fields (`validity_flags.csv` must contain at minimum)

`problem_id`, `run_name`, `has_triton_jit`, `torch_calls_in_jit`, `is_valid`, `needs_manual_review`, `notes`

### Generation manifest fields (`generation_manifest.csv` must contain at minimum)

`problem_id`, `run_name`, `status`, `raw_path`, `clean_path`, `kernel_path`, `meta_path`, `extraction_success`, `output_length`, `error_message`

---

## 5. Workflow (A's view)

### Step 1 — Unified environment + smoke test

1. A: publish the unified notebook and the locked configuration to B and C.
2. B: run problems 1–5. C: run problems 51–55.
3. A: take those two small batches and run a **merge + eval smoke test**:
   ```bash
   python scripts/run_smoke_test.py
   ```

### Step 2 — Pilot passes, move to full run

If the smoke test reveals no structural issues:

- B finishes 1–50; C finishes 51–100.
- A keeps refining `merge_shards.py`, `run_official_benchmark.py`, and the baseline script in parallel.

### Step 3 — Official evaluation

Once all shards from B and C have arrived, A runs the following on a **single machine, in a single session**:

```bash
# 1) Merge B/C shards into the official runs/
python scripts/merge_shards.py \
    --shards-dir shards \
    --runs-dir runs/stage0_qwen3_8b

# 2) Generate baselines and run the official benchmark. This also produces the summary tables and plots
python scripts/run_official_benchmark.py \
    --runs-dir runs/stage0_qwen3_8b \
    --levels 1
```

### Step 4 — Result roll-up

- A produces the tables and plots.
- One teammate then integrates them with the narrative and reporting deliverables.

---

## 6. Handoff Requirements from B / C to A

To avoid the situation where there is "a kernel file that no one can reproduce", every batch B or C hands to A should include:

1. All generated artifacts for that batch (per problem: `raw.txt`, `clean.txt`, `kernel.py`, `meta.json`).
2. `generation_manifest.csv` for that batch.
3. `validity_flags.csv` for that batch.
4. A short environment snapshot: GPU model, torch version, triton version, runtime version.
5. The range of problem ids completed in this batch.
6. A list of currently unresolved issues.

A only accepts batches dropped under `shards/roleB_problems_1_50/` or `shards/roleC_problems_51_100/` in this layout.

---

## 7. Final Stage 0 Deliverables

The complete Stage 0 deliverable for the week consists of:

- All Qwen3-8B think / no-think generations for Level 1.
- A unified `runs/` directory.
- Static-validity check results.
- The official benchmark summary.
- The per-problem result table.
- A shared foundation that Stage 1 / Stage 2 can build on directly.

---

## 8. Environment

A records the environment metadata from the unified-session run to `runs/stage0_qwen3_8b/env.json`:

- GPU model
- CUDA / torch / triton / transformers / vLLM versions
- notebook runtime version

For dependencies, see `KernelBench/requirements.txt`.

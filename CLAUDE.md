# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Role A** working directory for CS 639 Stage 0. Role A owns the unified environment, the shard merge step, and the single authoritative benchmark run. Roles B and C each generate kernels on Colab (B: problems 1–50, C: 51–100) using `notebooks/stage0_generation_roleBC.ipynb`, then hand off zip archives to A for merging and evaluation.

The model under evaluation is **Qwen3-8B** in two modes (`qwen3_8b_think` / `qwen3_8b_nothink`) on **KernelBench Level 1** (100 problems). All official speedup numbers come exclusively from A's unified-session run.

## KernelBench Setup

KernelBench is a git submodule at `./KernelBench/`. It requires Python 3.10 and a GPU for kernel execution.

```bash
# Install KernelBench (from the repo root)
pip install -e KernelBench/

# Or with uv (inside the KernelBench subdir)
cd KernelBench && uv sync --extra gpu
```

Copy `.env.example` to `.env` inside `KernelBench/` and fill in any required API keys for LLM providers.

## Stage 0 Pipeline Commands

### Step 1 — Merge B/C shards into `runs/`

```bash
python scripts/merge_shards.py \
    --shard_b shards/roleB_problems_1_50 \
    --shard_c shards/roleC_problems_51_100 \
    --runs_dir runs
```

Accepts either a directory or a `.zip` file for `--shard_b`/`--shard_c`. After merging, `runs/` contains unified `generation_manifest.csv` and `validity_flags.csv` (200 rows each: 2 models × 100 problems).

### Step 2 — Run official benchmark

```bash
python scripts/run_official_benchmark.py --runs_dir runs --levels 1
```

This generates baseline timings, runs `KernelBench/scripts/eval_from_generations.py`, and writes results to `results/`.

### Smoke test (subset only)

```bash
python scripts/run_smoke_test.py
```

### KernelBench eval scripts (run from inside `KernelBench/`)

```bash
# Evaluate a single problem
uv run python scripts/generate_and_eval_single_sample.py dataset_src=local level=1 problem_id=1

# Batch evaluation of generated kernels
uv run python scripts/eval_from_generations.py run_name=qwen3_8b_nothink dataset_src=local level=1 num_gpu_devices=1

# Analyze results and compute fast_p metric
uv run python scripts/benchmark_eval_analysis.py run_name=qwen3_8b_nothink level=1 hardware=<your_gpu>

# Generate baseline times for your GPU
uv run python scripts/generate_baseline_time.py
```

## Architecture

### Data flow

```
B/C Colab sessions
  └─ notebooks/stage0_generation_roleBC.ipynb
       ├─ generates: raw.txt, clean.txt, kernel.py, meta.json  (per problem × per run_name)
       ├─ writes:    generation_manifest.csv, validity_flags.csv  (per run_name dir)
       └─ exports:   role_{B|C}_shard.zip

Role A
  └─ scripts/merge_shards.py       → consolidates into runs/
  └─ scripts/run_official_benchmark.py → eval + results/
```

### `runs/` layout (post-merge)

```
runs/
├── qwen3_8b_think/
│   ├── level_1_problem_{N}_raw.txt
│   ├── level_1_problem_{N}_clean.txt
│   ├── level_1_problem_{N}_sample_0_kernel.py
│   ├── level_1_problem_{N}_meta.json
│   ├── env_info_roleB.json / env_info_roleC.json
│   └── (validity_flags.csv and generation_manifest.csv in per-role shards; merged at runs/ root)
├── qwen3_8b_nothink/   (same layout)
├── generation_manifest.csv   ← unified, 200 rows
├── validity_flags.csv        ← unified, 200 rows
├── env_info_merged.json
└── merge_report.txt
```

### Kernel validity checking

`notebooks/stage0_generation_roleBC.ipynb` (Section 6) contains an AST-based static validity checker (`check_triton_validity`) that verifies generated kernels have `@triton.jit` decorators and no torch calls inside JIT-decorated functions. Results go into `validity_flags.csv`.

### Locked configuration (do not modify)

- `LEVELS = [1]`
- Models/run names: `qwen3_8b_think` (temp=0.6, top_p=0.95, thinking=ON), `qwen3_8b_nothink` (temp=0.7, top_p=0.8, thinking=OFF)
- Both modes use `Qwen/Qwen3-8B` with vLLM, `dtype=bfloat16`, `gpu_memory_utilization=0.85`
- Problem split: B=1–50, C=51–100

### KernelBench benchmark metric

`fast_p` = fraction of problems where the generated kernel is both correct **and** achieves speedup ≥ p over PyTorch reference. `fast_0` = correctness rate, `fast_1` = fraction faster than PyTorch, `fast_2` = fraction ≥ 2× faster.

## Notebooks

- `run_official_benchmark.ipynb` — Role A's Colab notebook for the official benchmark (runs on A100 40 GB)
- `notebooks/stage0_generation_roleBC.ipynb` — canonical generation notebook for B/C; only change is setting `ROLE = "B"` or `"C"` in Section 0
- `notebooks/open_source_triton_benchmark.ipynb` — original starting point / reference
# Stage 0 Experiment Pipeline

## Overview

We evaluated three LLM configurations on **KernelBench Level 1**, a benchmark of 100 PyTorch operator implementations. Each model was prompted to rewrite a given PyTorch reference into an equivalent Triton GPU kernel. Generated kernels were compiled and executed on real hardware, then compared against the PyTorch baseline timing to compute speedup metrics.

The pipeline is divided into two phases: **distributed kernel generation** (Roles B and C, running in parallel on Google Colab) and a **single unified evaluation** (Role A), which is the sole source of official results.

---

## Models

Three configurations were evaluated:

| Run Tag | HuggingFace Model ID | Thinking Mode |
|---|---|---|
| `qwen3_8b_think` | `Qwen/Qwen3-8B` | Enabled (`<think>` chain-of-thought) |
| `qwen3_8b_nothink` | `Qwen/Qwen3-8B` | Disabled |
| `qwen25_coder_7b` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Disabled |

`qwen3_8b_think` and `qwen3_8b_nothink` use the same model weights but differ in sampling strategy. The think variant activates Qwen3's built-in chain-of-thought reasoning by injecting the `<think>` token, while the nothink variant suppresses it and generates answers directly. `qwen25_coder_7b` is a separate code-specialized model evaluated as a baseline.

---

## Inference Parameters

All models were served with **vLLM** using the following common model-loading configuration:

| Parameter | Value |
|---|---|
| Serving backend | vLLM |
| dtype | `bfloat16` |
| GPU memory utilization | `0.85` |
| Random seed | `0` |

Per-model sampling parameters:

| Run Tag | temperature | top_p | top_k | max_tokens |
|---|---|---|---|---|
| `qwen3_8b_think` | 0.6 | 0.95 | 20 | 16 384 |
| `qwen3_8b_nothink` | 0.7 | 0.80 | 20 | 4 096 |
| `qwen25_coder_7b` | 0.7 | 0.95 | 20 | 4 096 |

> **Note on `qwen3_8b_think` parameters:** The Qwen3-8B model card warns against greedy decoding (temperature = 0) when the `<think>` mode is active, as it leads to degenerate repetitive outputs. We therefore use temperature = 0.6, as recommended. The `max_tokens` budget is set to 16 384 (versus 4 096 for the other two) to give the model enough room to complete its full reasoning chain before producing the final kernel.

---

## Hardware and Software Environment

### Generation Phase

Kernel generation was distributed across two roles running in parallel on Google Colab. Role B handled problems 1–50; Role C handled problems 51–100.

| Role | Problems | GPU |
|---|---|---|
| Role B | 1–50 | NVIDIA A100-SXM4-40 GB |
| Role C (`qwen3` models) | 51–100 | NVIDIA A100-SXM4-40 GB |
| Role C (`qwen25_coder_7b`) | 51–100 | NVIDIA A100-SXM4-80 GB |

Software stack during generation:

| Package | `qwen3_8b_think` / `qwen3_8b_nothink` | `qwen25_coder_7b` |
|---|---|---|
| CUDA | 12.8 | 12.8 |
| PyTorch | 2.10.0+cu128 | 2.8.0+cu128 |
| Triton | 3.6.0 | 3.4.0 |
| vLLM | 0.19.1 | 0.10.2 |
| Transformers | 5.6.2 | 4.56.1 |

### Evaluation Phase

All generated kernels were merged and evaluated in a **single unified Role A session** to ensure consistent baseline timings. No timing numbers from the generation Colabs enter the final results table.

| Item | Value |
|---|---|
| GPU | NVIDIA L4 |
| VRAM | 23.7 GB |
| CUDA | 12.8 |
| PyTorch | 2.10.0+cu128 |
| Triton | 3.6.0 |
| Transformers | 5.0.0 |
| Python | 3.12.13 |
| OS | Linux 6.6 (x86\_64, glibc 2.35) |

---

## Pipeline Steps

### Step 1 — Kernel Generation (Roles B and C)

Roles B and C each ran `notebooks/stage0_generation_roleBC.ipynb` in independent Colab sessions. For every (problem, model) pair, the notebook:

1. Constructs a prompt from the KernelBench Level 1 problem statement.
2. Calls the vLLM-served model to generate a Triton kernel.
3. Saves four artifacts per problem: `raw.txt` (full model output), `clean.txt` (extracted code block), `kernel.py` (importable module), and `meta.json` (generation metadata).
4. Runs an AST-based static validity check (`check_triton_validity`) that verifies each kernel has a `@triton.jit` decorator and no raw `torch` calls inside JIT-decorated functions. Results are written to `validity_flags.csv`.

### Step 2 — Shard Merge (Role A)

Role A merged the two zip archives from B and C into a unified `runs/` directory:

```bash
python scripts/merge_shards.py \
    --shard_b shards/B_shard \
    --shard_c shards/C_shard \
    --runs_dir runs
```

This produced a single `generation_manifest.csv` and `validity_flags.csv`, each with **300 rows** (3 models × 100 problems). The merge completed with zero errors.

### Step 3 — Official Evaluation (Role A)

Role A ran the official benchmark in a single session on the NVIDIA L4:

```bash
python scripts/run_official_benchmark.py --runs_dir runs --levels 1
```

The script: generated PyTorch/torch.compile baseline timings, compiled and executed all 300 kernels, and computed per-problem speedups. Final metrics were written to `results/`.

---

## Evaluation Metric

The primary metric is **fast\_p**: the fraction of problems where the generated kernel is both correct *and* achieves at least p× speedup over the PyTorch reference.

- `fast_0` — correctness rate (kernel compiles, runs, and produces numerically correct output)
- `fast_1` — correct and no slower than PyTorch (speedup ≥ 1×)
- `fast_2` — correct and at least 2× faster than PyTorch

---

## Results

### Kernel Validity (Static Check)

| Run Name | Valid Kernels | Needs Review | Total |
|---|---|---|---|
| `qwen3_8b_think` | 82 | 10 | 100 |
| `qwen3_8b_nothink` | 67 | 23 | 100 |
| `qwen25_coder_7b` | 91 | 9 | 100 |

### Benchmark Metrics (Official Run — NVIDIA L4)

| Run Name | fast\_0 (pass@1) | fast\_1 (≥ 1×) | Geomean Speedup |
|---|---|---|---|
| `qwen3_8b_think` | 3% | 0% | 0.037× |
| `qwen3_8b_nothink` | 0% | 0% | — |
| `qwen25_coder_7b` | 1% | 0% | 0.434× |

All three models achieved very low correctness rates on KernelBench Level 1 at this stage. None produced a kernel that was both correct and faster than PyTorch. The nothink variant of Qwen3-8B produced the fewest statically valid kernels (67/100) and zero correct results, suggesting that the chain-of-thought suppression hurts code quality on this task at the current scale.
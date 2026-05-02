#!/usr/bin/env python3
"""
CS 639 · Stage 0 — Official Benchmark Session Runner
=====================================================
Role A's single entry-point for the entire official benchmark session.
Run this ONCE in one uninterrupted Colab session on a single A100 GPU.

What this script does (in order):
  1.  Verify the merged runs/ directory passes pre-flight checks
  2.  Record Role A's environment → runs/env_info_roleA.json
  3.  Generate PyTorch / torch.compile baseline timings for all 100 Level 1 problems
  4.  Run eval_from_generations.py for qwen3_8b_think
  5.  Run eval_from_generations.py for qwen3_8b_nothink
  6.  Run benchmark_analysis.py → all CSVs and figures
  7.  Print final deliverable checklist

Usage
-----
    python run_official_benchmark.py [--runs_dir <path>] [--kernelbench_root <path>]
                                     [--out_dir <path>] [--dry_run]

Arguments
---------
    --runs_dir          Merged runs/ directory (default: ./runs)
    --kernelbench_root  Path to KernelBench checkout (default: ./KernelBench)
    --out_dir           Where to write output files (default: same as runs_dir)
    --num_correct_trials  Correctness trials per kernel (default: 5)
    --num_perf_trials     Performance trials per kernel (default: 50)
    --dry_run           Print commands but do not execute them

Exit codes
----------
    0   All steps completed successfully
    1   A step failed — see output above
    2   Pre-flight validation failed — fix runs/ before retrying
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import importlib
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ── Constants ──────────────────────────────────────────────────────────────
RUN_NAMES    = ["qwen3_8b_think", "qwen3_8b_nothink", "qwen25_coder_7b"]
LEVEL        = 1
ALL_PIDS     = list(range(1, 101))
EXPECTED_PAIRS = len(RUN_NAMES) * len(ALL_PIDS)   # 200

REQUIRED_OUTPUTS = [
    "env_info_roleA.json",
    "baseline_times.csv",
    "eval_results_think.csv",
    "eval_results_nothink.csv",
    "eval_results_coder.csv",
    "stage0_summary.csv",
    "speedup_distribution.png",
    "pass_rate_comparison.png",
]


# ── Helpers ────────────────────────────────────────────────────────────────
def log(msg: str, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"{prefix}{msg}", flush=True)


def banner(title: str) -> None:
    bar = "=" * 65
    print(f"\n{bar}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{bar}", flush=True)


def run_cmd(cmd: list[str], step: str, dry_run: bool) -> None:
    """Run a subprocess command, stream output live, raise on failure."""
    log(f"$ {' '.join(str(c) for c in cmd)}", indent=1)
    if dry_run:
        log("[DRY RUN] skipping execution\n", indent=1)
        return
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        sys.exit(
            f"\n[ERROR] Step '{step}' failed with exit code {result.returncode}.\n"
            "Fix the issue and re-run. Do NOT continue benchmarking with partial results."
        )
    log(f"  ✓ {step} complete.\n")


def get_version(pkg: str) -> str:
    try:
        return importlib.import_module(pkg).__version__
    except Exception:
        return "not installed"


# ── Step 1: Pre-flight checks ──────────────────────────────────────────────
def preflight(runs_dir: Path) -> None:
    """
    Verify the merged runs/ directory is complete before touching the GPU.
    Mirrors the Role A pre-benchmark checklist from the task guide.
    """
    errors = []

    # merge_report.txt must exist and show zero errors
    report_path = runs_dir / "merge_report.txt"
    if not report_path.exists():
        errors.append("merge_report.txt not found — run merge_shards.py first.")
    else:
        report_text = report_path.read_text()
        if "MERGE FAILED" in report_text:
            errors.append(
                "merge_report.txt shows MERGE FAILED — resolve all errors before benchmarking."
            )
        if "zero errors" not in report_text and "MERGE OK" not in report_text:
            errors.append(
                "merge_report.txt does not confirm zero errors — re-run merge_shards.py."
            )

    # Check all 200 (problem, run_name) pairs
    for run_name in RUN_NAMES:
        run_dir = runs_dir / run_name
        if not run_dir.is_dir():
            errors.append(f"Missing run directory: {run_dir}")
            continue
        for pid in ALL_PIDS:
            kpath = run_dir / f"level_1_problem_{pid}_sample_0_kernel.py"
            if not kpath.exists():
                errors.append(f"Missing kernel: {run_name}/level_1_problem_{pid}_sample_0_kernel.py")
            elif kpath.stat().st_size == 0:
                errors.append(f"Empty kernel: {run_name}/level_1_problem_{pid}_sample_0_kernel.py")

    # CSV row counts
    import csv
    for csv_name, expected in [
        ("generation_manifest.csv", EXPECTED_PAIRS),
        ("validity_flags.csv",      EXPECTED_PAIRS),
    ]:
        csv_path = runs_dir / csv_name
        if not csv_path.exists():
            errors.append(f"Missing {csv_name} in runs/ root — re-run merge_shards.py.")
            continue
        with open(csv_path, newline="") as f:
            n = sum(1 for _ in csv.DictReader(f))
        if n != expected:
            errors.append(f"{csv_name} has {n} rows, expected {expected}.")

    if errors:
        print("\n[PRE-FLIGHT FAILED]")
        for e in errors:
            log(f"✗  {e}", indent=1)
        print(
            "\nResolve all issues above before running the official benchmark session.\n"
            "The entire session must run in a single uninterrupted Colab session.\n"
        )
        sys.exit(2)

    log("✓ Pre-flight checks passed — runs/ directory is complete.\n")


# ── Step 2: Record Role A environment ─────────────────────────────────────
def record_env(runs_dir: Path, dry_run: bool) -> dict:
    """
    Capture the exact hardware and software environment of this benchmark
    session and write it to runs/env_info_roleA.json.

    This is the authoritative environment record for the final report —
    all baseline timings and eval numbers come from this machine.
    """
    try:
        import torch
        gpu_model    = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
        cuda_version = torch.version.cuda or "unknown"
        torch_ver    = torch.__version__
        vram_gb      = (
            round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            if torch.cuda.is_available() else None
        )
    except ImportError:
        gpu_model = cuda_version = torch_ver = "torch not installed"
        vram_gb = None

    # Colab session ID (best-effort)
    notebook_runtime = "unknown"
    try:
        result = subprocess.run(
            ["cat", "/proc/1/cgroup"], capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if "docker" in line or "lxc" in line:
                container_id = line.strip().split("/")[-1][:12]
                notebook_runtime = f"colab-{container_id}"
                break
    except Exception:
        pass
    if notebook_runtime == "unknown":
        notebook_runtime = f"local-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    env = {
        "role":                  "A_official_benchmark",
        "recorded_at":           datetime.now(timezone.utc).isoformat(),
        "gpu_model":             gpu_model,
        "vram_gb":               vram_gb,
        "cuda_version":          cuda_version,
        "torch_version":         torch_ver,
        "triton_version":        get_version("triton"),
        "transformers_version":  get_version("transformers"),
        "vllm_version":          get_version("vllm"),
        "python_version":        platform.python_version(),
        "os":                    platform.platform(),
        "notebook_runtime":      notebook_runtime,
        "run_names":             RUN_NAMES,
        "level":                 LEVEL,
        "n_problems":            len(ALL_PIDS),
    }

    log("Captured environment:")
    for k, v in env.items():
        log(f"  {k:25s} = {v}", indent=1)

    if not dry_run:
        out_path = runs_dir / "env_info_roleA.json"
        out_path.write_text(json.dumps(env, indent=2))
        log(f"\n  Written: {out_path}")

    return env


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="CS 639 Stage 0 — official benchmark session runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--runs_dir",           default="./runs")
    parser.add_argument("--kernelbench_root",   default="./KernelBench")
    parser.add_argument("--out_dir",            default="./results")
    parser.add_argument("--num_correct_trials", default=5,  type=int)
    parser.add_argument("--num_perf_trials",    default=50, type=int)
    parser.add_argument("--dry_run",            action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    kb_root  = Path(args.kernelbench_root).expanduser().resolve()
    out_dir  = Path(args.out_dir).expanduser().resolve() if args.out_dir else runs_dir
    dry_run  = args.dry_run

    # Locate this script so benchmark_analysis.py can be called relative to it
    this_dir = Path(__file__).parent.resolve()
    analysis_script = this_dir / "benchmark_analysis.py"

    session_start = datetime.now(timezone.utc)

    banner("CS 639 · Stage 0 · Official Benchmark Session")
    log(f"Started      : {session_start.isoformat()}")
    log(f"runs_dir     : {runs_dir}")
    log(f"kernelbench  : {kb_root}")
    log(f"out_dir      : {out_dir}")
    log(f"dry_run      : {dry_run}")
    if dry_run:
        log("\n⚠  DRY RUN — commands will be printed but not executed.\n")

    # ── Validate KernelBench checkout ──────────────────────────────────────
    eval_script     = kb_root / "scripts" / "eval_from_generations.py"
    baseline_script = kb_root / "scripts" / "generate_baseline_time.py"
    for p, label in [(eval_script, "eval_from_generations.py"),
                     (baseline_script, "generate_baseline_time.py"),
                     (analysis_script, "benchmark_analysis.py")]:
        if not p.exists() and not dry_run:
            sys.exit(f"[ERROR] Cannot find {label} at {p}")

    # ──────────────────────────────────────────────────────────────────────
    banner("Step 1 — Pre-flight checks")
    if not dry_run:
        preflight(runs_dir)
    else:
        log("[DRY RUN] skipping pre-flight\n")

    # ──────────────────────────────────────────────────────────────────────
    banner("Step 2 — Record Role A environment")
    env = record_env(runs_dir, dry_run)

    # ──────────────────────────────────────────────────────────────────────
    banner("Step 3 — Generate baseline timings (torch.compile reference)")
    log("⚠  Do NOT interrupt between here and Step 6 — GPU warm-up must be consistent.\n")
    run_cmd(
        [
            sys.executable, str(baseline_script),
            f"level={LEVEL}",
            "dataset_src=huggingface",
            f"runs_dir={runs_dir}",
        ],
        step="generate_baseline_time",
        dry_run=dry_run,
    )

    # ── Step 3b — Copy baseline timings to runs_dir ────────────────────────
    banner("Step 3b — Copying baseline timings to runs_dir")
    if not dry_run:
        import glob
        
        timing_root = kb_root / "results" / "timing"
        
        pattern = str(timing_root / "**" / "baseline_time_torch.json")
        matches = glob.glob(pattern, recursive=True)
        
        if not matches:
            sys.exit("[ERROR] No baseline_time_torch.json found under KernelBench/results/timing/")
        
        # 按文件修改时间取最新的，而不是按字母顺序
        src = Path(max(matches, key=os.path.getmtime))
        dst = runs_dir / "baseline_times.json"
        
        log(f"  Copying {src} → {dst}")
        import shutil
        shutil.copy2(src, dst)
        log("  ✓ baseline_times.json copied to runs_dir.")
    else:
        log("[DRY RUN] skipping baseline copy\n")

    # ──────────────────────────────────────────────────────────────────────
    banner("Step 4 — Evaluate qwen3_8b_think kernels")
    run_cmd(
        [
            sys.executable, str(eval_script),
            "run_name=qwen3_8b_think",
            f"level={LEVEL}",
            "dataset_src=huggingface",
            "backend=triton",
            f"num_correct_trials={args.num_correct_trials}",
            f"num_perf_trials={args.num_perf_trials}",
            f"runs_dir={runs_dir}",
        ],
        step="eval qwen3_8b_think",
        dry_run=dry_run,
    )

    # ──────────────────────────────────────────────────────────────────────
    banner("Step 5 — Evaluate qwen3_8b_nothink kernels")
    run_cmd(
        [
            sys.executable, str(eval_script),
            "run_name=qwen3_8b_nothink",
            f"level={LEVEL}",
            "dataset_src=huggingface",
            "backend=triton",
            f"num_correct_trials={args.num_correct_trials}",
            f"num_perf_trials={args.num_perf_trials}",
            f"runs_dir={runs_dir}",
        ],
        step="eval qwen3_8b_nothink",
        dry_run=dry_run,
    )

    # ──────────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────
    banner("Step 7 — Benchmark analysis, CSVs, and figures")
    run_cmd(
        [
            sys.executable, str(analysis_script),
            f"--runs_dir={runs_dir}",
            f"--out_dir={out_dir}",
        ],
        step="benchmark_analysis",
        dry_run=dry_run,
    )

    # ──────────────────────────────────────────────────────────────────────
    banner("Step 8 — Final deliverable checklist")

    session_end     = datetime.now(timezone.utc)
    elapsed_minutes = (session_end - session_start).total_seconds() / 60

    all_present = True
    for fname in REQUIRED_OUTPUTS:
        fpath = out_dir / fname
        # env_info_roleA.json lives in runs_dir, not out_dir
        if fname == "env_info_roleA.json":
            fpath = runs_dir / fname
        exists = fpath.exists() or dry_run
        mark   = "✓" if exists else "✗"
        size   = f"  ({fpath.stat().st_size:,} bytes)" if (fpath.exists() and not dry_run) else ""
        log(f"  {mark}  {fname}{size}")
        if not exists:
            all_present = False

    log(f"\n  Session duration : {elapsed_minutes:.1f} min")
    log(f"  Finished at      : {session_end.isoformat()}")

    print()
    if all_present:
        print("✓ Stage 0 official benchmark session complete.")
        print("  All required deliverables produced. Ready to hand off to report author.")
    else:
        print("✗ Some deliverables are missing — check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

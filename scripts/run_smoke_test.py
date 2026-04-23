#!/usr/bin/env python3
"""
CS 639 · Stage 0 — Smoke Test Runner
=====================================
Validates the end-to-end pipeline using a small slice of problems from
Role B and Role C before committing to the full 100-problem run.

What this script does:
  1.  Verify B and C mini-shards contain the expected smoke problems
  2.  Run merge_shards.py --smoke  → smoke_runs/ directory
  3.  Run eval_from_generations.py on the smoke_runs/ for both run names
  4.  Verify eval_results.json was written and is non-empty
  5.  Print a pass/fail verdict and tell you whether to green-light B and C

Usage
-----
    python run_smoke_test.py \\
        --shard_b  <path>  \\        # B's mini-shard (e.g. problems 1-2)
        --shard_c  <path>  \\        # C's mini-shard (e.g. problems 51-52)
        --kernelbench_root <path>   # KernelBench checkout (default: ./KernelBench)
        --smoke_runs_dir   <path>   # where to write the smoke runs/ (default: ./smoke_runs)
        --num_correct_trials <int>  # default: 2  (keep small for speed)
        --num_perf_trials    <int>  # default: 5  (keep small for speed)

The script intentionally uses small trial counts so the eval completes in
minutes rather than hours. The goal is structural validation, not accurate
timing numbers.

Exit codes
----------
    0   Smoke test passed — safe to green-light B and C for full run
    1   Smoke test failed — structural issues found, do NOT green-light yet
    2   Argument / setup error
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────
RUN_NAMES = ["qwen3_8b_think", "qwen3_8b_nothink"]
LEVEL     = 1


# ── Helpers ────────────────────────────────────────────────────────────────
def banner(title: str) -> None:
    bar = "=" * 65
    print(f"\n{bar}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{bar}", flush=True)


def log(msg: str, indent: int = 0) -> None:
    print("  " * indent + msg, flush=True)


def run_cmd(cmd: list[str], step: str) -> bool:
    """
    Run a subprocess, stream output live.
    Returns True on success, False on failure (does NOT sys.exit).
    """
    log(f"$ {' '.join(str(c) for c in cmd)}", indent=1)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        log(f"✗  {step} exited with code {result.returncode}", indent=1)
        return False
    log(f"✓  {step} complete.", indent=1)
    return True


def find_script(this_dir: Path, name: str) -> Path:
    """Look for a sibling script; exit with a clear message if missing."""
    p = this_dir / name
    if not p.exists():
        sys.exit(
            f"[ERROR] Cannot find {name} — expected it next to run_smoke_test.py at {p}\n"
            "        Make sure all Role A scripts are in the same directory."
        )
    return p


# ── Step 1: Validate mini-shards contain smoke problems ───────────────────
def validate_mini_shard(shard_path: Path, role: str, errors: list[str]) -> list[int]:
    """
    Check that the mini-shard has at least one problem and the expected
    B/C problem range (B: 1-50, C: 51-100).
    Returns the list of problem IDs found.
    """
    import csv, re

    expected_range = range(1, 51) if role == "B" else range(51, 101)
    found_pids: set[int] = set()

    if not shard_path.exists():
        errors.append(f"[Role {role}] Shard path does not exist: {shard_path}")
        return []

    for run_name in RUN_NAMES:
        run_dir = shard_path / run_name
        if not run_dir.is_dir():
            errors.append(f"[Role {role}] Missing run directory: {run_dir}")
            continue

        # Try CSV first
        manifest = run_dir / "generation_manifest.csv"
        if manifest.exists():
            with open(manifest, newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        pid = int(row["problem_id"])
                        found_pids.add(pid)
                    except (KeyError, ValueError):
                        pass

        # Fallback: scan kernel files
        if not found_pids:
            for kf in run_dir.glob("level_1_problem_*_sample_0_kernel.py"):
                m = __import__("re").search(r"level_1_problem_(\d+)_sample_0_kernel\.py", kf.name)
                if m:
                    found_pids.add(int(m.group(1)))

    if not found_pids:
        errors.append(
            f"[Role {role}] No problem IDs found in shard. "
            "Check that generation_manifest.csv or kernel.py files are present."
        )
        return []

    # Verify all found pids are in the correct range for this role
    out_of_range = [p for p in found_pids if p not in expected_range]
    if out_of_range:
        errors.append(
            f"[Role {role}] Problem IDs out of expected range "
            f"({'1-50' if role == 'B' else '51-100'}): {sorted(out_of_range)}"
        )

    sorted_pids = sorted(found_pids)
    log(f"  [Role {role}] Found problems: {sorted_pids}")
    return sorted_pids


# ── Step 4: Verify eval_results.json ──────────────────────────────────────
def verify_eval_results(smoke_runs_dir: Path, b_pids: list[int], c_pids: list[int]) -> list[str]:
    """
    Check that eval_results.json was written for both run names and contains
    entries for every smoke problem. Returns a list of error strings.
    """
    errors = []
    all_pids = b_pids + c_pids

    for run_name in RUN_NAMES:
        eval_path = smoke_runs_dir / run_name / "eval_results.json"

        if not eval_path.exists():
            errors.append(f"[{run_name}] eval_results.json not found at {eval_path}")
            continue

        try:
            data = json.loads(eval_path.read_text())
        except json.JSONDecodeError as e:
            errors.append(f"[{run_name}] eval_results.json is malformed JSON: {e}")
            continue

        if not data:
            errors.append(f"[{run_name}] eval_results.json is empty")
            continue

        # Check every smoke pid has an entry
        for pid in all_pids:
            if str(pid) not in data and pid not in data:
                errors.append(f"[{run_name}] eval_results.json missing entry for problem {pid}")
                continue

            entries = data.get(str(pid), data.get(pid, []))
            if not entries:
                errors.append(f"[{run_name}] eval_results.json has empty entry for problem {pid}")
                continue

            entry = next((e for e in entries if e.get("sample_id", 0) == 0), None)
            if entry is None:
                errors.append(f"[{run_name}] eval_results.json missing sample_id=0 for problem {pid}")
                continue

            # Warn (not error) if nothing compiled — structural issue upstream
            if not entry.get("compiled", False):
                log(
                    f"  ⚠  [{run_name}] problem {pid}: kernel did not compile "
                    f"(check kernel.py for syntax errors)",
                    indent=1,
                )

        log(f"  [{run_name}] eval_results.json OK — {len(data)} problem(s) evaluated")

    return errors


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="CS 639 Stage 0 — smoke test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--shard_b",           required=True,
                        help="Path to Role B's mini-shard directory or zip")
    parser.add_argument("--shard_c",           required=True,
                        help="Path to Role C's mini-shard directory or zip")
    parser.add_argument("--kernelbench_root",  default="./KernelBench",
                        help="Path to KernelBench checkout (default: ./KernelBench)")
    parser.add_argument("--smoke_runs_dir",    default="./smoke_runs",
                        help="Output directory for smoke test runs (default: ./smoke_runs)")
    parser.add_argument("--num_correct_trials", default=2, type=int,
                        help="Correctness trials per kernel (default: 2)")
    parser.add_argument("--num_perf_trials",    default=5, type=int,
                        help="Performance trials per kernel (default: 5)")
    args = parser.parse_args()

    this_dir        = Path(__file__).parent.resolve()
    shard_b         = Path(args.shard_b).expanduser().resolve()
    shard_c         = Path(args.shard_c).expanduser().resolve()
    kb_root         = Path(args.kernelbench_root).expanduser().resolve()
    smoke_runs_dir  = Path(args.smoke_runs_dir).expanduser().resolve()
    eval_script     = kb_root / "scripts" / "eval_from_generations.py"
    merge_script    = find_script(this_dir, "merge_shards.py")

    started_at = datetime.now(timezone.utc)

    banner("CS 639 · Stage 0 · Smoke Test")
    log(f"Started        : {started_at.isoformat()}")
    log(f"Shard B        : {shard_b}")
    log(f"Shard C        : {shard_c}")
    log(f"KernelBench    : {kb_root}")
    log(f"smoke_runs_dir : {smoke_runs_dir}")
    log(f"correct_trials : {args.num_correct_trials}")
    log(f"perf_trials    : {args.num_perf_trials}")

    all_errors: list[str] = []
    step_failures: list[str] = []

    # ── Step 1: Validate mini-shards ──────────────────────────────────────
    banner("Step 1 — Validate mini-shards")
    b_pids = validate_mini_shard(shard_b, "B", all_errors)
    c_pids = validate_mini_shard(shard_c, "C", all_errors)

    if all_errors:
        log("\n✗  Mini-shard validation failed:")
        for e in all_errors:
            log(f"   {e}", indent=1)
        log("\nFix these issues before asking B and C to continue.\n")
        sys.exit(1)

    log(f"\n  Combined smoke problems: {b_pids + c_pids}")
    log("  ✓ Mini-shards look structurally valid.\n")

    # ── Step 2: merge_shards.py --smoke ───────────────────────────────────
    banner("Step 2 — Merge mini-shards (--smoke mode)")
    merge_ok = run_cmd(
        [
            "python", str(merge_script),
            "--shard_b",  str(shard_b),
            "--shard_c",  str(shard_c),
            "--runs_dir", str(smoke_runs_dir),
            "--smoke",
        ],
        step="merge_shards --smoke",
    )
    if not merge_ok:
        step_failures.append("merge_shards --smoke")
        log("\n✗  Merge step failed. Check merge_report.txt for details.\n")
        # Don't sys.exit yet — report all failures at the end

    # ── Step 3: eval_from_generations.py for both run names ───────────────
    if merge_ok:
        if not eval_script.exists():
            log(f"\n⚠  eval_from_generations.py not found at {eval_script}")
            log("   Skipping eval step — install KernelBench first.\n")
            step_failures.append("eval (KernelBench not found)")
        else:
            all_pids = b_pids + c_pids
            subset_arg = f"subset=({min(all_pids)},{max(all_pids)})"

            for run_name in RUN_NAMES:
                banner(f"Step 3 — Eval: {run_name}")
                eval_ok = run_cmd(
                    [
                        "uv", "run", "python", str(eval_script),
                        f"run_name={run_name}",
                        f"level={LEVEL}",
                        "dataset_src=huggingface",
                        "backend=cpu",
                        f"num_correct_trials={args.num_correct_trials}",
                        f"num_perf_trials={args.num_perf_trials}",
                        f"runs_dir={smoke_runs_dir}",
                        subset_arg,
                    ],
                    step=f"eval {run_name}",
                )
                if not eval_ok:
                    step_failures.append(f"eval {run_name}")

            # ── Step 4: Verify eval_results.json ──────────────────────────
            if not step_failures:
                banner("Step 4 — Verify eval_results.json")
                eval_errors = verify_eval_results(smoke_runs_dir, b_pids, c_pids)
                all_errors.extend(eval_errors)
                if eval_errors:
                    step_failures.append("eval_results verification")

    # ── Final verdict ──────────────────────────────────────────────────────
    banner("Smoke Test Verdict")

    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds() / 60
    log(f"  Elapsed : {elapsed:.1f} min")

    if step_failures or all_errors:
        log("\n  ✗  SMOKE TEST FAILED\n")

        if step_failures:
            log("  Failed steps:")
            for s in step_failures:
                log(f"    • {s}", indent=1)

        if all_errors:
            log("\n  Errors:")
            for e in all_errors:
                log(f"    • {e}", indent=1)

        log("""
  ──────────────────────────────────────────────────────────────
  DO NOT green-light B and C for the full run yet.
  Fix the issues above, ask B and C to re-send corrected shards,
  and re-run this script until it passes.
  ──────────────────────────────────────────────────────────────
""")
        sys.exit(1)

    else:
        log("""
  ✓  SMOKE TEST PASSED — zero errors.

  ──────────────────────────────────────────────────────────────
  GREEN LIGHT: Tell B and C to proceed with their full runs
               (B: problems 1-50, C: problems 51-100).

  While they generate, use this time to:
    • Finalize and re-test merge_shards.py on the full shards
    • Prepare run_official_benchmark.py for the final session
    • Collect partial uploads from B and C every 10-20 problems
  ──────────────────────────────────────────────────────────────
""")
        sys.exit(0)


if __name__ == "__main__":
    main()

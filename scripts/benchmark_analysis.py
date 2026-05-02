#!/usr/bin/env python3
"""
CS 639 · Stage 0 — Benchmark Analysis Script
=============================================
Reads eval_results.json + baseline_times.csv from the merged runs/ directory,
computes all required metrics, exports CSVs and figures.

Must be run AFTER the official benchmark session has produced:
  runs/qwen3_8b_think/eval_results.json
  runs/qwen3_8b_nothink/eval_results.json
  runs/baseline_times.csv   (or baseline_times.json from KernelBench)

Usage
-----
    python benchmark_analysis.py [--runs_dir <path>] [--out_dir <path>]

Output files (all written to --out_dir, default = same as --runs_dir)
----------------------------------------------------------------------
    eval_results_think.csv       per-problem metrics, think run
    eval_results_nothink.csv     per-problem metrics, nothink run
    stage0_summary.csv           aggregated metrics, both runs
    speedup_distribution.png     figure
    pass_rate_comparison.png     figure
"""

import argparse
import ast
import json
import math
import os
import re
import sys
import csv
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ── Optional imports — give a clear error if missing ──────────────────────
try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")          # headless — safe in Colab and terminal
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError as e:
    sys.exit(
        f"[ERROR] Missing dependency: {e}\n"
        "Install with:  pip install pandas numpy matplotlib"
    )

# ── Constants ──────────────────────────────────────────────────────────────
RUN_NAMES   = ["qwen3_8b_think", "qwen3_8b_nothink", "qwen25_coder_7b"]
LEVEL       = 1
ALL_PIDS    = list(range(1, 101))

# Required columns for per-problem export CSV
PER_PROBLEM_COLS = [
    "problem_id", "run_name", "compiled", "correctness",
    "is_valid", "needs_manual_review",
    "runtime_ms", "baseline_ms", "speedup_vs_baseline",
    "kernel_generated", "extraction_success", "error_message",
]

# Required columns for stage0_summary.csv (from task guide schema)
SUMMARY_COLS = [
    "run_name", "pass_at_1", "fast_at_1",
    "avg_speedup_geomean", "n_valid_kernels",
    "n_hacking_flagged", "n_total_problems",
]

FIGURE_DPI = 150


# ── Triton validity check (mirrors generation notebook) ───────────────────
ALLOWED_FORWARD_TORCH_CALLS = {
    "empty",
    "empty_like",
    "empty_strided",
}


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _collect_import_aliases(tree: ast.AST) -> dict[str, set[str]]:
    aliases = {
        "triton": {"triton"},
        "triton_jit": set(),
        "torch": {"torch"},
        "torch_modules": set(),
        "torch_functions": set(),
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                asname = alias.asname or alias.name.split(".")[0]
                if alias.name == "triton":
                    aliases["triton"].add(asname)
                elif alias.name == "torch":
                    aliases["torch"].add(asname)
                elif alias.name in {"torch.nn", "torch.nn.functional"}:
                    aliases["torch_modules"].add(alias.asname or alias.name.split(".")[-1])

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                asname = alias.asname or alias.name
                if module == "triton" and alias.name == "jit":
                    aliases["triton_jit"].add(asname)
                elif module == "torch":
                    aliases["torch_functions"].add(asname)
                elif module in {"torch.nn", "torch.nn.functional"}:
                    aliases["torch_functions"].add(asname)

    return aliases


def _is_triton_jit_decorator(decorator: ast.AST, aliases: dict[str, set[str]]) -> bool:
    name = _dotted_name(decorator)
    if name in aliases["triton_jit"]:
        return True

    parts = name.split(".")
    return len(parts) >= 2 and parts[-1] == "jit" and parts[0] in aliases["triton"]


def _torch_call_name(call: ast.Call, aliases: dict[str, set[str]]) -> str | None:
    name = _dotted_name(call.func)
    if not name:
        return None

    root = name.split(".")[0]
    if root in aliases["torch"] or root in aliases["torch_modules"]:
        return name
    if root in aliases["torch_functions"]:
        return name
    return None


class _TorchUseVisitor(ast.NodeVisitor):
    def __init__(self, aliases: dict[str, set[str]]) -> None:
        self.aliases = aliases
        self.found = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.aliases["torch"] or node.id in self.aliases["torch_functions"]:
            self.found = True

    def visit_Attribute(self, node: ast.Attribute) -> None:
        root = _dotted_name(node).split(".")[0]
        if root in self.aliases["torch"] or root in self.aliases["torch_modules"]:
            self.found = True
        self.generic_visit(node)


def _contains_torch_use(nodes: list[ast.stmt], aliases: dict[str, set[str]]) -> bool:
    visitor = _TorchUseVisitor(aliases)
    for node in nodes:
        visitor.visit(node)
        if visitor.found:
            return True
    return False


def _find_forward_torch_fallbacks(
    tree: ast.AST,
    aliases: dict[str, set[str]],
) -> list[str]:
    fallback_calls: list[str] = []

    for class_node in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if class_node.name != "ModelNew":
            continue
        for fn in [n for n in class_node.body if isinstance(n, ast.FunctionDef)]:
            if fn.name != "forward":
                continue
            for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
                call_name = _torch_call_name(call, aliases)
                if not call_name:
                    continue
                leaf = call_name.split(".")[-1]
                if leaf not in ALLOWED_FORWARD_TORCH_CALLS:
                    fallback_calls.append(call_name)

    return sorted(set(fallback_calls))


def check_triton_validity(source: str) -> dict:
    notes: list[str] = []
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        return {
            "has_triton_jit": bool(re.search(r"@(?:\w+\.)?jit", source)),
            "torch_calls_in_jit": False,
            "is_valid": False,
            "needs_manual_review": True,
            "notes": f"syntax_error: {exc}",
        }

    aliases = _collect_import_aliases(tree)
    jit_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(_is_triton_jit_decorator(dec, aliases) for dec in node.decorator_list)
    ]

    has_triton_jit = bool(jit_functions)
    torch_in_jit = any(_contains_torch_use(fn.body, aliases) for fn in jit_functions)
    forward_fallbacks = _find_forward_torch_fallbacks(tree, aliases)

    if not has_triton_jit:
        notes.append("missing_triton_jit")
    if torch_in_jit:
        notes.append("torch_used_inside_triton_jit")
    if forward_fallbacks:
        notes.append("possible_forward_torch_fallback=" + ",".join(forward_fallbacks[:8]))

    return {
        "has_triton_jit":      has_triton_jit,
        "torch_calls_in_jit":  torch_in_jit,
        "is_valid":            has_triton_jit and not torch_in_jit and not forward_fallbacks,
        "needs_manual_review": torch_in_jit or bool(forward_fallbacks),
        "notes":               "; ".join(notes),
    }


# ── Metric helpers ─────────────────────────────────────────────────────────
def geometric_mean(values: list[float]) -> float:
    """Geometric mean of a list of positive floats. Returns 0.0 if empty."""
    vals = [v for v in values if v and v > 0]
    if not vals:
        return 0.0
    log_sum = sum(math.log(v) for v in vals)
    return math.exp(log_sum / len(vals))


def fast_at_p(rows: list[dict], threshold: float) -> float:
    """
    Fraction of problems where the kernel is correct AND achieves
    speedup > threshold vs baseline.  fast@1 uses threshold=1.0.
    """
    if not rows:
        return 0.0
    n = len(rows)
    count = sum(
        1 for r in rows
        if r["correctness"]
        and r["speedup_vs_baseline"] is not None
        and r["speedup_vs_baseline"] > threshold
    )
    return count / n


def pass_at_1(rows: list[dict]) -> float:
    """Fraction of problems where the kernel compiled AND is correct."""
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["correctness"]) / len(rows)


# ── Data loaders ───────────────────────────────────────────────────────────
def load_baseline(runs_dir: Path) -> dict[int, float]:
    """
    Load baseline timings.  Tries, in order:
      1. runs_dir/baseline_times.csv  (preferred — matches task guide schema)
      2. runs_dir/baseline_times.json (KernelBench native format)
    Returns dict: problem_id (int) → baseline_ms (float)
    """
    # ── CSV format ──────────────────────────────────────────────────────
    csv_path = runs_dir / "baseline_times.csv"
    if csv_path.exists():
        baselines: dict[int, float] = {}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                # Accept both problem_id-keyed and problem_name-keyed rows
                pid_raw = row.get("problem_id") or row.get("pid")
                ms_raw  = row.get("baseline_ms") or row.get("mean_ms") or row.get("time_ms")
                if pid_raw and ms_raw:
                    try:
                        baselines[int(pid_raw)] = float(ms_raw)
                    except ValueError:
                        pass
        if baselines:
            print(f"  Loaded {len(baselines)} baseline timings from {csv_path.name}")
            return baselines

    # ── JSON format (KernelBench generate_baseline_time.py output) ──────
    json_path = runs_dir / "baseline_times.json"
    if json_path.exists():
        with open(json_path) as f:
            raw = json.load(f)
        baselines = {}
        # Format: {"level1": {"problem_name": {"mean": ms, ...}, ...}}
        level_key = f"level{LEVEL}"
        level_data = raw.get(level_key, raw)  # fall back if flat
        for key, val in level_data.items():
            # key may be "problem_name" or stringified pid
            ms = None
            if isinstance(val, dict):
                ms = val.get("mean") or val.get("mean_ms") or val.get("time_ms")
            elif isinstance(val, (int, float)):
                ms = float(val)
            # Extract numeric pid from problem name like "1_ReLU" or "problem_1"
            pid_match = re.search(r"\d+", str(key))
            if pid_match and ms is not None:
                pid = int(pid_match.group())
                if 1 <= pid <= 100:
                    baselines[pid] = float(ms)
        if baselines:
            print(f"  Loaded {len(baselines)} baseline timings from {json_path.name}")
            return baselines

    print(
        "  WARNING: No baseline_times.csv or baseline_times.json found in runs_dir.\n"
        "           Speedup metrics will be NaN. Run generate_baseline_time.py first."
    )
    return {}


def load_manifest(runs_dir: Path, run_name: str) -> dict[int, dict]:
    """
    Load generation_manifest.csv for a run.
    Returns dict: pid → row dict (or empty dict if file missing).
    """
    path = runs_dir / run_name / "generation_manifest.csv"
    result: dict[int, dict] = {}
    if not path.exists():
        return result
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                pid = int(row["problem_id"])
                result[pid] = row
            except (KeyError, ValueError):
                pass
    return result


def load_eval_results(runs_dir: Path, run_name: str) -> dict[int, dict]:
    """
    Load eval_results.json written by eval_from_generations.py.
    Format: {"<pid>": [{"sample_id": 0, "compiled": bool, "correctness": bool,
                         "runtime": float_ms, ...}, ...]}
    Returns dict: pid → entry dict for sample_id=0.
    """
    path = runs_dir / run_name / "eval_results.json"
    if not path.exists():
        print(f"  WARNING: {path} not found — run eval_from_generations.py first.")
        return {}
    with open(path) as f:
        raw = json.load(f)
    result: dict[int, dict] = {}
    for pid_str, entries in raw.items():
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        entry = next((e for e in entries if e.get("sample_id", 0) == 0), None)
        if entry:
            result[pid] = entry
    return result


def build_per_problem_rows(
    run_name: str,
    runs_dir: Path,
    baselines: dict[int, float],
) -> list[dict]:
    """
    Join eval_results + manifest + validity check into one row per problem.
    """
    eval_data = load_eval_results(runs_dir, run_name)
    manifest  = load_manifest(runs_dir, run_name)

    rows = []
    for pid in ALL_PIDS:
        eval_entry = eval_data.get(pid, {})
        mrow       = manifest.get(pid, {})

        # Kernel file path
        kernel_path = runs_dir / run_name / f"level_1_problem_{pid}_sample_0_kernel.py"
        kernel_generated = kernel_path.exists()

        # Validity check
        if kernel_generated:
            validity = check_triton_validity(kernel_path.read_text(encoding="utf-8"))
        else:
            validity = {"is_valid": False, "needs_manual_review": False}

        compiled    = bool(eval_entry.get("compiled", False))
        correctness = bool(eval_entry.get("correctness", False))
        runtime_ms  = eval_entry.get("runtime")   # may be None
        baseline_ms = baselines.get(pid)           # may be None

        if runtime_ms and baseline_ms and float(runtime_ms) > 0:
            speedup = float(baseline_ms) / float(runtime_ms)
        else:
            speedup = None

        rows.append({
            "problem_id":          pid,
            "run_name":            run_name,
            "compiled":            compiled,
            "correctness":         correctness,
            "is_valid":            validity["is_valid"],
            "needs_manual_review": validity["needs_manual_review"],
            "runtime_ms":          runtime_ms,
            "baseline_ms":         baseline_ms,
            "speedup_vs_baseline": speedup,
            "kernel_generated":    kernel_generated,
            "extraction_success":  mrow.get("extraction_success", ""),
            "error_message":       mrow.get("error_message", ""),
        })

    return rows


def build_summary_row(run_name: str, rows: list[dict]) -> dict:
    """Compute stage0_summary.csv metrics for one run."""
    n_total    = len(rows)
    n_valid    = sum(1 for r in rows if r["is_valid"])
    n_hacking  = sum(1 for r in rows if r["needs_manual_review"])

    p1   = pass_at_1(rows)
    f1   = fast_at_p(rows, threshold=1.0)

    correct_speedups = [
        r["speedup_vs_baseline"]
        for r in rows
        if r["correctness"] and r["speedup_vs_baseline"] is not None
    ]
    geo_mean = geometric_mean(correct_speedups)

    return {
        "run_name":            run_name,
        "pass_at_1":           round(p1,       4),
        "fast_at_1":           round(f1,       4),
        "avg_speedup_geomean": round(geo_mean, 4),
        "n_valid_kernels":     n_valid,
        "n_hacking_flagged":   n_hacking,
        "n_total_problems":    n_total,
    }


# ── CSV writers ────────────────────────────────────────────────────────────
def write_per_problem_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_PROBLEM_COLS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in PER_PROBLEM_COLS})
    print(f"  Written: {path.name}  ({len(rows)} rows)")


def write_summary_csv(summary_rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        w.writeheader()
        for row in summary_rows:
            w.writerow({c: row.get(c, "") for c in SUMMARY_COLS})
    print(f"  Written: {path.name}  ({len(summary_rows)} rows)")


# ── Figures ────────────────────────────────────────────────────────────────
COLORS = {
    "qwen3_8b_think":    "#4C72B0",
    "qwen3_8b_nothink":  "#DD8452",
    "qwen25_coder_7b":   "#55A868",
}
LABELS = {
    "qwen3_8b_think":    "Qwen3-8B think",
    "qwen3_8b_nothink":  "Qwen3-8B no-think",
    "qwen25_coder_7b":   "Qwen2.5-Coder-7B",
}


def plot_speedup_distribution(
    all_rows: dict[str, list[dict]],
    out_path: Path,
) -> None:
    """
    Speedup distribution — side-by-side violin + strip plot, log2 y-axis.
    Only correct kernels with a valid speedup are included.
    Matches the task guide required filename: speedup_distribution.png
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    positions   = []
    plot_data   = []
    tick_labels = []

    for i, run_name in enumerate(RUN_NAMES):
        rows = all_rows[run_name]
        speedups = [
            math.log2(r["speedup_vs_baseline"])
            for r in rows
            if r["correctness"]
            and r["speedup_vs_baseline"] is not None
            and r["speedup_vs_baseline"] > 0
        ]
        positions.append(i + 1)
        plot_data.append(speedups)
        tick_labels.append(LABELS[run_name])

    non_empty = [
        (pos, speedups, run_name)
        for pos, speedups, run_name in zip(positions, plot_data, RUN_NAMES)
        if speedups
    ]

    if non_empty:
        # Violin
        parts = ax.violinplot(
            [speedups for _, speedups, _ in non_empty],
            positions=[pos for pos, _, _ in non_empty],
            showmedians=True,
            showextrema=True,
        )
        for pc, (_, _, run_name) in zip(parts["bodies"], non_empty):
            pc.set_facecolor(COLORS[run_name])
            pc.set_alpha(0.6)
        for part_name in ("cmedians", "cbars", "cmins", "cmaxes"):
            parts[part_name].set_color("black")
            parts[part_name].set_linewidth(1.2)
    else:
        ax.text(
            0.5, 0.55,
            "No correct kernels with baseline speedup data",
            ha="center", va="center", transform=ax.transAxes, fontsize=11,
        )
        ax.text(
            0.5, 0.45,
            "Run generate_baseline_time.py and eval_from_generations.py first.",
            ha="center", va="center", transform=ax.transAxes, fontsize=9,
        )

    # Individual points (jittered)
    rng = np.random.default_rng(42)
    for i, (speedups, run_name) in enumerate(zip(plot_data, RUN_NAMES)):
        jitter = rng.uniform(-0.08, 0.08, size=len(speedups))
        ax.scatter(
            np.array(positions[i]) + jitter, speedups,
            color=COLORS[run_name], alpha=0.5, s=18, zorder=3,
        )

    # Reference line at 1× (log2 = 0)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.4,
               label="1× baseline (torch.compile)")

    # Y-axis: log2 ticks → human-readable labels
    y_vals = sorted({v for speeds in plot_data for v in speeds})
    if y_vals:
        y_min = math.floor(min(y_vals)) - 1
        y_max = math.ceil(max(y_vals))  + 1
    else:
        y_min, y_max = -2, 3
    tick_range = range(max(-6, y_min), min(6, y_max) + 1)
    ax.set_yticks(list(tick_range))
    ax.set_yticklabels([f"{2**t:.2f}×" for t in tick_range])
    ax.set_ylim(y_min - 0.3, y_max + 0.3)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel("Speedup vs torch.compile  (log₂ scale)", fontsize=11)
    ax.set_title(
        "Speedup Distribution — Correct Kernels Only\n"
        "(Triton backend, Level 1, A100 40 GB)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"  Written: {out_path.name}")


def plot_pass_rate_comparison(
    summary_rows: list[dict],
    out_path: Path,
) -> None:
    """
    Grouped bar chart comparing pass@1, fast@1, valid% across both runs.
    Matches the task guide required filename: pass_rate_comparison.png
    """
    metrics      = ["pass_at_1", "fast_at_1", "valid_pct"]
    metric_labels = ["pass@1\n(correctness)", "fast@1\n(correct + faster)", "valid%\n(Triton kernel)"]

    fig, ax = plt.subplots(figsize=(8, 5))

    n_runs    = len(summary_rows)
    bar_width = 0.28
    x         = np.arange(len(metrics))

    for i, row in enumerate(summary_rows):
        run_name = row["run_name"]
        # valid_pct is not in summary schema — derive from n_valid / n_total
        valid_pct = (
            row["n_valid_kernels"] / row["n_total_problems"]
            if row["n_total_problems"] > 0 else 0.0
        )
        vals = [
            float(row["pass_at_1"]),
            float(row["fast_at_1"]),
            valid_pct,
        ]
        offset = (i - n_runs / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, [v * 100 for v in vals],
            bar_width,
            label=LABELS.get(run_name, run_name),
            color=COLORS.get(run_name, "#888888"),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, v in zip(bars, vals):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 1.0,
                f"{v*100:.1f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title(
        "Pass Rate Comparison — Qwen3-8B Think / No-Think / Qwen2.5-Coder-7B\n"
        "(Triton backend, Level 1, A100 40 GB)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"  Written: {out_path.name}")


# ── Console summary table ──────────────────────────────────────────────────
def print_summary_table(summary_rows: list[dict]) -> None:
    sep = "=" * 75
    print(f"\n{sep}")
    print("  STAGE 0 BENCHMARK SUMMARY")
    print(f"  Triton backend · Level 1 · torch.compile baseline · A100 40 GB")
    print(sep)
    hdr = (
        f"  {'Run':<28} {'pass@1':>8} {'fast@1':>8} "
        f"{'GeoMean':>9} {'Valid':>7} {'Hacking':>9} {'Total':>7}"
    )
    print(hdr)
    print("  " + "-" * 71)
    for row in summary_rows:
        print(
            f"  {row['run_name']:<28} "
            f"{float(row['pass_at_1'])*100:>7.1f}% "
            f"{float(row['fast_at_1'])*100:>7.1f}% "
            f"{float(row['avg_speedup_geomean']):>8.3f}× "
            f"{row['n_valid_kernels']:>7} "
            f"{row['n_hacking_flagged']:>9} "
            f"{row['n_total_problems']:>7}"
        )
    print(sep + "\n")


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="CS 639 Stage 0 — benchmark analysis and figure export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--runs_dir", default="./runs",
        help="Path to the merged runs/ directory (default: ./runs)",
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Where to write output files (default: same as --runs_dir)",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    out_dir  = Path(args.out_dir).expanduser().resolve() if args.out_dir else runs_dir

    if not runs_dir.exists():
        sys.exit(f"[ERROR] runs_dir does not exist: {runs_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("CS 639 · Stage 0 · Benchmark Analysis")
    print("=" * 60)
    print(f"  runs_dir : {runs_dir}")
    print(f"  out_dir  : {out_dir}\n")

    # ── 1. Load baselines ──────────────────────────────────────────────────
    print("Step 1 — Loading baseline timings ...")
    baselines = load_baseline(runs_dir)
    print(f"  {len(baselines)} problems have baseline timings.")

    # Always write baseline_times.csv (required deliverable per task guide)
    baseline_csv = out_dir / "baseline_times.csv"
    with open(baseline_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["problem_id", "baseline_ms"])
        w.writeheader()
        for pid in sorted(baselines):
            w.writerow({"problem_id": pid, "baseline_ms": baselines[pid]})
    print(f"  Written: baseline_times.csv  ({len(baselines)} rows)\n")

    # ── 2. Build per-problem rows for each run ─────────────────────────────
    print("Step 2 — Building per-problem result rows ...")
    all_rows:     dict[str, list[dict]] = {}
    summary_rows: list[dict]            = []

    for run_name in RUN_NAMES:
        print(f"  [{run_name}]")
        rows = build_per_problem_rows(run_name, runs_dir, baselines)
        all_rows[run_name] = rows

        n_compiled   = sum(1 for r in rows if r["compiled"])
        n_correct    = sum(1 for r in rows if r["correctness"])
        n_valid      = sum(1 for r in rows if r["is_valid"])
        n_with_speed = sum(1 for r in rows if r["speedup_vs_baseline"] is not None)
        print(f"    compiled={n_compiled}/100  correct={n_correct}/100  "
              f"valid={n_valid}/100  speedup_pairs={n_with_speed}/100")

    print()

    # ── 3. Export per-problem CSVs ─────────────────────────────────────────
    print("Step 3 — Writing per-problem CSVs ...")
    csv_name_map = {
        "qwen3_8b_think":   "eval_results_think.csv",
        "qwen3_8b_nothink": "eval_results_nothink.csv",
        "qwen25_coder_7b":  "eval_results_coder.csv",
    }
    for run_name in RUN_NAMES:
        write_per_problem_csv(
            all_rows[run_name],
            out_dir / csv_name_map[run_name],
        )
    print()

    # ── 4. Compute and export summary ─────────────────────────────────────
    print("Step 4 — Computing summary metrics ...")
    for run_name in RUN_NAMES:
        summary_rows.append(build_summary_row(run_name, all_rows[run_name]))

    write_summary_csv(summary_rows, out_dir / "stage0_summary.csv")
    print_summary_table(summary_rows)

    # ── 5. Figures ─────────────────────────────────────────────────────────
    print("Step 5 — Generating figures ...")
    plot_speedup_distribution(all_rows, out_dir / "speedup_distribution.png")
    plot_pass_rate_comparison(summary_rows, out_dir / "pass_rate_comparison.png")
    print()

    # ── 6. Final checklist ─────────────────────────────────────────────────
    required_outputs = [
        "baseline_times.csv",
        "eval_results_think.csv",
        "eval_results_nothink.csv",
        "eval_results_coder.csv",
        "stage0_summary.csv",
        "speedup_distribution.png",
        "pass_rate_comparison.png",
    ]
    print("Step 6 — Final output checklist ...")
    all_present = True
    for fname in required_outputs:
        fpath = out_dir / fname
        exists = fpath.exists()
        mark   = "✓" if exists else "✗"
        size   = f"  ({fpath.stat().st_size:,} bytes)" if exists else ""
        print(f"  {mark}  {fname}{size}")
        if not exists:
            all_present = False

    print()
    if all_present:
        print("✓ All required output files produced. Stage 0 analysis complete.")
    else:
        print("✗ Some output files are missing — check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

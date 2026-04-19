#!/usr/bin/env python3
"""
CS 639 · Stage 0 — Role A Merge Script
=======================================
Merges Role B's shard (problems 1–50) and Role C's shard (problems 51–100)
into a single canonical runs/ directory tree.

Usage
-----
    # Full merge (problems 1-50 from B, 51-100 from C)
    python merge_shards.py --shard_b <path> --shard_c <path> [--runs_dir <path>]

    # Smoke-test merge (only validates problems actually present in the shards)
    python merge_shards.py --shard_b <path> --shard_c <path> --smoke [--runs_dir <path>]

Arguments
---------
    --shard_b   Path to Role B's shard directory or zip file
    --shard_c   Path to Role C's shard directory or zip file
    --runs_dir  Output runs/ directory (default: ./runs)
    --dry_run   Validate only -- do not copy any files
    --smoke     Smoke-test mode: infer expected problem IDs from the CSVs
                actually present in each shard instead of requiring the full
                1-50 / 51-100 ranges. Use this when B and C have only
                completed a few problems (e.g. 1-2 and 51-52).

Exit codes
----------
    0   Merge complete, zero errors
    1   Validation failed -- missing or malformed files (see merge_report.txt)
    2   Argument / filesystem error
"""

import argparse
import csv
import io
import json
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


# ── Constants (must match the generation notebook exactly) ─────────────────
LEVELS     = [1]
RUN_NAMES  = ["qwen3_8b_think", "qwen3_8b_nothink"]
PROBLEMS_B = range(1, 51)
PROBLEMS_C = range(51, 101)
ALL_PROBLEMS = list(PROBLEMS_B) + list(PROBLEMS_C)

REQUIRED_PER_PAIR = {
    # logical_name: filename_template  (use {pid})
    "raw.txt":    "level_1_problem_{pid}_raw.txt",
    "clean.txt":  "level_1_problem_{pid}_clean.txt",
    "kernel.py":  "level_1_problem_{pid}_sample_0_kernel.py",
    "meta.json":  "level_1_problem_{pid}_meta.json",
}

MANIFEST_COLS = [
    "problem_id", "run_name", "status", "raw_path", "clean_path",
    "kernel_path", "meta_path", "extraction_success", "output_length",
    "error_message",
]
VALIDITY_COLS = [
    "problem_id", "run_name", "has_triton_jit", "torch_calls_in_jit",
    "is_valid", "needs_manual_review", "notes",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def log(msg: str, indent: int = 0) -> None:
    print("  " * indent + msg)


def extract_zip_to_tmp(zip_path: Path) -> Path:
    """Unzip a shard zip into a sibling temp directory and return its path."""
    tmp_dir = zip_path.parent / (zip_path.stem + "_extracted")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)
    log(f"  Extracted {zip_path.name} → {tmp_dir}")
    return tmp_dir


def resolve_shard_root(path_str: str, role: str) -> Path:
    """
    Resolve a shard argument to a directory that contains runs/<run_name>/.
    Accepts either a directory or a .zip file.
    Returns the directory that is the *parent* of the runs/ subtree
    (i.e., the directory where `runs/qwen3_8b_think/` lives).
    """
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        sys.exit(f"[ERROR] Role {role} shard path does not exist: {p}")

    if p.is_file():
        if p.suffix != ".zip":
            sys.exit(f"[ERROR] Role {role} shard file is not a .zip: {p}")
        p = extract_zip_to_tmp(p)

    # Accept the shard root itself OR a parent that contains it
    # The shard root is wherever runs/<run_name>/ directories live.
    # Try p directly, then p/runs.
    for candidate in [p, p / "runs"]:
        if any((candidate / rn).is_dir() for rn in RUN_NAMES):
            return candidate

    # Last resort: search one level deep
    for child in sorted(p.iterdir()):
        if child.is_dir():
            if any((child / rn).is_dir() for rn in RUN_NAMES):
                return child

    sys.exit(
        f"[ERROR] Could not find run directories ({RUN_NAMES}) "
        f"anywhere under Role {role} shard: {p}"
    )


def read_csv_rows(path: Path) -> list[dict]:
    """Read a CSV into a list of dicts, or return [] if the file is missing."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, cols: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})



# ── Smoke-test problem range inference ────────────────────────────────────

def infer_smoke_problems(shard_root: Path, role: str) -> list[int]:
    """
    In smoke-test mode, infer which problem IDs to validate from the
    generation_manifest.csv files actually present in the shard.
    Falls back to scanning for kernel.py files if CSVs are absent.
    Returns a sorted list of integer problem IDs.
    """
    pids: set[int] = set()

    for run_name in RUN_NAMES:
        run_dir = shard_root / run_name
        if not run_dir.is_dir():
            continue

        # Try reading from generation_manifest.csv first
        manifest_path = run_dir / "generation_manifest.csv"
        if manifest_path.exists():
            with open(manifest_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        pids.add(int(row["problem_id"]))
                    except (KeyError, ValueError):
                        pass
            if pids:
                continue  # found via CSV, skip file scan for this run

        # Fallback: scan for kernel files
        for kfile in run_dir.glob("level_1_problem_*_sample_0_kernel.py"):
            m = re.search(r"level_1_problem_(\d+)_sample_0_kernel\.py", kfile.name)
            if m:
                pids.add(int(m.group(1)))

    if not pids:
        sys.exit(
            f"[ERROR] --smoke: could not infer any problem IDs from Role {role} shard at {shard_root}.\n"
            "        Make sure generation_manifest.csv or kernel.py files are present."
        )

    sorted_pids = sorted(pids)
    log(f"  [Role {role}] smoke problems inferred: {sorted_pids}")
    return sorted_pids


# ── Core validation ────────────────────────────────────────────────────────

def validate_shard(
    shard_root: Path,
    role: str,
    problems: range,
    errors: list[str],
    warnings: list[str],
) -> dict:
    """
    Validate one shard. Returns a dict of
        (run_name, pid) → {logical_name: absolute_path}
    for all files that *do* exist (even if some are missing).
    """
    found: dict[tuple, dict] = {}

    for run_name in RUN_NAMES:
        run_dir = shard_root / run_name
        if not run_dir.is_dir():
            errors.append(f"[Role {role}] Missing run directory: {run_dir}")
            continue

        # env_info.json
        env_path = run_dir / "env_info.json"
        if not env_path.exists():
            warnings.append(f"[Role {role}] Missing env_info.json in {run_dir}")
        else:
            try:
                env = json.loads(env_path.read_text())
                required_env_keys = [
                    "gpu_model", "cuda_version", "torch_version",
                    "triton_version", "transformers_version",
                    "vllm_version", "notebook_runtime",
                ]
                for k in required_env_keys:
                    if k not in env:
                        warnings.append(
                            f"[Role {role}] env_info.json missing key '{k}' in {run_dir}"
                        )
            except json.JSONDecodeError as e:
                errors.append(f"[Role {role}] env_info.json is malformed JSON: {e}")

        for pid in problems:
            pair_files: dict[str, Path] = {}
            for logical, template in REQUIRED_PER_PAIR.items():
                fpath = run_dir / template.format(pid=pid)
                if not fpath.exists():
                    errors.append(
                        f"[Role {role}] MISSING  ({run_name}, problem {pid:03d}): {logical}"
                    )
                elif fpath.stat().st_size == 0:
                    errors.append(
                        f"[Role {role}] EMPTY    ({run_name}, problem {pid:03d}): {logical}"
                    )
                else:
                    pair_files[logical] = fpath

            # meta.json schema check
            if "meta.json" in pair_files:
                try:
                    meta = json.loads(pair_files["meta.json"].read_text())
                    for k in ("problem_id", "run_name", "level"):
                        if k not in meta:
                            warnings.append(
                                f"[Role {role}] meta.json missing key '{k}' "
                                f"({run_name}, problem {pid})"
                            )
                    if meta.get("problem_id") != pid:
                        errors.append(
                            f"[Role {role}] meta.json problem_id mismatch "
                            f"(expected {pid}, got {meta.get('problem_id')}) "
                            f"({run_name}, problem {pid})"
                        )
                    if meta.get("run_name") != run_name:
                        errors.append(
                            f"[Role {role}] meta.json run_name mismatch "
                            f"(expected {run_name!r}, got {meta.get('run_name')!r}) "
                            f"({run_name}, problem {pid})"
                        )
                except json.JSONDecodeError as e:
                    errors.append(
                        f"[Role {role}] meta.json malformed JSON "
                        f"({run_name}, problem {pid}): {e}"
                    )

            found[(run_name, pid)] = pair_files

    return found


def validate_csv(
    shard_root: Path,
    role: str,
    problems: range,
    errors: list[str],
    warnings: list[str],
) -> dict[str, list[dict]]:
    """Validate manifest and validity CSVs; return their rows keyed by csv name."""
    result = {}
    expected_pids = set(problems)

    for run_name in RUN_NAMES:
        run_dir = shard_root / run_name

        for csv_name, cols in [
            ("generation_manifest.csv", MANIFEST_COLS),
            ("validity_flags.csv",      VALIDITY_COLS),
        ]:
            csv_path = run_dir / csv_name
            key = f"{run_name}/{csv_name}"

            if not csv_path.exists():
                errors.append(f"[Role {role}] MISSING CSV: {csv_path}")
                result[key] = []
                continue

            rows = read_csv_rows(csv_path)

            # Check required columns
            if rows:
                missing_cols = [c for c in cols if c not in rows[0]]
                if missing_cols:
                    errors.append(
                        f"[Role {role}] {csv_name} missing columns: {missing_cols}"
                    )

            # Check problem_id coverage
            csv_pids = set()
            for row in rows:
                try:
                    csv_pids.add(int(row["problem_id"]))
                except (KeyError, ValueError):
                    warnings.append(
                        f"[Role {role}] {csv_name} has row with invalid problem_id: {row}"
                    )

            for pid in expected_pids - csv_pids:
                errors.append(
                    f"[Role {role}] {csv_name} missing row for problem {pid} "
                    f"(run_name={run_name})"
                )

            result[key] = rows

    return result


# ── Merge helpers ──────────────────────────────────────────────────────────

def copy_run_files(
    shard_root: Path,
    role: str,
    problems: range,
    out_runs: Path,
    dry_run: bool,
) -> None:
    """Copy all kernel/raw/clean/meta files from shard into the output runs/ tree."""
    for run_name in RUN_NAMES:
        src_run = shard_root / run_name
        dst_run = out_runs / run_name
        if not dry_run:
            dst_run.mkdir(parents=True, exist_ok=True)

        for pid in problems:
            for logical, template in REQUIRED_PER_PAIR.items():
                fname = template.format(pid=pid)
                src = src_run / fname
                dst = dst_run / fname
                if src.exists() and not dry_run:
                    shutil.copy2(src, dst)

        # Copy env_info.json (prefer to keep both; Role A's version wins on conflict)
        env_src = src_run / "env_info.json"
        env_dst = dst_run / f"env_info_role{role}.json"
        if env_src.exists() and not dry_run:
            shutil.copy2(env_src, env_dst)


def merge_csvs(
    b_root: Path,
    c_root: Path,
    out_runs: Path,
    errors: list[str],
    dry_run: bool,
) -> dict[str, list[dict]]:
    """
    Concatenate generation_manifest.csv and validity_flags.csv from B and C,
    de-duplicate headers, sort by (run_name, problem_id), and write merged files.
    Returns merged rows keyed by csv_name for the report.
    """
    merged: dict[str, list[dict]] = {}

    for csv_name, cols in [
        ("generation_manifest.csv", MANIFEST_COLS),
        ("validity_flags.csv",      VALIDITY_COLS),
    ]:
        all_rows: list[dict] = []

        for role, shard_root in [("B", b_root), ("C", c_root)]:
            for run_name in RUN_NAMES:
                path = shard_root / run_name / csv_name
                rows = read_csv_rows(path)
                all_rows.extend(rows)

        # De-duplicate on (problem_id, run_name) — last write wins
        seen: dict[tuple, dict] = {}
        for row in all_rows:
            try:
                key = (int(row["problem_id"]), row["run_name"])
                seen[key] = row
            except (KeyError, ValueError):
                errors.append(f"Skipping malformed row in {csv_name}: {row}")

        # Sort by run_name then problem_id
        sorted_rows = [
            v for _, v in sorted(seen.items(), key=lambda x: (x[0][1], x[0][0]))
        ]
        merged[csv_name] = sorted_rows

        if not dry_run:
            write_csv(out_runs / csv_name, cols, sorted_rows)

    return merged


def merge_env_info(
    b_root: Path,
    c_root: Path,
    out_runs: Path,
    dry_run: bool,
) -> None:
    """
    Write a combined env_info_merged.json at the runs/ root that records
    both B and C's environment metadata side-by-side.
    """
    combined: dict[str, dict] = {}
    for role, shard_root in [("B", b_root), ("C", c_root)]:
        for run_name in RUN_NAMES:
            env_path = shard_root / run_name / "env_info.json"
            if env_path.exists():
                try:
                    combined[f"role_{role}_{run_name}"] = json.loads(env_path.read_text())
                except json.JSONDecodeError:
                    pass

    if not dry_run:
        out_path = out_runs / "env_info_merged.json"
        out_path.write_text(json.dumps(combined, indent=2))


# ── Report writer ──────────────────────────────────────────────────────────

def write_report(
    out_runs: Path,
    errors: list[str],
    warnings: list[str],
    merged_csvs: dict[str, list[dict]],
    b_root: Path,
    c_root: Path,
    dry_run: bool,
    smoke: bool = False,
    smoke_b_pids: list[int] | None = None,
    smoke_c_pids: list[int] | None = None,
) -> bool:
    """Write merge_report.txt. Returns True if zero errors."""
    now = datetime.now(timezone.utc).isoformat()
    mode_label = "SMOKE TEST" if smoke else "FULL MERGE"

    lines = [
        f"CS 639 \u00b7 Stage 0 \u00b7 Merge Report ({mode_label})",
        "=" * 60,
        f"Generated : {now}",
        f"Mode      : {mode_label}",
        f"Dry run   : {dry_run}",
        f"Shard B   : {b_root}",
        f"Shard C   : {c_root}",
        f"Output    : {out_runs}",
        "",
    ]

    if smoke:
        b_pids = smoke_b_pids or []
        c_pids = smoke_c_pids or []
        lines += [
            f"  Smoke B problems : {b_pids}",
            f"  Smoke C problems : {c_pids}",
            "",
        ]

    # Summary counts
    manifest_rows = merged_csvs.get("generation_manifest.csv", [])
    validity_rows = merged_csvs.get("validity_flags.csv", [])
    if smoke:
        all_smoke_pids = (smoke_b_pids or []) + (smoke_c_pids or [])
        expected_pairs = len(RUN_NAMES) * len(all_smoke_pids)
    else:
        expected_pairs = len(RUN_NAMES) * len(ALL_PROBLEMS)  # 200

    lines += [
        "── Summary ─────────────────────────────────────────────────",
        f"  Expected (problem, run_name) pairs : {expected_pairs}",
        f"  generation_manifest.csv rows       : {len(manifest_rows)}",
        f"  validity_flags.csv rows            : {len(validity_rows)}",
        f"  Errors                             : {len(errors)}",
        f"  Warnings                           : {len(warnings)}",
        "",
    ]

    if errors:
        lines.append("── ERRORS (must fix before benchmarking) ────────────────────")
        for e in errors:
            lines.append(f"  {e}")
        lines.append("")

    if warnings:
        lines.append("── Warnings (investigate but non-blocking) ──────────────────")
        for w in warnings:
            lines.append(f"  {w}")
        lines.append("")

    # Per-run validity summary
    if validity_rows:
        lines.append("── Validity Summary ──────────────────────────────────────────")
        for run_name in RUN_NAMES:
            run_rows = [r for r in validity_rows if r.get("run_name") == run_name]
            n_valid  = sum(1 for r in run_rows if str(r.get("is_valid", "")).lower() == "true")
            n_review = sum(1 for r in run_rows if str(r.get("needs_manual_review", "")).lower() == "true")
            lines.append(
                f"  {run_name:30s}  valid={n_valid}/{len(run_rows)}"
                f"  needs_review={n_review}"
            )
        lines.append("")

    if not errors:
        if smoke:
            lines.append("✓ SMOKE TEST PASSED — zero errors. Give B and C the green light for full run.")
        else:
            lines.append("✓ MERGE OK — zero errors. Ready for official benchmark session.")
    else:
        if smoke:
            lines.append("✗ SMOKE TEST FAILED — fix structural issues before B and C run all 100 problems.")
        else:
            lines.append("✗ MERGE FAILED — resolve all errors above before proceeding.")

    report_text = "\n".join(lines) + "\n"

    print(report_text)

    if not dry_run:
        (out_runs / "merge_report.txt").write_text(report_text, encoding="utf-8")

    return len(errors) == 0


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CS 639 Stage 0 — merge Role B and C shards into runs/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--shard_b",  required=True,
                        help="Path to Role B's shard directory or zip file")
    parser.add_argument("--shard_c",  required=True,
                        help="Path to Role C's shard directory or zip file")
    parser.add_argument("--runs_dir", default="./runs",
                        help="Output runs/ directory (default: ./runs)")
    parser.add_argument("--dry_run",  action="store_true",
                        help="Validate only — do not copy or write any files")
    parser.add_argument("--smoke",    action="store_true",
                        help="Smoke-test mode: validate only the problems present in each shard")
    args = parser.parse_args()

    dry_run  = args.dry_run
    smoke    = args.smoke
    out_runs = Path(args.runs_dir).expanduser().resolve()

    print("CS 639 · Stage 0 · Merge Script")
    print("=" * 60)
    if smoke:
        print("  SMOKE TEST MODE — validating partial shards only\n")
    if dry_run:
        print("  DRY RUN — no files will be written\n")

    # ── 1. Resolve shard roots ─────────────────────────────────────────────
    log("Step 1 — Resolving shard roots ...")
    b_root = resolve_shard_root(args.shard_b, "B")
    c_root = resolve_shard_root(args.shard_c, "C")
    log(f"  Role B root : {b_root}")
    log(f"  Role C root : {c_root}")
    log(f"  Output      : {out_runs}\n")

    errors:   list[str] = []
    warnings: list[str] = []

    # ── 2. Determine problem ranges (smoke vs full) ────────────────────────
    if smoke:
        log("Step 2a — Inferring smoke-test problem IDs from shards ...")
        b_problems = infer_smoke_problems(b_root, "B")
        c_problems = infer_smoke_problems(c_root, "C")
        log("")
    else:
        b_problems = list(PROBLEMS_B)
        c_problems = list(PROBLEMS_C)

    # ── 3. Validate shards ─────────────────────────────────────────────────
    log("Step 2 — Validating Role B shard ...")
    validate_shard(b_root, "B", b_problems, errors, warnings)
    validate_csv(b_root, "B", b_problems, errors, warnings)
    log(f"  Errors so far: {len(errors)}  Warnings: {len(warnings)}\n")

    log("Step 3 — Validating Role C shard ...")
    validate_shard(c_root, "C", c_problems, errors, warnings)
    validate_csv(c_root, "C", c_problems, errors, warnings)
    log(f"  Errors so far: {len(errors)}  Warnings: {len(warnings)}\n")

    # ── 4. Check for problem range overlap / gaps ──────────────────────────
    log("Step 4 — Checking problem ID coverage ...")
    b_set   = set(b_problems)
    c_set   = set(c_problems)
    overlap = b_set & c_set
    if overlap:
        errors.append(f"Problem ID overlap between B and C: {sorted(overlap)}")
    if not smoke:
        gap = set(ALL_PROBLEMS) - b_set - c_set
        if gap:
            errors.append(f"Uncovered problem IDs: {sorted(gap)}")
    log(f"  B covers problems {b_problems}")
    log(f"  C covers problems {c_problems}")
    log(f"  Combined: {len(b_problems)+len(c_problems)} problems, "
        f"{len(RUN_NAMES)} run names = "
        f"{(len(b_problems)+len(c_problems)) * len(RUN_NAMES)} pairs\n")

    # ── 4. If validation failed and we're not dry-running, bail before copy ─
    if errors and not dry_run:
        log("Step 5 — SKIPPING file copy (validation errors present)")
        log("         Fix errors then re-run. Writing report ...\n")
        out_runs.mkdir(parents=True, exist_ok=True)
        write_report(out_runs, errors, warnings, {}, b_root, c_root, dry_run)
        sys.exit(1)

    # ── 5. Copy files ──────────────────────────────────────────────────────
    if not dry_run:
        log("Step 5 — Copying files into output runs/ ...")
        out_runs.mkdir(parents=True, exist_ok=True)
        copy_run_files(b_root, "B", b_problems, out_runs, dry_run=False)
        copy_run_files(c_root, "C", c_problems, out_runs, dry_run=False)
        log("  Copy complete.\n")
    else:
        log("Step 5 — [DRY RUN] Skipping file copy\n")

    # ── 6. Merge CSVs ─────────────────────────────────────────────────────
    log("Step 6 — Merging CSVs ...")
    merged_csvs = merge_csvs(b_root, c_root, out_runs, errors, dry_run)
    for csv_name, rows in merged_csvs.items():
        log(f"  {csv_name}: {len(rows)} rows")
    log("")

    # ── 7. Merge env_info ─────────────────────────────────────────────────
    log("Step 7 — Merging env_info.json ...")
    merge_env_info(b_root, c_root, out_runs, dry_run)
    if not dry_run:
        log(f"  Written: {out_runs}/env_info_merged.json\n")

    # ── 8. Final pair count check ─────────────────────────────────────────
    log("Step 8 — Final pair count check ...")
    manifest_rows = merged_csvs.get("generation_manifest.csv", [])
    validity_rows  = merged_csvs.get("validity_flags.csv", [])
    expected_pairs = len(RUN_NAMES) * (len(b_problems) + len(c_problems))
    if len(manifest_rows) != expected_pairs:
        errors.append(
            f"generation_manifest.csv has {len(manifest_rows)} rows, "
            f"expected {expected_pairs}"
        )
    if len(validity_rows) != expected_pairs:
        errors.append(
            f"validity_flags.csv has {len(validity_rows)} rows, "
            f"expected {expected_pairs}"
        )
    log(f"  Expected pairs : {expected_pairs}")
    log(f"  Manifest rows  : {len(manifest_rows)}")
    log(f"  Validity rows  : {len(validity_rows)}\n")

    # ── 9. Write report ───────────────────────────────────────────────────
    log("Step 9 — Writing merge_report.txt ...")
    ok = write_report(
        out_runs, errors, warnings, merged_csvs, b_root, c_root, dry_run,
        smoke=smoke, smoke_b_pids=b_problems, smoke_c_pids=c_problems,
    )

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

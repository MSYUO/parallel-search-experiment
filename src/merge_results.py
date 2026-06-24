#!/usr/bin/env python3
"""Merge per-worker JSON files into one v2 trial-level CSV row.

This script preserves the current worker output format and the existing T_real
definition: after merging all worker found_times, T_real is the global k-th
success timestamp.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


FIELDNAMES = [
    "experiment_name",
    "environment",
    "difficulty",
    "k",
    "N",
    "trial",
    "T_real",
    "total_successes_collected",
    "total_worker_hashes",
    "mean_worker_elapsed_seconds",
    "max_worker_elapsed_seconds",
    "min_worker_elapsed_seconds",
    "result_file_count",
    "timestamp",
]


class MergeError(ValueError):
    """Raised when worker files cannot be merged safely."""


def _as_float_list(values: Any, path: Path) -> list[float]:
    if not isinstance(values, list):
        raise MergeError(f"{path}: found_times must be a list")

    found_times: list[float] = []
    for value in values:
        try:
            found_times.append(float(value))
        except (TypeError, ValueError) as exc:
            raise MergeError(f"{path}: found_times contains a non-numeric value") from exc
    return found_times


def _optional_float(value: Any, path: Path, field: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MergeError(f"{path}: {field} must be numeric when present") from exc


def _optional_int(value: Any, path: Path, field: str) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MergeError(f"{path}: {field} must be an integer when present") from exc


def load_worker_result(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise MergeError(f"{path}: invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise MergeError(f"{path}: worker result must be a JSON object")

    return {
        "path": path,
        "found_times": _as_float_list(data.get("found_times"), path),
        "total_hashes": _optional_int(data.get("total_hashes"), path, "total_hashes"),
        "elapsed_seconds": _optional_float(
            data.get("elapsed_seconds"), path, "elapsed_seconds"
        ),
    }


def find_worker_files(results_dir: Path, trial: int) -> list[Path]:
    pattern = str(results_dir / f"result_w*_t{trial}.json")
    return [Path(path) for path in sorted(glob.glob(pattern))]


def merge_trial(
    *,
    experiment_name: str,
    environment: str,
    difficulty: int,
    k: int,
    n_workers: int,
    trial: int,
    results_dir: Path,
    expected_n: int | None,
    allow_partial: bool,
    timestamp: str | None = None,
) -> dict[str, Any]:
    worker_files = find_worker_files(results_dir, trial)

    if not worker_files:
        raise MergeError(
            f"no worker JSON files found for trial={trial} in {results_dir}"
        )

    if expected_n is not None and len(worker_files) != expected_n:
        message = (
            f"expected {expected_n} worker result file(s) for trial={trial}, "
            f"but found {len(worker_files)} in {results_dir}"
        )
        if not allow_partial:
            raise MergeError(message)
        print(f"[merge_results] WARNING: {message}", file=sys.stderr)

    all_found_times: list[float] = []
    worker_elapsed_seconds: list[float] = []
    total_worker_hashes = 0

    for path in worker_files:
        worker = load_worker_result(path)
        all_found_times.extend(worker["found_times"])
        total_worker_hashes += worker["total_hashes"]
        if worker["elapsed_seconds"] is not None:
            worker_elapsed_seconds.append(worker["elapsed_seconds"])

    all_found_times.sort()
    total_successes_collected = len(all_found_times)

    if total_successes_collected < k:
        raise MergeError(
            f"need at least k={k} successes for trial={trial}, "
            f"but only collected {total_successes_collected}"
        )

    t_real = all_found_times[k - 1]

    if worker_elapsed_seconds:
        mean_elapsed = mean(worker_elapsed_seconds)
        max_elapsed = max(worker_elapsed_seconds)
        min_elapsed = min(worker_elapsed_seconds)
    else:
        mean_elapsed = max_elapsed = min_elapsed = 0.0
        print(
            "[merge_results] WARNING: no elapsed_seconds fields found; "
            "worker elapsed summary set to 0.0",
            file=sys.stderr,
        )

    return {
        "experiment_name": experiment_name,
        "environment": environment,
        "difficulty": difficulty,
        "k": k,
        "N": n_workers,
        "trial": trial,
        "T_real": round(t_real, 6),
        "total_successes_collected": total_successes_collected,
        "total_worker_hashes": total_worker_hashes,
        "mean_worker_elapsed_seconds": round(mean_elapsed, 6),
        "max_worker_elapsed_seconds": round(max_elapsed, 6),
        "min_worker_elapsed_seconds": round(min_elapsed, 6),
        "result_file_count": len(worker_files),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    }


def append_csv(row: dict[str, Any], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge worker result JSON files for one trial into a v2 merged CSV row."
        )
    )
    parser.add_argument("--experiment-name", required=True, help="Experiment name.")
    parser.add_argument(
        "--environment",
        default="docker-compose",
        help="Execution environment label. Default: docker-compose.",
    )
    parser.add_argument("--difficulty", type=positive_int, required=True)
    parser.add_argument("--k", type=positive_int, required=True)
    parser.add_argument(
        "--N",
        "--n-workers",
        dest="n_workers",
        type=positive_int,
        required=True,
        help="Worker count recorded for the merged row.",
    )
    parser.add_argument("--trial", type=positive_int, required=True)
    parser.add_argument(
        "--expected-N",
        dest="expected_n",
        type=positive_int,
        default=None,
        help=(
            "Expected number of worker JSON files. Defaults to --N when omitted."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Warn instead of failing when the worker file count differs from "
            "expected-N. Use only for manual recovery of partial data."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing result_w*_t<trial>.json files.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/merged/merged_trials_v2.csv"),
        help="Output merged CSV path. Parent directories are created.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional timestamp override for reproducible imports.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    expected_n = args.expected_n if args.expected_n is not None else args.n_workers

    try:
        row = merge_trial(
            experiment_name=args.experiment_name,
            environment=args.environment,
            difficulty=args.difficulty,
            k=args.k,
            n_workers=args.n_workers,
            trial=args.trial,
            results_dir=args.results_dir,
            expected_n=expected_n,
            allow_partial=args.allow_partial,
            timestamp=args.timestamp,
        )
        append_csv(row, args.csv)
    except MergeError as exc:
        print(f"[merge_results] ERROR: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"[merge_results] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[merge_results] T_real={row['T_real']}s")
    print(f"[merge_results] result_file_count={row['result_file_count']}")
    print(f"[merge_results] appended row to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

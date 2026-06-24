#!/usr/bin/env python3
"""Build v2 summary statistics from merged trial CSV data.

The analysis uses workload-specific N=1 calibration for each
experiment/environment/difficulty/k group. It intentionally avoids a global
lambda estimate.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


INPUT_REQUIRED_FIELDS = {
    "experiment_name",
    "environment",
    "difficulty",
    "k",
    "N",
    "trial",
    "T_real",
}

SUMMARY_FIELDNAMES = [
    "experiment_name",
    "environment",
    "difficulty",
    "k",
    "N",
    "trials_completed",
    "mean_T_real",
    "std_T_real",
    "ci95_low",
    "ci95_high",
    "lambda_hat",
    "T_theory",
    "delta_N",
    "speedup",
    "parallel_efficiency",
    "relative_error_percent",
    "oracle_N",
    "oracle_T",
    "core_count_N",
    "core_count_T",
    "core_count_regret_seconds",
    "core_count_regret_percent",
    "oracle_differs_from_core_count",
]

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


class SummaryError(ValueError):
    """Raised when merged CSV data cannot be summarized."""


def parse_int(row: dict[str, str], field: str, row_number: int) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise SummaryError(f"row {row_number}: {field} must be an integer") from exc


def parse_float(row: dict[str, str], field: str, row_number: int) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise SummaryError(f"row {row_number}: {field} must be numeric") from exc


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        raise SummaryError(f"input CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []

        missing = sorted(INPUT_REQUIRED_FIELDS - set(reader.fieldnames))
        if missing:
            raise SummaryError(
                f"input CSV is missing required field(s): {', '.join(missing)}"
            )

        rows: list[dict[str, Any]] = []
        for row_number, row in enumerate(reader, start=2):
            experiment_name = (row.get("experiment_name") or "").strip()
            environment = (row.get("environment") or "").strip()
            if not experiment_name:
                raise SummaryError(f"row {row_number}: experiment_name is required")
            if not environment:
                raise SummaryError(f"row {row_number}: environment is required")

            rows.append(
                {
                    "experiment_name": experiment_name,
                    "environment": environment,
                    "difficulty": parse_int(row, "difficulty", row_number),
                    "k": parse_int(row, "k", row_number),
                    "N": parse_int(row, "N", row_number),
                    "trial": parse_int(row, "trial", row_number),
                    "T_real": parse_float(row, "T_real", row_number),
                }
            )
    return rows


def t_critical_95(df: int) -> float:
    if df <= 0:
        return 0.0
    return T_CRITICAL_95.get(df, 1.96)


def summarize_values(values: list[float]) -> dict[str, float]:
    n = len(values)
    avg = mean(values)
    if n < 2:
        return {
            "mean": avg,
            "std": 0.0,
            "ci95_low": avg,
            "ci95_high": avg,
        }

    sample_std = stdev(values)
    half_width = t_critical_95(n - 1) * sample_std / math.sqrt(n)
    return {
        "mean": avg,
        "std": sample_std,
        "ci95_low": avg - half_width,
        "ci95_high": avg + half_width,
    }


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def format_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


def build_summary(rows: list[dict[str, Any]], core_count_n: int) -> list[dict[str, str]]:
    grouped_values: dict[tuple[str, str, int, int, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["experiment_name"],
            row["environment"],
            row["difficulty"],
            row["k"],
            row["N"],
        )
        grouped_values[key].append(row["T_real"])

    condition_stats: dict[tuple[str, str, int, int, int], dict[str, float]] = {}
    for key, values in grouped_values.items():
        condition_stats[key] = summarize_values(values)

    workload_keys = sorted(
        {
            (experiment_name, environment, difficulty, k)
            for experiment_name, environment, difficulty, k, _n in grouped_values
        }
    )

    workload_metadata: dict[tuple[str, str, int, int], dict[str, float | int | None]] = {}

    for workload_key in workload_keys:
        experiment_name, environment, difficulty, k = workload_key
        n_keys = [
            key
            for key in condition_stats
            if key[:4] == workload_key
        ]

        oracle_key = min(n_keys, key=lambda key: condition_stats[key]["mean"])
        oracle_n = oracle_key[4]
        oracle_t = condition_stats[oracle_key]["mean"]

        n1_key = (experiment_name, environment, difficulty, k, 1)
        n1_stats = condition_stats.get(n1_key)
        if n1_stats is None:
            print(
                "[analysis_summary] WARNING: missing N=1 calibration for "
                f"experiment={experiment_name}, environment={environment}, "
                f"difficulty={difficulty}, k={k}; theory-based metrics skipped",
                file=sys.stderr,
            )
            lambda_hat = None
            mean_t_n1 = None
        else:
            mean_t_n1 = n1_stats["mean"]
            lambda_hat = k / mean_t_n1 if mean_t_n1 > 0 else None

        core_key = (experiment_name, environment, difficulty, k, core_count_n)
        core_stats = condition_stats.get(core_key)
        core_t = core_stats["mean"] if core_stats is not None else None
        if core_t is None:
            core_regret_seconds = None
            core_regret_percent = None
        else:
            core_regret_seconds = core_t - oracle_t
            core_regret_percent = (
                (core_regret_seconds / oracle_t) * 100.0 if oracle_t > 0 else None
            )

        workload_metadata[workload_key] = {
            "lambda_hat": lambda_hat,
            "mean_t_n1": mean_t_n1,
            "oracle_N": oracle_n,
            "oracle_T": oracle_t,
            "core_count_T": core_t,
            "core_count_regret_seconds": core_regret_seconds,
            "core_count_regret_percent": core_regret_percent,
            "oracle_differs_from_core_count": oracle_n != core_count_n,
        }

    summary_rows: list[dict[str, str]] = []
    for key in sorted(condition_stats):
        experiment_name, environment, difficulty, k, n_workers = key
        stats = condition_stats[key]
        metadata = workload_metadata[key[:4]]

        lambda_hat = metadata["lambda_hat"]
        mean_t_n1 = metadata["mean_t_n1"]
        if isinstance(lambda_hat, float) and lambda_hat > 0:
            t_theory = k / (n_workers * lambda_hat)
            delta_n = stats["mean"] - t_theory
            speedup = (
                mean_t_n1 / stats["mean"]
                if isinstance(mean_t_n1, float) and stats["mean"] > 0
                else None
            )
            parallel_efficiency = speedup / n_workers if speedup is not None else None
            relative_error_percent = (
                (delta_n / t_theory) * 100.0 if t_theory > 0 else None
            )
        else:
            t_theory = None
            delta_n = None
            speedup = None
            parallel_efficiency = None
            relative_error_percent = None

        summary_rows.append(
            {
                "experiment_name": experiment_name,
                "environment": environment,
                "difficulty": str(difficulty),
                "k": str(k),
                "N": str(n_workers),
                "trials_completed": str(len(grouped_values[key])),
                "mean_T_real": format_float(stats["mean"]),
                "std_T_real": format_float(stats["std"]),
                "ci95_low": format_float(stats["ci95_low"]),
                "ci95_high": format_float(stats["ci95_high"]),
                "lambda_hat": format_float(
                    lambda_hat if isinstance(lambda_hat, float) else None
                ),
                "T_theory": format_float(t_theory),
                "delta_N": format_float(delta_n),
                "speedup": format_float(speedup),
                "parallel_efficiency": format_float(parallel_efficiency),
                "relative_error_percent": format_float(relative_error_percent),
                "oracle_N": format_int(
                    metadata["oracle_N"] if isinstance(metadata["oracle_N"], int) else None
                ),
                "oracle_T": format_float(
                    metadata["oracle_T"]
                    if isinstance(metadata["oracle_T"], float)
                    else None
                ),
                "core_count_N": str(core_count_n),
                "core_count_T": format_float(
                    metadata["core_count_T"]
                    if isinstance(metadata["core_count_T"], float)
                    else None
                ),
                "core_count_regret_seconds": format_float(
                    metadata["core_count_regret_seconds"]
                    if isinstance(metadata["core_count_regret_seconds"], float)
                    else None
                ),
                "core_count_regret_percent": format_float(
                    metadata["core_count_regret_percent"]
                    if isinstance(metadata["core_count_regret_percent"], float)
                    else None
                ),
                "oracle_differs_from_core_count": str(
                    metadata["oracle_differs_from_core_count"]
                ).lower(),
            }
        )

    return summary_rows


def write_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


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
            "Compute v2 summary statistics from merged trial CSV data using "
            "workload-specific N=1 lambda calibration."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/merged/merged_trials_v2.csv"),
        help="Input merged trial CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/summary/summary_statistics_v2.csv"),
        help="Output summary CSV path.",
    )
    parser.add_argument(
        "--core-count-N",
        dest="core_count_n",
        type=positive_int,
        default=6,
        help="Core-count baseline worker count. Default: 6.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        rows = load_rows(args.csv)
        if not rows:
            print(
                f"[analysis_summary] WARNING: input CSV has no data rows: {args.csv}",
                file=sys.stderr,
            )
        summary_rows = build_summary(rows, args.core_count_n) if rows else []
        write_summary(summary_rows, args.output)
    except (OSError, SummaryError) as exc:
        print(f"[analysis_summary] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[analysis_summary] wrote {len(summary_rows)} summary row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

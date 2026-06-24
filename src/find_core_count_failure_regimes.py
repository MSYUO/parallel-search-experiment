#!/usr/bin/env python3
"""Find workload regimes where Core-count is not near-Oracle.

This script reads v2 summary statistics and writes one row per
experiment/environment/difficulty/k workload group. It does not run experiments
or mutate source data.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "experiment_name",
    "environment",
    "difficulty",
    "k",
    "N",
    "mean_T_real",
}

OUTPUT_FIELDNAMES = [
    "experiment_name",
    "environment",
    "difficulty",
    "k",
    "oracle_N",
    "oracle_T",
    "core_count_N",
    "core_count_T",
    "core_count_regret_seconds",
    "core_count_regret_percent",
    "core_count_failure",
    "strong_failure",
    "recommended_for_confirmatory",
    "notes",
]


class FailureRegimeError(ValueError):
    """Raised when input data cannot be analyzed."""


def parse_int(row: dict[str, str], field: str, row_number: int) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise FailureRegimeError(
            f"row {row_number}: {field} must be an integer"
        ) from exc


def parse_float(row: dict[str, str], field: str, row_number: int) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise FailureRegimeError(f"row {row_number}: {field} must be numeric") from exc


def load_summary_rows(summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        raise FailureRegimeError(f"summary CSV not found: {summary_path}")

    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []

        missing = sorted(REQUIRED_FIELDS - set(reader.fieldnames))
        if missing:
            raise FailureRegimeError(
                f"summary CSV is missing required field(s): {', '.join(missing)}"
            )

        rows: list[dict[str, Any]] = []
        for row_number, row in enumerate(reader, start=2):
            experiment_name = (row.get("experiment_name") or "").strip()
            environment = (row.get("environment") or "").strip()
            if not experiment_name:
                raise FailureRegimeError(
                    f"row {row_number}: experiment_name is required"
                )
            if not environment:
                raise FailureRegimeError(f"row {row_number}: environment is required")

            rows.append(
                {
                    "experiment_name": experiment_name,
                    "environment": environment,
                    "difficulty": parse_int(row, "difficulty", row_number),
                    "k": parse_int(row, "k", row_number),
                    "N": parse_int(row, "N", row_number),
                    "mean_T_real": parse_float(row, "mean_T_real", row_number),
                }
            )
    return rows


def group_by_workload(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["experiment_name"],
            row["environment"],
            row["difficulty"],
            row["k"],
        )
        grouped[key].append(row)
    return grouped


def collapse_duplicate_n_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Average duplicate N rows defensively; summary input should normally be unique."""
    by_n: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_n[row["N"]].append(row["mean_T_real"])

    collapsed: list[dict[str, Any]] = []
    notes: list[str] = []
    for n_workers in sorted(by_n):
        values = by_n[n_workers]
        if len(values) > 1:
            notes.append(f"duplicate N={n_workers} rows averaged")
        collapsed.append({"N": n_workers, "mean_T_real": sum(values) / len(values)})
    return collapsed, notes


def analyze_group(
    key: tuple[str, str, int, int],
    rows: list[dict[str, Any]],
    *,
    core_count_n: int,
    failure_threshold: float,
    strong_failure_threshold: float,
) -> dict[str, Any]:
    experiment_name, environment, difficulty, k = key
    n_rows, notes = collapse_duplicate_n_rows(rows)

    oracle_row = min(n_rows, key=lambda row: row["mean_T_real"])
    oracle_n = oracle_row["N"]
    oracle_t = oracle_row["mean_T_real"]

    core_rows = [row for row in n_rows if row["N"] == core_count_n]
    if not core_rows:
        notes.append(
            f"Core-count N={core_count_n} not available; failure cannot be assessed"
        )
        return {
            "experiment_name": experiment_name,
            "environment": environment,
            "difficulty": difficulty,
            "k": k,
            "oracle_N": oracle_n,
            "oracle_T": oracle_t,
            "core_count_N": core_count_n,
            "core_count_T": None,
            "core_count_regret_seconds": None,
            "core_count_regret_percent": None,
            "core_count_failure": False,
            "strong_failure": False,
            "recommended_for_confirmatory": False,
            "notes": "; ".join(notes),
        }

    core_t = core_rows[0]["mean_T_real"]
    regret_seconds = core_t - oracle_t
    regret_percent = (regret_seconds / oracle_t) * 100.0 if oracle_t > 0 else None

    core_count_failure = (
        regret_percent is not None and regret_percent > failure_threshold
    )
    strong_failure = (
        regret_percent is not None and regret_percent > strong_failure_threshold
    )

    if strong_failure:
        notes.append("Core-count regret exceeds strong-failure threshold")
    elif core_count_failure:
        notes.append("Core-count regret exceeds failure threshold")
    else:
        notes.append("Core-count is near-Oracle under configured threshold")

    return {
        "experiment_name": experiment_name,
        "environment": environment,
        "difficulty": difficulty,
        "k": k,
        "oracle_N": oracle_n,
        "oracle_T": oracle_t,
        "core_count_N": core_count_n,
        "core_count_T": core_t,
        "core_count_regret_seconds": regret_seconds,
        "core_count_regret_percent": regret_percent,
        "core_count_failure": core_count_failure,
        "strong_failure": strong_failure,
        "recommended_for_confirmatory": False,
        "notes": "; ".join(notes),
    }


def mark_recommendations(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    failures = [
        row
        for row in rows
        if row["core_count_failure"] and row["core_count_regret_percent"] is not None
    ]
    failures.sort(key=lambda row: row["core_count_regret_percent"], reverse=True)

    recommended_ids = {id(row) for row in failures[:top_k]}
    for row in rows:
        if id(row) in recommended_ids:
            row["recommended_for_confirmatory"] = True
            row["notes"] = append_note(
                row["notes"],
                "recommended as a top Core-count failure regime for confirmatory validation",
            )
    return rows


def append_note(notes: str, note: str) -> str:
    if not notes:
        return note
    return f"{notes}; {note}"


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def format_bool(value: bool) -> str:
    return str(value).lower()


def output_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "experiment_name": row["experiment_name"],
        "environment": row["environment"],
        "difficulty": str(row["difficulty"]),
        "k": str(row["k"]),
        "oracle_N": str(row["oracle_N"]),
        "oracle_T": format_float(row["oracle_T"]),
        "core_count_N": str(row["core_count_N"]),
        "core_count_T": format_float(row["core_count_T"]),
        "core_count_regret_seconds": format_float(row["core_count_regret_seconds"]),
        "core_count_regret_percent": format_float(row["core_count_regret_percent"]),
        "core_count_failure": format_bool(row["core_count_failure"]),
        "strong_failure": format_bool(row["strong_failure"]),
        "recommended_for_confirmatory": format_bool(
            row["recommended_for_confirmatory"]
        ),
        "notes": row["notes"],
    }


def write_failure_report(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_row(row) for row in rows)


def print_regime(row: dict[str, Any]) -> None:
    regret = row["core_count_regret_percent"]
    regret_text = "N/A" if regret is None else f"{regret:.2f}%"
    core_t = row["core_count_T"]
    core_text = "N/A" if core_t is None else f"{core_t:.6f}s"
    print(
        "  "
        f"{row['experiment_name']} | {row['environment']} | "
        f"difficulty={row['difficulty']} k={row['k']} | "
        f"oracle_N={row['oracle_N']} oracle_T={row['oracle_T']:.6f}s | "
        f"core_N={row['core_count_N']} core_T={core_text} | "
        f"regret={regret_text}"
    )


def print_report(rows: list[dict[str, Any]], top_k: int) -> None:
    assessable = [row for row in rows if row["core_count_regret_percent"] is not None]
    failures = [row for row in assessable if row["core_count_failure"]]
    strong_failures = [row for row in assessable if row["strong_failure"]]
    near_oracle = [row for row in assessable if not row["core_count_failure"]]

    failures_sorted = sorted(
        failures, key=lambda row: row["core_count_regret_percent"], reverse=True
    )
    recommended = [row for row in rows if row["recommended_for_confirmatory"]]

    print(f"[core-count] regimes assessed: {len(assessable)}")
    print(f"[core-count] near-Oracle regimes: {len(near_oracle)}")
    print(f"[core-count] failure regimes: {len(failures)}")
    print(f"[core-count] strong failure regimes: {len(strong_failures)}")

    if failures_sorted:
        print(f"[core-count] top {min(top_k, len(failures_sorted))} strongest failure regimes:")
        for row in failures_sorted[:top_k]:
            print_regime(row)
    else:
        print("[core-count] no Core-count failure regimes found.")
        print(
            "[core-count] Recommendation: report this honestly and consider "
            "reframing the paper as a measurement study of Core-count strength."
        )

    if recommended:
        print("[core-count] recommended confirmatory regimes:")
        for row in recommended:
            print_regime(row)
    else:
        print("[core-count] recommended confirmatory regimes: none")


def find_failure_regimes(
    summary_path: Path,
    out_path: Path,
    *,
    core_count_n: int,
    failure_threshold: float,
    strong_failure_threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    summary_rows = load_summary_rows(summary_path)
    grouped = group_by_workload(summary_rows)

    report_rows = [
        analyze_group(
            key,
            rows,
            core_count_n=core_count_n,
            failure_threshold=failure_threshold,
            strong_failure_threshold=strong_failure_threshold,
        )
        for key, rows in sorted(grouped.items())
    ]

    mark_recommendations(report_rows, top_k)
    write_failure_report(report_rows, out_path)
    return report_rows


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be numeric") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Identify difficulty/k workload regimes where Core-count is not "
            "near-Oracle."
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/summary/summary_statistics_v2.csv"),
        help="Input v2 summary statistics CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/summary/core_count_failure_regimes.csv"),
        help="Output Core-count failure regime CSV.",
    )
    parser.add_argument(
        "--core-count-N",
        dest="core_count_n",
        type=positive_int,
        default=6,
        help="Core-count baseline worker count. Default: 6.",
    )
    parser.add_argument(
        "--failure-threshold",
        type=nonnegative_float,
        default=5.0,
        help="Failure threshold in regret percent. Default: 5.",
    )
    parser.add_argument(
        "--strong-failure-threshold",
        type=nonnegative_float,
        default=10.0,
        help="Strong failure threshold in regret percent. Default: 10.",
    )
    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=5,
        help="Number of strongest failures to print and recommend. Default: 5.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.strong_failure_threshold < args.failure_threshold:
        print(
            "[core-count] ERROR: strong-failure-threshold must be greater than "
            "or equal to failure-threshold",
            file=sys.stderr,
        )
        return 1

    try:
        rows = find_failure_regimes(
            args.summary,
            args.out,
            core_count_n=args.core_count_n,
            failure_threshold=args.failure_threshold,
            strong_failure_threshold=args.strong_failure_threshold,
            top_k=args.top_k,
        )
    except (OSError, FailureRegimeError) as exc:
        print(f"[core-count] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[core-count] wrote {len(rows)} regime row(s) to {args.out}")
    print_report(rows, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

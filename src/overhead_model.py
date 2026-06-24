#!/usr/bin/env python3
"""Fit empirical overhead models for summary delta_N values.

Models are fit separately for each experiment/environment/difficulty/k workload
group. Predictions are conservative: predicted delta is clipped to be
nonnegative.
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
    "lambda_hat",
    "T_theory",
    "delta_N",
}

MODEL_TYPES = {"lookup", "linear", "quadratic"}

OUTPUT_FIELDNAMES = [
    "experiment_name",
    "environment",
    "difficulty",
    "k",
    "model_type",
    "N",
    "observed_delta",
    "predicted_delta",
    "prediction_error",
    "notes",
]


class OverheadModelError(ValueError):
    """Raised when overhead model input cannot be processed."""


def parse_int(row: dict[str, str], field: str, row_number: int) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise OverheadModelError(
            f"row {row_number}: {field} must be an integer"
        ) from exc


def parse_required_float(row: dict[str, str], field: str, row_number: int) -> float:
    raw_value = row.get(field, "")
    if raw_value == "":
        raise OverheadModelError(f"row {row_number}: {field} is required")
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise OverheadModelError(f"row {row_number}: {field} must be numeric") from exc


def load_summary_rows(summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        raise OverheadModelError(f"summary CSV not found: {summary_path}")

    rows: list[dict[str, Any]] = []
    skipped_missing_delta = 0

    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []

        missing = sorted(REQUIRED_FIELDS - set(reader.fieldnames))
        if missing:
            raise OverheadModelError(
                f"summary CSV is missing required field(s): {', '.join(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            experiment_name = (row.get("experiment_name") or "").strip()
            environment = (row.get("environment") or "").strip()
            if not experiment_name:
                raise OverheadModelError(
                    f"row {row_number}: experiment_name is required"
                )
            if not environment:
                raise OverheadModelError(f"row {row_number}: environment is required")

            if (row.get("delta_N") or "") == "":
                skipped_missing_delta += 1
                continue

            rows.append(
                {
                    "experiment_name": experiment_name,
                    "environment": environment,
                    "difficulty": parse_int(row, "difficulty", row_number),
                    "k": parse_int(row, "k", row_number),
                    "N": parse_int(row, "N", row_number),
                    "mean_T_real": parse_required_float(
                        row, "mean_T_real", row_number
                    ),
                    "lambda_hat": parse_required_float(row, "lambda_hat", row_number),
                    "T_theory": parse_required_float(row, "T_theory", row_number),
                    "delta_N": parse_required_float(row, "delta_N", row_number),
                }
            )

    if skipped_missing_delta:
        print(
            "[overhead_model] WARNING: skipped "
            f"{skipped_missing_delta} row(s) with missing delta_N",
            file=sys.stderr,
        )

    return rows


def group_by_workload(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["experiment_name"], row["environment"], row["difficulty"], row["k"])
        ].append(row)
    return grouped


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense linear system with Gaussian elimination."""
    n = len(vector)
    augmented = [matrix[i][:] + [vector[i]] for i in range(n)]

    for column in range(n):
        pivot_row = max(
            range(column, n), key=lambda row_index: abs(augmented[row_index][column])
        )
        if abs(augmented[pivot_row][column]) < 1e-12:
            raise OverheadModelError("singular polynomial fit")

        if pivot_row != column:
            augmented[column], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[column],
            )

        pivot = augmented[column][column]
        for col in range(column, n + 1):
            augmented[column][col] /= pivot

        for row_index in range(n):
            if row_index == column:
                continue
            factor = augmented[row_index][column]
            for col in range(column, n + 1):
                augmented[row_index][col] -= factor * augmented[column][col]

    return [augmented[row_index][n] for row_index in range(n)]


def fit_polynomial(points: list[tuple[float, float]], degree: int) -> list[float]:
    """Return coefficients c0..cd for y = c0 + c1*x + ... + cd*x^d."""
    if len(points) < degree + 1:
        raise OverheadModelError(
            f"need at least {degree + 1} points for degree-{degree} fit"
        )
    distinct_x = {x for x, _y in points}
    if len(distinct_x) < degree + 1:
        raise OverheadModelError(
            f"need at least {degree + 1} distinct N values for degree-{degree} fit"
        )

    size = degree + 1
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    vector = [0.0 for _ in range(size)]

    for x, y in points:
        powers = [1.0]
        for _ in range(1, 2 * degree + 1):
            powers.append(powers[-1] * x)
        for row in range(size):
            vector[row] += y * powers[row]
            for col in range(size):
                matrix[row][col] += powers[row + col]

    return solve_linear_system(matrix, vector)


def predict_polynomial(coefficients: list[float], x: float) -> float:
    prediction = 0.0
    power = 1.0
    for coefficient in coefficients:
        prediction += coefficient * power
        power *= x
    return prediction


def conservative_delta(value: float) -> float:
    return max(0.0, value)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def skipped_rows(
    rows: list[dict[str, Any]], model_type: str, note: str
) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: item["N"]):
        output_rows.append(
            {
                "experiment_name": row["experiment_name"],
                "environment": row["environment"],
                "difficulty": str(row["difficulty"]),
                "k": str(row["k"]),
                "model_type": model_type,
                "N": str(row["N"]),
                "observed_delta": format_float(row["delta_N"]),
                "predicted_delta": "",
                "prediction_error": "",
                "notes": note,
            }
        )
    return output_rows


def output_prediction_row(
    row: dict[str, Any], model_type: str, predicted_delta: float, notes: str
) -> dict[str, str]:
    return {
        "experiment_name": row["experiment_name"],
        "environment": row["environment"],
        "difficulty": str(row["difficulty"]),
        "k": str(row["k"]),
        "model_type": model_type,
        "N": str(row["N"]),
        "observed_delta": format_float(row["delta_N"]),
        "predicted_delta": format_float(predicted_delta),
        "prediction_error": format_float(predicted_delta - row["delta_N"]),
        "notes": notes,
    }


def fit_lookup(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: item["N"]):
        predicted_delta = conservative_delta(row["delta_N"])
        output_rows.append(
            output_prediction_row(
                row,
                "lookup",
                predicted_delta,
                "lookup uses observed delta_N at measured N",
            )
        )
    return output_rows


def fit_polynomial_model(
    rows: list[dict[str, Any]], model_type: str, degree: int
) -> list[dict[str, str]]:
    points = [(float(row["N"]), float(row["delta_N"])) for row in rows]
    try:
        coefficients = fit_polynomial(points, degree)
    except OverheadModelError as exc:
        note = f"skipped {model_type}: {exc}"
        print(f"[overhead_model] WARNING: {note}", file=sys.stderr)
        return skipped_rows(rows, model_type, note)

    output_rows: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: item["N"]):
        fitted_delta = predict_polynomial(coefficients, float(row["N"]))
        predicted_delta = conservative_delta(fitted_delta)
        output_rows.append(
            output_prediction_row(
                row,
                model_type,
                predicted_delta,
                "conservative nonnegative polynomial prediction",
            )
        )
    return output_rows


def fit_models_for_group(
    rows: list[dict[str, Any]], model_types: list[str]
) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for model_type in model_types:
        if model_type == "lookup":
            output_rows.extend(fit_lookup(rows))
        elif model_type == "linear":
            output_rows.extend(fit_polynomial_model(rows, "linear", degree=1))
        elif model_type == "quadratic":
            output_rows.extend(fit_polynomial_model(rows, "quadratic", degree=2))
        else:
            raise OverheadModelError(f"unsupported model type: {model_type}")
    return output_rows


def parse_model_types(raw_value: str) -> list[str]:
    values = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one model type is required")
    invalid = sorted(set(values) - MODEL_TYPES)
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unsupported model type(s): {', '.join(invalid)}"
        )
    return values


def write_report(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    summary_path: Path, out_path: Path, model_types: list[str]
) -> list[dict[str, str]]:
    rows = load_summary_rows(summary_path)
    grouped = group_by_workload(rows)

    report_rows: list[dict[str, str]] = []
    for _key, group_rows in sorted(grouped.items()):
        report_rows.extend(fit_models_for_group(group_rows, model_types))

    write_report(report_rows, out_path)
    return report_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit empirical overhead models for delta_N from v2 summary statistics."
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
        default=Path("results/summary/overhead_model_report.csv"),
        help="Output overhead model report CSV.",
    )
    parser.add_argument(
        "--model-types",
        type=parse_model_types,
        default=parse_model_types("lookup,linear,quadratic"),
        help="Comma-separated model types: lookup,linear,quadratic.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        report_rows = build_report(args.summary, args.out, args.model_types)
    except (OSError, OverheadModelError) as exc:
        print(f"[overhead_model] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[overhead_model] wrote {len(report_rows)} row(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

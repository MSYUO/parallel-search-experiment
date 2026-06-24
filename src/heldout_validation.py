#!/usr/bin/env python3
"""Held-out validation for empirical overhead prediction.

This script evaluates whether delta_N models predict data outside the rows used
for fitting. It does not execute experiments and does not select or claim a
deployed policy.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from overhead_model import (
        OverheadModelError,
        conservative_delta,
        fit_polynomial,
        predict_polynomial,
    )
except ImportError:  # pragma: no cover - supports python -m src.heldout_validation
    from src.overhead_model import (
        OverheadModelError,
        conservative_delta,
        fit_polynomial,
        predict_polynomial,
    )


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
    "validation_mode",
    "experiment_name",
    "environment",
    "train_difficulty",
    "test_difficulty",
    "train_k",
    "test_k",
    "heldout_N",
    "model_type",
    "predicted_delta",
    "observed_delta",
    "delta_error",
    "predicted_T",
    "observed_T",
    "oracle_N",
    "oracle_T",
    "regret_seconds",
    "regret_percent",
    "notes",
]

DIFFICULTY_TRAIN = [4, 5]
DIFFICULTY_TEST = [6]
K_TRAIN = [1, 3, 10]
K_TEST = [20, 40]


class HeldoutValidationError(ValueError):
    """Raised when held-out validation input cannot be processed."""


def parse_int(row: dict[str, str], field: str, row_number: int) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise HeldoutValidationError(
            f"row {row_number}: {field} must be an integer"
        ) from exc


def parse_optional_float(
    row: dict[str, str], field: str, row_number: int
) -> float | None:
    raw_value = row.get(field, "")
    if raw_value == "":
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise HeldoutValidationError(
            f"row {row_number}: {field} must be numeric when present"
        ) from exc


def parse_required_float(row: dict[str, str], field: str, row_number: int) -> float:
    value = parse_optional_float(row, field, row_number)
    if value is None:
        raise HeldoutValidationError(f"row {row_number}: {field} is required")
    return value


def load_summary_rows(summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        raise HeldoutValidationError(f"summary CSV not found: {summary_path}")

    rows: list[dict[str, Any]] = []
    skipped_missing_delta = 0

    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []

        missing = sorted(REQUIRED_FIELDS - set(reader.fieldnames))
        if missing:
            raise HeldoutValidationError(
                f"summary CSV is missing required field(s): {', '.join(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            experiment_name = (row.get("experiment_name") or "").strip()
            environment = (row.get("environment") or "").strip()
            if not experiment_name:
                raise HeldoutValidationError(
                    f"row {row_number}: experiment_name is required"
                )
            if not environment:
                raise HeldoutValidationError(
                    f"row {row_number}: environment is required"
                )

            delta_n = parse_optional_float(row, "delta_N", row_number)
            if delta_n is None:
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
                    "lambda_hat": parse_optional_float(
                        row, "lambda_hat", row_number
                    ),
                    "T_theory": parse_optional_float(row, "T_theory", row_number),
                    "delta_N": delta_n,
                }
            )

    if skipped_missing_delta:
        print(
            "[heldout_validation] WARNING: skipped "
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


def unique_join(values: list[int] | set[int]) -> str:
    return ";".join(str(value) for value in sorted(values))


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def target_lambda_for_workload(
    test_rows: list[dict[str, Any]], context: str
) -> float | None:
    n1_rows = [row for row in test_rows if row["N"] == 1 and row["lambda_hat"]]
    if not n1_rows:
        print(
            "[heldout_validation] WARNING: missing target N=1 lambda calibration "
            f"for {context}; skipping held-out case",
            file=sys.stderr,
        )
        return None
    return n1_rows[0]["lambda_hat"]


def oracle_for_rows(test_rows: list[dict[str, Any]]) -> tuple[int, float]:
    oracle_row = min(test_rows, key=lambda row: row["mean_T_real"])
    return oracle_row["N"], oracle_row["mean_T_real"]


def fit_lookup_model(train_rows: list[dict[str, Any]]) -> dict[int, float]:
    by_n: dict[int, list[float]] = defaultdict(list)
    for row in train_rows:
        by_n[row["N"]].append(row["delta_N"])
    return {
        n_workers: sum(values) / len(values)
        for n_workers, values in by_n.items()
    }


def fit_delta_model(
    train_rows: list[dict[str, Any]], model_type: str
) -> tuple[Any | None, str | None]:
    if not train_rows:
        return None, "no training rows available"

    if model_type == "lookup":
        return fit_lookup_model(train_rows), None

    degree_by_type = {"linear": 1, "quadratic": 2}
    if model_type not in degree_by_type:
        return None, f"unsupported model type: {model_type}"

    points = [(float(row["N"]), float(row["delta_N"])) for row in train_rows]
    degree = degree_by_type[model_type]
    try:
        return fit_polynomial(points, degree), None
    except OverheadModelError as exc:
        return None, str(exc)


def predict_delta(model: Any, model_type: str, n_workers: int) -> tuple[float | None, str | None]:
    if model_type == "lookup":
        if n_workers not in model:
            return None, f"lookup model has no training delta for N={n_workers}"
        return conservative_delta(model[n_workers]), None

    fitted_delta = predict_polynomial(model, float(n_workers))
    return conservative_delta(fitted_delta), None


def make_output_row(
    *,
    validation_mode: str,
    experiment_name: str,
    environment: str,
    train_difficulty: str,
    test_difficulty: str,
    train_k: str,
    test_k: str,
    test_row: dict[str, Any],
    model_type: str,
    predicted_delta: float,
    lambda_hat: float,
    oracle_n: int,
    oracle_t: float,
    notes: str,
) -> dict[str, str]:
    k_value = test_row["k"]
    n_workers = test_row["N"]
    predicted_t = k_value / (n_workers * lambda_hat) + predicted_delta
    observed_t = test_row["mean_T_real"]
    observed_delta = test_row["delta_N"]
    delta_error = predicted_delta - observed_delta
    regret_seconds = observed_t - oracle_t
    regret_percent = (regret_seconds / oracle_t) * 100.0 if oracle_t > 0 else None

    return {
        "validation_mode": validation_mode,
        "experiment_name": experiment_name,
        "environment": environment,
        "train_difficulty": train_difficulty,
        "test_difficulty": test_difficulty,
        "train_k": train_k,
        "test_k": test_k,
        "heldout_N": str(n_workers),
        "model_type": model_type,
        "predicted_delta": format_float(predicted_delta),
        "observed_delta": format_float(observed_delta),
        "delta_error": format_float(delta_error),
        "predicted_T": format_float(predicted_t),
        "observed_T": format_float(observed_t),
        "oracle_N": str(oracle_n),
        "oracle_T": format_float(oracle_t),
        "regret_seconds": format_float(regret_seconds),
        "regret_percent": format_float(regret_percent),
        "notes": notes,
    }


def evaluate_test_rows(
    *,
    validation_mode: str,
    experiment_name: str,
    environment: str,
    train_difficulty: str,
    test_difficulty: str,
    train_k: str,
    test_k: str,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]] | None,
    model_type: str,
    notes: str,
) -> list[dict[str, str]]:
    context = (
        f"mode={validation_mode}, experiment={experiment_name}, "
        f"environment={environment}, difficulty={test_difficulty}, k={test_k}"
    )
    lambda_hat = target_lambda_for_workload(test_rows, context)
    if lambda_hat is None:
        return []

    model, skip_reason = fit_delta_model(train_rows, model_type)
    if model is None:
        print(
            "[heldout_validation] WARNING: skipping "
            f"{context}; model fit failed: {skip_reason}",
            file=sys.stderr,
        )
        return []

    oracle_n, oracle_t = oracle_for_rows(test_rows)
    output_rows: list[dict[str, str]] = []
    rows_to_evaluate = evaluation_rows if evaluation_rows is not None else test_rows
    for test_row in sorted(rows_to_evaluate, key=lambda row: row["N"]):
        predicted_delta, prediction_note = predict_delta(
            model, model_type, test_row["N"]
        )
        if predicted_delta is None:
            print(
                "[heldout_validation] WARNING: skipping "
                f"{context}, N={test_row['N']}: {prediction_note}",
                file=sys.stderr,
            )
            continue

        row_notes = notes
        if test_row["N"] == 1 and validation_mode == "leave_one_N_out":
            row_notes = (
                f"{row_notes}; N=1 uses target N=1 calibration for lambda, "
                "while delta model fit excludes held-out N=1"
            )

        output_rows.append(
            make_output_row(
                validation_mode=validation_mode,
                experiment_name=experiment_name,
                environment=environment,
                train_difficulty=train_difficulty,
                test_difficulty=test_difficulty,
                train_k=train_k,
                test_k=test_k,
                test_row=test_row,
                model_type=model_type,
                predicted_delta=predicted_delta,
                lambda_hat=lambda_hat,
                oracle_n=oracle_n,
                oracle_t=oracle_t,
                notes=row_notes,
            )
        )
    return output_rows


def run_difficulty_holdout(
    rows: list[dict[str, Any]], model_type: str
) -> list[dict[str, str]]:
    grouped = group_by_workload(rows)
    output_rows: list[dict[str, str]] = []
    found_test = False

    exp_env_k_values = sorted(
        {
            (experiment_name, environment, k)
            for experiment_name, environment, _difficulty, k in grouped
        }
    )

    for experiment_name, environment, k in exp_env_k_values:
        for test_difficulty in DIFFICULTY_TEST:
            test_key = (experiment_name, environment, test_difficulty, k)
            test_rows = grouped.get(test_key, [])
            if not test_rows:
                continue
            found_test = True

            train_rows: list[dict[str, Any]] = []
            train_difficulties: set[int] = set()
            for train_difficulty in DIFFICULTY_TRAIN:
                train_key = (experiment_name, environment, train_difficulty, k)
                selected_rows = grouped.get(train_key, [])
                if selected_rows:
                    train_difficulties.add(train_difficulty)
                    train_rows.extend(selected_rows)

            if not train_rows:
                print(
                    "[heldout_validation] WARNING: difficulty_holdout cannot train "
                    f"for experiment={experiment_name}, environment={environment}, "
                    f"k={k}; no difficulty 4/5 rows available",
                    file=sys.stderr,
                )
                continue

            output_rows.extend(
                evaluate_test_rows(
                    validation_mode="difficulty_holdout",
                    experiment_name=experiment_name,
                    environment=environment,
                    train_difficulty=unique_join(train_difficulties),
                    test_difficulty=str(test_difficulty),
                    train_k=str(k),
                    test_k=str(k),
                    train_rows=train_rows,
                    test_rows=test_rows,
                    evaluation_rows=None,
                    model_type=model_type,
                    notes=(
                        "trained on available difficulty 4/5 rows with same k; "
                        "evaluated on held-out difficulty 6"
                    ),
                )
            )

    if not found_test:
        print(
            "[heldout_validation] WARNING: difficulty_holdout skipped; "
            "difficulty=6 data is not available yet",
            file=sys.stderr,
        )

    return output_rows


def run_k_holdout(rows: list[dict[str, Any]], model_type: str) -> list[dict[str, str]]:
    grouped = group_by_workload(rows)
    output_rows: list[dict[str, str]] = []
    found_test = False

    exp_env_difficulty_values = sorted(
        {
            (experiment_name, environment, difficulty)
            for experiment_name, environment, difficulty, _k in grouped
        }
    )

    for experiment_name, environment, difficulty in exp_env_difficulty_values:
        for test_k in K_TEST:
            test_key = (experiment_name, environment, difficulty, test_k)
            test_rows = grouped.get(test_key, [])
            if not test_rows:
                continue
            found_test = True

            train_rows: list[dict[str, Any]] = []
            train_k_values: set[int] = set()
            for train_k in K_TRAIN:
                train_key = (experiment_name, environment, difficulty, train_k)
                selected_rows = grouped.get(train_key, [])
                if selected_rows:
                    train_k_values.add(train_k)
                    train_rows.extend(selected_rows)

            if not train_rows:
                print(
                    "[heldout_validation] WARNING: k_holdout cannot train "
                    f"for experiment={experiment_name}, environment={environment}, "
                    f"difficulty={difficulty}; no k=1/3/10 rows available",
                    file=sys.stderr,
                )
                continue

            output_rows.extend(
                evaluate_test_rows(
                    validation_mode="k_holdout",
                    experiment_name=experiment_name,
                    environment=environment,
                    train_difficulty=str(difficulty),
                    test_difficulty=str(difficulty),
                    train_k=unique_join(train_k_values),
                    test_k=str(test_k),
                    train_rows=train_rows,
                    test_rows=test_rows,
                    evaluation_rows=None,
                    model_type=model_type,
                    notes=(
                        "trained on available k=1/3/10 rows with same difficulty; "
                        "evaluated on held-out k=20/40"
                    ),
                )
            )

    if not found_test:
        print(
            "[heldout_validation] WARNING: k_holdout skipped; "
            "k=20/40 data is not available yet",
            file=sys.stderr,
        )

    return output_rows


def run_leave_one_n_out(
    rows: list[dict[str, Any]], model_type: str
) -> list[dict[str, str]]:
    grouped = group_by_workload(rows)
    output_rows: list[dict[str, str]] = []

    for (experiment_name, environment, difficulty, k), workload_rows in sorted(
        grouped.items()
    ):
        if len({row["N"] for row in workload_rows}) < 2:
            print(
                "[heldout_validation] WARNING: leave_one_N_out skipped for "
                f"experiment={experiment_name}, environment={environment}, "
                f"difficulty={difficulty}, k={k}; need at least two N values",
                file=sys.stderr,
            )
            continue

        for test_row in sorted(workload_rows, key=lambda row: row["N"]):
            heldout_n = test_row["N"]
            train_rows = [row for row in workload_rows if row["N"] != heldout_n]
            output_rows.extend(
                evaluate_test_rows(
                    validation_mode="leave_one_N_out",
                    experiment_name=experiment_name,
                    environment=environment,
                    train_difficulty=str(difficulty),
                    test_difficulty=str(difficulty),
                    train_k=str(k),
                    test_k=str(k),
                    train_rows=train_rows,
                    test_rows=workload_rows,
                    evaluation_rows=[test_row],
                    model_type=model_type,
                    notes=(
                        f"fit excludes held-out N={heldout_n}; evaluation uses "
                        "observed held-out N only after prediction"
                    ),
                )
            )

    return output_rows


def write_report(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run_validations(
    summary_path: Path, out_path: Path, model_type: str
) -> list[dict[str, str]]:
    rows = load_summary_rows(summary_path)
    output_rows: list[dict[str, str]] = []
    if not rows:
        print(
            f"[heldout_validation] WARNING: no usable summary rows in {summary_path}",
            file=sys.stderr,
        )
        write_report(output_rows, out_path)
        return output_rows

    output_rows.extend(run_difficulty_holdout(rows, model_type))
    output_rows.extend(run_k_holdout(rows, model_type))
    output_rows.extend(run_leave_one_n_out(rows, model_type))
    write_report(output_rows, out_path)
    return output_rows


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_model_type(value: str) -> str:
    if value not in MODEL_TYPES:
        raise argparse.ArgumentTypeError(
            f"unsupported model type {value!r}; choose from lookup, linear, quadratic"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate empirical overhead prediction using difficulty, k, and "
            "leave-one-N held-out validation."
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
        default=Path("results/summary/heldout_validation_report.csv"),
        help="Output held-out validation report CSV.",
    )
    parser.add_argument(
        "--model-type",
        type=parse_model_type,
        default="quadratic",
        help="Overhead model type: lookup, linear, or quadratic. Default: quadratic.",
    )
    parser.add_argument(
        "--core-count-N",
        dest="core_count_n",
        type=positive_int,
        default=6,
        help=(
            "Core-count worker count reserved for downstream comparisons. "
            "Default: 6."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        rows = run_validations(args.summary, args.out, args.model_type)
    except (OSError, HeldoutValidationError) as exc:
        print(f"[heldout_validation] ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"[heldout_validation] wrote {len(rows)} row(s) to {args.out} "
        f"(model_type={args.model_type}, core_count_N={args.core_count_n})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

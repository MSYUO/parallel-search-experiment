#!/usr/bin/env python3
"""Evaluate an end-to-end online worker-allocation scenario.

The proposed selection uses only training delta_N values and the target
workload's N=1 lambda calibration. Observed test runtimes for all N are used
only after selection, for evaluation and baseline comparison.
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
except ImportError:  # pragma: no cover - supports python -m src.online_scenario
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
    "delta_N",
}

MODEL_TYPES = {"lookup", "linear", "quadratic"}

OUTPUT_FIELDNAMES = [
    "scenario_id",
    "experiment_name",
    "environment",
    "train_setting",
    "test_setting",
    "calibration_N",
    "lambda_hat",
    "selected_N",
    "predicted_T",
    "observed_T",
    "oracle_N",
    "oracle_T",
    "regret_seconds",
    "regret_percent",
    "baseline_name",
    "baseline_N",
    "baseline_observed_T",
    "baseline_regret_percent",
    "notes",
]

DEFAULT_TRAIN_DIFFICULTIES = [4, 5]
DEFAULT_TEST_DIFFICULTIES = [6]
DEFAULT_TRAIN_K = [1, 3, 10]
DEFAULT_TEST_K = [20, 40]
DEFAULT_PILOT_CANDIDATES = [1, 4, 8]


class OnlineScenarioError(ValueError):
    """Raised when online scenario input cannot be processed."""


def parse_int(row: dict[str, str], field: str, row_number: int) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise OnlineScenarioError(
            f"row {row_number}: {field} must be an integer"
        ) from exc


def parse_required_float(row: dict[str, str], field: str, row_number: int) -> float:
    raw_value = row.get(field, "")
    if raw_value == "":
        raise OnlineScenarioError(f"row {row_number}: {field} is required")
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise OnlineScenarioError(f"row {row_number}: {field} must be numeric") from exc


def parse_optional_float(
    row: dict[str, str], field: str, row_number: int
) -> float | None:
    raw_value = row.get(field, "")
    if raw_value == "":
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise OnlineScenarioError(
            f"row {row_number}: {field} must be numeric when present"
        ) from exc


def parse_int_list(raw_value: str) -> list[int]:
    values: list[int] = []
    for part in raw_value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            value = int(stripped)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid integer list value: {stripped!r}"
            ) from exc
        if value <= 0:
            raise argparse.ArgumentTypeError("list values must be positive integers")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("duplicate values are not allowed")
    return values


def load_summary_rows(summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        raise OnlineScenarioError(f"summary CSV not found: {summary_path}")

    rows: list[dict[str, Any]] = []

    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []

        missing = sorted(REQUIRED_FIELDS - set(reader.fieldnames))
        if missing:
            raise OnlineScenarioError(
                f"summary CSV is missing required field(s): {', '.join(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            experiment_name = (row.get("experiment_name") or "").strip()
            environment = (row.get("environment") or "").strip()
            if not experiment_name:
                raise OnlineScenarioError(
                    f"row {row_number}: experiment_name is required"
                )
            if not environment:
                raise OnlineScenarioError(f"row {row_number}: environment is required")

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
                    "delta_N": parse_optional_float(row, "delta_N", row_number),
                }
            )

    return rows


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def join_values(values: list[int] | set[int]) -> str:
    return ";".join(str(value) for value in sorted(values))


def exp_env_keys(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return sorted({(row["experiment_name"], row["environment"]) for row in rows})


def group_workloads(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["experiment_name"], row["environment"], row["difficulty"], row["k"])
        ].append(row)
    return grouped


def fit_lookup_model(train_rows: list[dict[str, Any]]) -> dict[int, float]:
    by_n: dict[int, list[float]] = defaultdict(list)
    for row in train_rows:
        if row["delta_N"] is not None:
            by_n[row["N"]].append(row["delta_N"])
    return {
        n_workers: sum(values) / len(values)
        for n_workers, values in by_n.items()
        if values
    }


def fit_delta_model(
    train_rows: list[dict[str, Any]], model_type: str
) -> tuple[Any | None, str | None]:
    usable_rows = [row for row in train_rows if row["delta_N"] is not None]
    if not usable_rows:
        return None, "no training rows with delta_N available"

    if model_type == "lookup":
        model = fit_lookup_model(usable_rows)
        if not model:
            return None, "lookup model has no usable training deltas"
        return model, None

    degree_by_type = {"linear": 1, "quadratic": 2}
    if model_type not in degree_by_type:
        return None, f"unsupported model type: {model_type}"

    points = [(float(row["N"]), float(row["delta_N"])) for row in usable_rows]
    try:
        return fit_polynomial(points, degree_by_type[model_type]), None
    except OverheadModelError as exc:
        return None, str(exc)


def predict_delta(model: Any, model_type: str, n_workers: int) -> tuple[float | None, str | None]:
    if model_type == "lookup":
        if n_workers not in model:
            return None, f"lookup model has no training delta for N={n_workers}"
        return conservative_delta(model[n_workers]), None

    fitted_delta = predict_polynomial(model, float(n_workers))
    return conservative_delta(fitted_delta), None


def target_lambda(test_rows: list[dict[str, Any]]) -> float | None:
    n1_rows = [row for row in test_rows if row["N"] == 1 and row["lambda_hat"]]
    if not n1_rows:
        return None
    return n1_rows[0]["lambda_hat"]


def oracle_for_rows(test_rows: list[dict[str, Any]]) -> tuple[int, float]:
    oracle_row = min(test_rows, key=lambda row: row["mean_T_real"])
    return oracle_row["N"], oracle_row["mean_T_real"]


def choose_smallest_within_percent(
    predictions: dict[int, float], within_percent: float
) -> tuple[int, float]:
    min_predicted_t = min(predictions.values())
    threshold = min_predicted_t * (1.0 + within_percent / 100.0)
    eligible = [
        n_workers
        for n_workers, predicted_t in predictions.items()
        if predicted_t <= threshold
    ]
    selected_n = min(eligible)
    return selected_n, predictions[selected_n]


def observed_t_for_n(test_by_n: dict[int, dict[str, Any]], n_workers: int) -> float | None:
    row = test_by_n.get(n_workers)
    if row is None:
        return None
    return row["mean_T_real"]


def regret_percent(observed_t: float | None, oracle_t: float) -> float | None:
    if observed_t is None or oracle_t <= 0:
        return None
    return ((observed_t - oracle_t) / oracle_t) * 100.0


def regret_seconds(observed_t: float | None, oracle_t: float) -> float | None:
    if observed_t is None:
        return None
    return observed_t - oracle_t


def training_mean_by_n(train_rows: list[dict[str, Any]]) -> dict[int, float]:
    by_n: dict[int, list[float]] = defaultdict(list)
    for row in train_rows:
        by_n[row["N"]].append(row["mean_T_real"])
    return {n_workers: sum(values) / len(values) for n_workers, values in by_n.items()}


def select_saturation_stop(
    train_rows: list[dict[str, Any]], saturation_threshold: float
) -> tuple[int | None, str]:
    mean_by_n = training_mean_by_n(train_rows)
    if 1 not in mean_by_n:
        return None, "missing N=1 in training data for saturation-stop"
    if len(mean_by_n) < 2:
        return None, "need at least two training N values for saturation-stop"

    ns = sorted(mean_by_n)
    speedups = {
        n_workers: mean_by_n[1] / mean_by_n[n_workers]
        for n_workers in ns
        if mean_by_n[n_workers] > 0
    }

    previous_n = ns[0]
    previous_speedup = speedups[previous_n]
    for n_workers in ns[1:]:
        current_speedup = speedups[n_workers]
        improvement_percent = (
            ((current_speedup - previous_speedup) / previous_speedup) * 100.0
            if previous_speedup > 0
            else 0.0
        )
        if improvement_percent < saturation_threshold:
            return (
                n_workers,
                "selected first training N where marginal speedup improvement "
                f"({improvement_percent:.2f}%) is below {saturation_threshold:.2f}%",
            )
        previous_n = n_workers
        previous_speedup = current_speedup

    return ns[-1], "no saturation point found in training data; selected largest training N"


def select_pilot_sweep(
    test_by_n: dict[int, dict[str, Any]], pilot_candidates: list[int]
) -> tuple[int | None, str]:
    available = [
        n_workers for n_workers in pilot_candidates if n_workers in test_by_n
    ]
    if not available:
        return None, "none of the pilot candidates are available for test workload"

    selected_n = min(available, key=lambda n_workers: test_by_n[n_workers]["mean_T_real"])
    return (
        selected_n,
        "uses only observed pilot candidates, not the full test worker-count sweep",
    )


def baseline_specs(
    *,
    test_by_n: dict[int, dict[str, Any]],
    train_rows: list[dict[str, Any]],
    core_count_n: int,
    thread_count_n: int,
    saturation_threshold: float,
    pilot_candidates: list[int],
) -> list[tuple[str, int | None, str]]:
    baselines: list[tuple[str, int | None, str]] = []

    baselines.append(("Static-8", 8, "static N=8 baseline"))

    if test_by_n:
        baselines.append(("Max-worker", max(test_by_n), "largest available test N"))
    else:
        baselines.append(("Max-worker", None, "no test N values available"))

    baselines.append(("Core-count", core_count_n, "physical-core heuristic baseline"))
    baselines.append(
        ("Thread-count", thread_count_n, "logical-thread heuristic baseline")
    )

    saturation_n, saturation_note = select_saturation_stop(
        train_rows, saturation_threshold
    )
    baselines.append(("Saturation-stop", saturation_n, saturation_note))

    pilot_n, pilot_note = select_pilot_sweep(test_by_n, pilot_candidates)
    baselines.append(("Pilot-sweep", pilot_n, pilot_note))

    if test_by_n:
        oracle_n, _oracle_t = oracle_for_rows(list(test_by_n.values()))
        baselines.append(("Oracle", oracle_n, "offline upper bound; evaluation only"))
    else:
        baselines.append(("Oracle", None, "no test rows available"))

    return baselines


def make_report_row(
    *,
    scenario_id: str,
    experiment_name: str,
    environment: str,
    train_setting: str,
    test_setting: str,
    lambda_hat: float,
    selected_n: int,
    selected_predicted_t: float,
    selected_observed_t: float | None,
    oracle_n: int,
    oracle_t: float,
    baseline_name: str,
    baseline_n: int | None,
    baseline_observed_t: float | None,
    baseline_note: str,
    scenario_note: str,
) -> dict[str, str]:
    selected_regret_seconds = regret_seconds(selected_observed_t, oracle_t)
    selected_regret_percent = regret_percent(selected_observed_t, oracle_t)
    baseline_regret = regret_percent(baseline_observed_t, oracle_t)

    notes = scenario_note
    if baseline_n is None:
        notes = f"{notes}; {baseline_name} missing: {baseline_note}"
    elif baseline_observed_t is None:
        notes = (
            f"{notes}; {baseline_name} selected N={baseline_n}, "
            "but observed test runtime is missing"
        )
    else:
        notes = f"{notes}; {baseline_name}: {baseline_note}"

    return {
        "scenario_id": scenario_id,
        "experiment_name": experiment_name,
        "environment": environment,
        "train_setting": train_setting,
        "test_setting": test_setting,
        "calibration_N": "1",
        "lambda_hat": format_float(lambda_hat),
        "selected_N": str(selected_n),
        "predicted_T": format_float(selected_predicted_t),
        "observed_T": format_float(selected_observed_t),
        "oracle_N": str(oracle_n),
        "oracle_T": format_float(oracle_t),
        "regret_seconds": format_float(selected_regret_seconds),
        "regret_percent": format_float(selected_regret_percent),
        "baseline_name": baseline_name,
        "baseline_N": "" if baseline_n is None else str(baseline_n),
        "baseline_observed_T": format_float(baseline_observed_t),
        "baseline_regret_percent": format_float(baseline_regret),
        "notes": notes,
    }


def infer_candidate_ns(
    rows: list[dict[str, Any]], explicit_candidate_ns: list[int] | None
) -> list[int]:
    if explicit_candidate_ns is not None:
        return sorted(explicit_candidate_ns)
    return sorted({row["N"] for row in rows})


def build_online_report(
    *,
    rows: list[dict[str, Any]],
    model_type: str,
    core_count_n: int,
    thread_count_n: int,
    within_percent: float,
    saturation_threshold: float,
    pilot_candidates: list[int],
    train_difficulties: list[int],
    test_difficulties: list[int],
    train_k_values: list[int],
    test_k_values: list[int],
    explicit_candidate_ns: list[int] | None,
) -> list[dict[str, str]]:
    grouped = group_workloads(rows)
    report_rows: list[dict[str, str]] = []
    scenario_counter = 0

    for experiment_name, environment in exp_env_keys(rows):
        exp_env_rows = [
            row
            for row in rows
            if row["experiment_name"] == experiment_name
            and row["environment"] == environment
        ]
        train_rows = [
            row
            for row in exp_env_rows
            if row["difficulty"] in train_difficulties
            and row["k"] in train_k_values
            and row["delta_N"] is not None
        ]
        if not train_rows:
            print(
                "[online_scenario] WARNING: no training rows available for "
                f"experiment={experiment_name}, environment={environment}; "
                "skipping online scenarios",
                file=sys.stderr,
            )
            continue

        model, fit_note = fit_delta_model(train_rows, model_type)
        if model is None:
            print(
                "[online_scenario] WARNING: model fit failed for "
                f"experiment={experiment_name}, environment={environment}: {fit_note}",
                file=sys.stderr,
            )
            continue

        candidate_ns = infer_candidate_ns(exp_env_rows, explicit_candidate_ns)
        if not candidate_ns:
            print(
                "[online_scenario] WARNING: no candidate N values available for "
                f"experiment={experiment_name}, environment={environment}",
                file=sys.stderr,
            )
            continue

        found_test_workload = False
        for test_difficulty in test_difficulties:
            for test_k in test_k_values:
                test_key = (experiment_name, environment, test_difficulty, test_k)
                test_rows = grouped.get(test_key, [])
                if not test_rows:
                    continue
                found_test_workload = True

                lambda_hat = target_lambda(test_rows)
                if lambda_hat is None:
                    print(
                        "[online_scenario] WARNING: missing N=1 lambda calibration "
                        f"for experiment={experiment_name}, environment={environment}, "
                        f"difficulty={test_difficulty}, k={test_k}; skipping scenario",
                        file=sys.stderr,
                    )
                    continue

                predictions: dict[int, float] = {}
                prediction_skips: list[str] = []
                for n_workers in candidate_ns:
                    predicted_delta, prediction_note = predict_delta(
                        model, model_type, n_workers
                    )
                    if predicted_delta is None:
                        prediction_skips.append(prediction_note or f"N={n_workers}")
                        continue
                    predictions[n_workers] = (
                        test_k / (n_workers * lambda_hat) + predicted_delta
                    )

                if not predictions:
                    print(
                        "[online_scenario] WARNING: no candidate predictions "
                        f"available for experiment={experiment_name}, "
                        f"environment={environment}, difficulty={test_difficulty}, "
                        f"k={test_k}; skipping scenario",
                        file=sys.stderr,
                    )
                    continue

                selected_n, selected_predicted_t = choose_smallest_within_percent(
                    predictions, within_percent
                )
                test_by_n = {row["N"]: row for row in test_rows}
                selected_observed_t = observed_t_for_n(test_by_n, selected_n)
                oracle_n, oracle_t = oracle_for_rows(test_rows)

                scenario_counter += 1
                scenario_id = f"online_{scenario_counter:03d}"
                train_setting = (
                    f"difficulty={join_values(train_difficulties)},"
                    f"k={join_values(train_k_values)},model_type={model_type}"
                )
                test_setting = (
                    f"difficulty={test_difficulty},k={test_k},"
                    f"candidate_N={join_values(candidate_ns)}"
                )
                scenario_note = (
                    "Proposed selection uses training delta model and target N=1 "
                    f"lambda calibration; selected smallest N within "
                    f"{within_percent:.2f}% of minimum predicted time"
                )
                if prediction_skips:
                    scenario_note = (
                        f"{scenario_note}; skipped candidate prediction(s): "
                        f"{' | '.join(prediction_skips)}"
                    )
                if selected_observed_t is None:
                    scenario_note = (
                        f"{scenario_note}; selected_N has no observed test runtime"
                    )

                for baseline_name, baseline_n, baseline_note in baseline_specs(
                    test_by_n=test_by_n,
                    train_rows=train_rows,
                    core_count_n=core_count_n,
                    thread_count_n=thread_count_n,
                    saturation_threshold=saturation_threshold,
                    pilot_candidates=pilot_candidates,
                ):
                    baseline_observed_t = (
                        None
                        if baseline_n is None
                        else observed_t_for_n(test_by_n, baseline_n)
                    )
                    report_rows.append(
                        make_report_row(
                            scenario_id=scenario_id,
                            experiment_name=experiment_name,
                            environment=environment,
                            train_setting=train_setting,
                            test_setting=test_setting,
                            lambda_hat=lambda_hat,
                            selected_n=selected_n,
                            selected_predicted_t=selected_predicted_t,
                            selected_observed_t=selected_observed_t,
                            oracle_n=oracle_n,
                            oracle_t=oracle_t,
                            baseline_name=baseline_name,
                            baseline_n=baseline_n,
                            baseline_observed_t=baseline_observed_t,
                            baseline_note=baseline_note,
                            scenario_note=scenario_note,
                        )
                    )

        if not found_test_workload:
            print(
                "[online_scenario] WARNING: held-out online test workloads are "
                f"not available for experiment={experiment_name}, "
                f"environment={environment}; expected difficulty="
                f"{join_values(test_difficulties)}, k={join_values(test_k_values)}",
                file=sys.stderr,
            )

    return report_rows


def write_report(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_model_type(value: str) -> str:
    if value not in MODEL_TYPES:
        raise argparse.ArgumentTypeError(
            f"unsupported model type {value!r}; choose from lookup, linear, quadratic"
        )
    return value


def nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be numeric") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


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
            "Evaluate an end-to-end online scenario using training overhead "
            "models and target N=1 calibration."
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
        default=Path("results/summary/online_scenario_report.csv"),
        help="Output online scenario report CSV.",
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
        help="Core-count baseline worker count. Default: 6.",
    )
    parser.add_argument(
        "--thread-count-N",
        dest="thread_count_n",
        type=positive_int,
        default=12,
        help="Thread-count baseline worker count. Default: 12.",
    )
    parser.add_argument(
        "--within-percent",
        type=nonnegative_float,
        default=5.0,
        help="Select the smallest N within this percent of min predicted time.",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=nonnegative_float,
        default=10.0,
        help="Marginal speedup-improvement threshold for Saturation-stop.",
    )
    parser.add_argument(
        "--pilot-candidates",
        type=parse_int_list,
        default=DEFAULT_PILOT_CANDIDATES,
        help="Comma-separated Pilot-sweep candidate N values. Default: 1,4,8.",
    )
    parser.add_argument(
        "--candidate-N",
        dest="candidate_ns",
        type=parse_int_list,
        default=None,
        help="Optional comma-separated candidate N values for Proposed selection.",
    )
    parser.add_argument(
        "--train-difficulties",
        type=parse_int_list,
        default=DEFAULT_TRAIN_DIFFICULTIES,
        help="Comma-separated training difficulty values. Default: 4,5.",
    )
    parser.add_argument(
        "--test-difficulties",
        type=parse_int_list,
        default=DEFAULT_TEST_DIFFICULTIES,
        help="Comma-separated held-out test difficulty values. Default: 6.",
    )
    parser.add_argument(
        "--train-k",
        dest="train_k_values",
        type=parse_int_list,
        default=DEFAULT_TRAIN_K,
        help="Comma-separated training k values. Default: 1,3,10.",
    )
    parser.add_argument(
        "--test-k",
        dest="test_k_values",
        type=parse_int_list,
        default=DEFAULT_TEST_K,
        help="Comma-separated held-out test k values. Default: 20,40.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        rows = load_summary_rows(args.summary)
        if not rows:
            print(
                f"[online_scenario] WARNING: no summary rows in {args.summary}",
                file=sys.stderr,
            )
            report_rows: list[dict[str, str]] = []
        else:
            report_rows = build_online_report(
                rows=rows,
                model_type=args.model_type,
                core_count_n=args.core_count_n,
                thread_count_n=args.thread_count_n,
                within_percent=args.within_percent,
                saturation_threshold=args.saturation_threshold,
                pilot_candidates=args.pilot_candidates,
                train_difficulties=args.train_difficulties,
                test_difficulties=args.test_difficulties,
                train_k_values=args.train_k_values,
                test_k_values=args.test_k_values,
                explicit_candidate_ns=args.candidate_ns,
            )
        write_report(report_rows, args.out)
    except (OSError, OnlineScenarioError) as exc:
        print(f"[online_scenario] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[online_scenario] wrote {len(report_rows)} row(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate and summarize a JIPS experiment config.

This utility is inspection-only. It does not launch Docker, Kubernetes, kind, or
cloud experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "experiment_name",
    "environment",
    "difficulty_values",
    "k_values",
    "worker_counts",
    "trials",
    "output_root",
    "notes",
    "lambda_calibration_required",
}

LARGE_GRID_TRIAL_THRESHOLD = 500


class ConfigError(ValueError):
    """Raised when a config is malformed."""


def _require_string(config: dict[str, Any], key: str) -> None:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string")


def _require_positive_int(config: dict[str, Any], key: str) -> None:
    value = config.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{key} must be a positive integer")


def _require_int_list(config: dict[str, Any], key: str) -> None:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{key} must be a non-empty list")
    if any(not isinstance(item, int) or item <= 0 for item in value):
        raise ConfigError(f"{key} must contain only positive integers")
    if len(set(value)) != len(value):
        raise ConfigError(f"{key} must not contain duplicate values")


def _validate_optional_split(config: dict[str, Any], key: str, parent_key: str) -> None:
    if key not in config:
        return
    _require_int_list(config, key)
    parent_values = set(config[parent_key])
    split_values = set(config[key])
    if not split_values.issubset(parent_values):
        raise ConfigError(f"{key} must be a subset of {parent_key}")


def load_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError("config root must be a JSON object")
    return config


def validate_config(config: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_FIELDS - set(config))
    if missing:
        raise ConfigError(f"missing required field(s): {', '.join(missing)}")

    for key in ("experiment_name", "environment", "output_root", "notes"):
        _require_string(config, key)

    for key in ("difficulty_values", "k_values", "worker_counts"):
        _require_int_list(config, key)

    _require_positive_int(config, "trials")

    if config["lambda_calibration_required"] is not True:
        raise ConfigError("lambda_calibration_required must be true")

    if config["lambda_calibration_required"] and 1 not in config["worker_counts"]:
        raise ConfigError("worker_counts must include N=1 for lambda calibration")

    for key, parent_key in (
        ("train_difficulty_values", "difficulty_values"),
        ("test_difficulty_values", "difficulty_values"),
        ("train_k_values", "k_values"),
        ("test_k_values", "k_values"),
    ):
        _validate_optional_split(config, key, parent_key)

    if "calibration_N" in config:
        calibration_n = config["calibration_N"]
        if not isinstance(calibration_n, int) or calibration_n <= 0:
            raise ConfigError("calibration_N must be a positive integer")
        if calibration_n not in config["worker_counts"]:
            raise ConfigError("calibration_N must be present in worker_counts")


def count_expanded_conditions(config: dict[str, Any]) -> tuple[int, str]:
    worker_count_count = len(config["worker_counts"])
    split_keys = {
        "train_difficulty_values",
        "test_difficulty_values",
        "train_k_values",
        "test_k_values",
    }

    if split_keys.issubset(config):
        train_conditions = (
            len(config["train_difficulty_values"])
            * len(config["train_k_values"])
            * worker_count_count
        )
        test_conditions = (
            len(config["test_difficulty_values"])
            * len(config["test_k_values"])
            * worker_count_count
        )
        return train_conditions + test_conditions, "train/test split"

    conditions = (
        len(config["difficulty_values"])
        * len(config["k_values"])
        * worker_count_count
    )
    return conditions, "full grid"


def print_summary(path: Path, config: dict[str, Any]) -> None:
    expanded_conditions, count_mode = count_expanded_conditions(config)
    total_trial_runs = expanded_conditions * config["trials"]
    includes_n1 = 1 in config["worker_counts"]

    print(f"config_path: {path}")
    print(f"experiment_name: {config['experiment_name']}")
    print(f"environment: {config['environment']}")
    print(f"expanded_conditions: {expanded_conditions} ({count_mode})")
    print(f"total_expected_trial_runs: {total_trial_runs}")
    print(f"n1_included_for_lambda_calibration: {'yes' if includes_n1 else 'no'}")

    if total_trial_runs > LARGE_GRID_TRIAL_THRESHOLD:
        print(
            "warning: grid is large "
            f"({total_trial_runs} trial runs > {LARGE_GRID_TRIAL_THRESHOLD})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a JIPS experiment config."
    )
    parser.add_argument("config", type=Path, help="Path to a JSON config file")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        validate_config(config)
    except OSError as exc:
        print(f"[ERROR] could not read {args.config}: {exc}", file=sys.stderr)
        return 1
    except ConfigError as exc:
        print(f"[ERROR] invalid config {args.config}: {exc}", file=sys.stderr)
        return 1

    print_summary(args.config, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run JIPS v5 experiment grids from JSON configs.

This wrapper preserves the existing worker and Docker Compose pipeline. It
executes one expanded trial job at a time, archives raw worker JSON files into a
structured directory, and appends a v2 merged trial row.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from merge_results import MergeError, append_csv, merge_trial
except ImportError:  # pragma: no cover - supports python -m src.run_config_experiments
    from src.merge_results import MergeError, append_csv, merge_trial


REQUIRED_CONFIG_FIELDS = {
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

MERGED_KEY_FIELDS = ["experiment_name", "environment", "difficulty", "k", "N", "trial"]


class ConfigRunnerError(ValueError):
    """Raised when config-driven execution cannot proceed safely."""


@dataclass(frozen=True)
class TrialJob:
    experiment_name: str
    environment: str
    difficulty: int
    k: int
    n_workers: int
    trial: int

    @property
    def key(self) -> tuple[str, str, int, int, int, int]:
        return (
            self.experiment_name,
            self.environment,
            self.difficulty,
            self.k,
            self.n_workers,
            self.trial,
        )


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise ConfigRunnerError(f"config file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigRunnerError(f"invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigRunnerError("config root must be a JSON object")
    validate_config(config)
    return config


def require_string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigRunnerError(f"{key} must be a non-empty string")
    return value.strip()


def require_positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ConfigRunnerError(f"{key} must be a positive integer")
    return value


def require_int_list(config: dict[str, Any], key: str) -> list[int]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigRunnerError(f"{key} must be a non-empty list")
    if any(not isinstance(item, int) or item <= 0 for item in value):
        raise ConfigRunnerError(f"{key} must contain only positive integers")
    if len(set(value)) != len(value):
        raise ConfigRunnerError(f"{key} must not contain duplicate values")
    return value


def validate_config(config: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_CONFIG_FIELDS - set(config))
    if missing:
        raise ConfigRunnerError(f"config missing required field(s): {', '.join(missing)}")

    environment = require_string(config, "environment")
    if environment != "docker-compose":
        raise ConfigRunnerError(
            "run_config_experiments.py currently supports only "
            f"environment='docker-compose', got {environment!r}"
        )

    require_string(config, "experiment_name")
    require_string(config, "output_root")
    require_string(config, "notes")
    require_positive_int(config, "trials")

    for key in ("difficulty_values", "k_values", "worker_counts"):
        require_int_list(config, key)

    if config["lambda_calibration_required"] is not True:
        raise ConfigRunnerError("lambda_calibration_required must be true")
    if 1 not in config["worker_counts"]:
        raise ConfigRunnerError("worker_counts must include N=1 for lambda calibration")


def expand_jobs(config: dict[str, Any]) -> list[TrialJob]:
    experiment_name = require_string(config, "experiment_name")
    environment = require_string(config, "environment")
    difficulties = require_int_list(config, "difficulty_values")
    k_values = require_int_list(config, "k_values")
    worker_counts = require_int_list(config, "worker_counts")
    trials = require_positive_int(config, "trials")

    jobs: list[TrialJob] = []
    for difficulty in difficulties:
        for k in k_values:
            for n_workers in worker_counts:
                for trial in range(1, trials + 1):
                    jobs.append(
                        TrialJob(
                            experiment_name=experiment_name,
                            environment=environment,
                            difficulty=difficulty,
                            k=k,
                            n_workers=n_workers,
                            trial=trial,
                        )
                    )
    return jobs


def read_completed_keys(merged_csv: Path) -> set[tuple[str, str, int, int, int, int]]:
    if not merged_csv.exists():
        return set()

    completed: set[tuple[str, str, int, int, int, int]] = set()
    with merged_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return completed
        missing = sorted(set(MERGED_KEY_FIELDS) - set(reader.fieldnames))
        if missing:
            print(
                "[run_config] WARNING: existing merged CSV is missing resume key "
                f"field(s): {', '.join(missing)}; resume detection disabled",
                file=sys.stderr,
            )
            return set()

        for row_number, row in enumerate(reader, start=2):
            try:
                completed.add(
                    (
                        (row.get("experiment_name") or "").strip(),
                        (row.get("environment") or "").strip(),
                        int(row["difficulty"]),
                        int(row["k"]),
                        int(row["N"]),
                        int(row["trial"]),
                    )
                )
            except (TypeError, ValueError):
                print(
                    "[run_config] WARNING: ignoring malformed resume key at "
                    f"{merged_csv}:{row_number}",
                    file=sys.stderr,
                )
    return completed


def raw_trial_dir(raw_root: Path, job: TrialJob) -> Path:
    return (
        raw_root
        / job.experiment_name
        / f"difficulty_{job.difficulty}"
        / f"k_{job.k}"
        / f"N_{job.n_workers}"
        / f"trial_{job.trial}"
    )


def root_results_dir() -> Path:
    # The current docker-compose.yml mounts ./results to /results in the worker.
    return Path("results")


def worker_json_files(results_dir: Path, trial: int) -> list[Path]:
    return sorted(results_dir.glob(f"result_w*_t{trial}.json"))


def clean_root_trial_files(results_dir: Path, trial: int) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for path in worker_json_files(results_dir, trial):
        path.unlink()
    for path in results_dir.glob(".claimed_id_*"):
        path.unlink()


def archive_worker_jsons(job: TrialJob, raw_dir: Path) -> list[Path]:
    results_dir = root_results_dir()
    files = worker_json_files(results_dir, job.trial)
    if len(files) != job.n_workers:
        raise ConfigRunnerError(
            f"expected {job.n_workers} worker JSON file(s) for {describe_job(job)}, "
            f"but found {len(files)} in {results_dir}"
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    for stale in raw_dir.glob(f"result_w*_t{job.trial}.json"):
        stale.unlink()

    archived: list[Path] = []
    for path in files:
        destination = raw_dir / path.name
        shutil.copy2(path, destination)
        archived.append(destination)
    return archived


def docker_compose_up_command(
    compose_file: Path, job: TrialJob, *, include_build: bool
) -> list[str]:
    command = ["docker", "compose", "-f", str(compose_file), "up"]
    if include_build:
        command.append("--build")
    command.extend(["--scale", f"worker={job.n_workers}"])
    return command


def docker_compose_down_command(compose_file: Path) -> list[str]:
    return ["docker", "compose", "-f", str(compose_file), "down", "--remove-orphans"]


def job_environment(job: TrialJob) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "DIFFICULTY": str(job.difficulty),
            "K_TARGET": str(job.k),
            "TRIAL_NUMBER": str(job.trial),
            "N_WORKERS": str(job.n_workers),
            "RESULTS_DIR": "/results",
        }
    )
    return env


def describe_job(job: TrialJob) -> str:
    return (
        f"experiment={job.experiment_name}, environment={job.environment}, "
        f"difficulty={job.difficulty}, k={job.k}, N={job.n_workers}, "
        f"trial={job.trial}"
    )


def print_dry_run_job(
    *,
    job: TrialJob,
    compose_file: Path,
    raw_dir: Path,
    merged_csv: Path,
    include_build: bool,
    skip_reason: str | None,
) -> None:
    print(f"[dry-run] {describe_job(job)}")
    if skip_reason:
        print(f"  skip: {skip_reason}")
        return
    print(
        "  env: "
        f"DIFFICULTY={job.difficulty} K_TARGET={job.k} "
        f"TRIAL_NUMBER={job.trial} N_WORKERS={job.n_workers} RESULTS_DIR=/results"
    )
    print(f"  command: {' '.join(docker_compose_up_command(compose_file, job, include_build=include_build))}")
    print(f"  cleanup: {' '.join(docker_compose_down_command(compose_file))}")
    print(f"  archive_raw_to: {raw_dir}")
    print(f"  merge_csv: {merged_csv}")


def run_subprocess(command: list[str], env: dict[str, str] | None = None) -> bool:
    print(f"[run_config] command: {' '.join(command)}")
    completed = subprocess.run(command, env=env, check=False)
    return completed.returncode == 0


def execute_job(
    *,
    job: TrialJob,
    compose_file: Path,
    raw_root: Path,
    merged_csv: Path,
    include_build: bool,
) -> None:
    results_dir = root_results_dir()
    raw_dir = raw_trial_dir(raw_root, job)
    print(f"[run_config] starting {describe_job(job)}")

    clean_root_trial_files(results_dir, job.trial)

    compose_ok = False
    try:
        compose_ok = run_subprocess(
            docker_compose_up_command(compose_file, job, include_build=include_build),
            env=job_environment(job),
        )
    finally:
        down_ok = run_subprocess(docker_compose_down_command(compose_file))
        if not down_ok:
            print(
                "[run_config] WARNING: docker compose down failed; check Docker state",
                file=sys.stderr,
            )

    if not compose_ok:
        raise ConfigRunnerError(f"Docker Compose failed for {describe_job(job)}")

    archive_worker_jsons(job, raw_dir)
    row = merge_trial(
        experiment_name=job.experiment_name,
        environment=job.environment,
        difficulty=job.difficulty,
        k=job.k,
        n_workers=job.n_workers,
        trial=job.trial,
        results_dir=raw_dir,
        expected_n=job.n_workers,
        allow_partial=False,
    )
    append_csv(row, merged_csv)
    print(f"[run_config] completed {describe_job(job)} | T_real={row['T_real']}s")


def run_jobs(
    *,
    jobs: list[TrialJob],
    merged_csv: Path,
    raw_root: Path,
    compose_file: Path,
    dry_run: bool,
    rerun_existing: bool,
    stop_on_error: bool,
) -> int:
    completed_keys = read_completed_keys(merged_csv)
    attempted = 0
    skipped = 0
    failed = 0
    built = False

    for job in jobs:
        skip_reason = None
        if not rerun_existing and job.key in completed_keys:
            skip_reason = "merged trial row already exists"

        include_build = not built
        raw_dir = raw_trial_dir(raw_root, job)

        if dry_run:
            print_dry_run_job(
                job=job,
                compose_file=compose_file,
                raw_dir=raw_dir,
                merged_csv=merged_csv,
                include_build=include_build,
                skip_reason=skip_reason,
            )
            if skip_reason:
                skipped += 1
            else:
                attempted += 1
                built = True
            continue

        if skip_reason:
            print(f"[run_config] SKIP {describe_job(job)} ({skip_reason})")
            skipped += 1
            continue

        try:
            execute_job(
                job=job,
                compose_file=compose_file,
                raw_root=raw_root,
                merged_csv=merged_csv,
                include_build=include_build,
            )
            attempted += 1
            built = True
        except (ConfigRunnerError, MergeError, OSError) as exc:
            failed += 1
            print(f"[run_config] ERROR: {exc}", file=sys.stderr)
            if stop_on_error:
                break

    print(
        "[run_config] summary: "
        f"attempted={attempted}, skipped={skipped}, failed={failed}, "
        f"planned={len(jobs)}"
    )
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a JIPS v5 Docker Compose experiment grid from a JSON config "
            "and append v2 merged trial rows."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Config JSON path.")
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=Path("results/merged/merged_trials_v2.csv"),
        help="Output merged v2 trial CSV path.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("results/raw"),
        help="Root directory for structured raw worker JSON archives.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running Docker or writing outputs.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Run only the first N expanded trial jobs.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Rerun jobs even when an identical merged trial row already exists.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failed trial instead of continuing.",
    )
    parser.add_argument(
        "--docker-compose-file",
        type=Path,
        default=Path("docker-compose.yml"),
        help="Docker Compose file path. Default: docker-compose.yml.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        jobs = expand_jobs(config)
        total_jobs = len(jobs)
        if args.limit is not None:
            jobs = jobs[: args.limit]

        print(f"[run_config] config: {args.config}")
        print(f"[run_config] experiment_name: {config['experiment_name']}")
        print(f"[run_config] environment: {config['environment']}")
        print(f"[run_config] total expanded trial jobs: {total_jobs}")
        if args.limit is not None:
            print(f"[run_config] limit: first {len(jobs)} trial job(s)")

        if not args.dry_run and not args.docker_compose_file.exists():
            raise ConfigRunnerError(
                f"Docker Compose file not found: {args.docker_compose_file}"
            )

        return run_jobs(
            jobs=jobs,
            merged_csv=args.merged_csv,
            raw_root=args.raw_root,
            compose_file=args.docker_compose_file,
            dry_run=args.dry_run,
            rerun_existing=args.rerun_existing,
            stop_on_error=args.stop_on_error,
        )
    except ConfigRunnerError as exc:
        print(f"[run_config] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

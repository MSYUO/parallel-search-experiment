# Result Schema

This document defines the result files used by the JIPS v5 extension. The schema
keeps the existing SHA-256 worker output intact and adds a clearer merged-trial
CSV plus a grouped summary CSV for workload-specific analysis.

## Raw Worker JSON Inputs

Each worker container writes one JSON file for a trial. The existing file naming
pattern is:

```text
results/result_w{worker_id}_t{trial}.json
```

The current worker JSON fields are:

| Field | Description |
| --- | --- |
| `worker_id` | Worker identifier claimed by the worker process. |
| `trial` | Trial number passed through the environment. |
| `difficulty` | SHA-256 prefix difficulty, equal to the number of leading zero hex characters required. |
| `k_target` | Number of local successes each worker records before exiting. |
| `seed` | Worker/trial-specific seed prefix used to keep search streams independent. |
| `elapsed_seconds` | Worker-local elapsed time until that worker collected `k_target` successes. |
| `total_hashes` | Number of hashes attempted by that worker. |
| `found_times` | Worker-local timestamps for each success found by the worker. |

The v5 result pipeline does not require a change to this worker format.

## Definition of `T_real`

For a trial with `N` workers and target count `k`, all worker `found_times` are
merged into one list and sorted in ascending order.

```text
T_real = sorted(all_worker_found_times)[k - 1]
```

That is, `T_real` is the time of the global k-th success after merging all
worker success timestamps. This preserves the original KIPS experiment's
completion-time definition.

## Merged Trial CSV Schema

`src/merge_results.py` writes one row per completed trial. The default output is:

```text
results/merged/merged_trials_v2.csv
```

Required columns:

| Column | Description |
| --- | --- |
| `experiment_name` | Name of the experiment grid or scenario. |
| `environment` | Execution environment label, such as `docker-compose`. |
| `difficulty` | SHA-256 prefix difficulty. |
| `k` | Global target success count used to define `T_real`. |
| `N` | Worker count for the trial. |
| `trial` | Trial number. |
| `T_real` | Time of the global k-th success after merging and sorting worker `found_times`. |
| `total_successes_collected` | Total number of success timestamps collected across all worker files. |
| `total_worker_hashes` | Sum of `total_hashes` across worker files. |
| `mean_worker_elapsed_seconds` | Mean worker-local elapsed time across worker files. |
| `max_worker_elapsed_seconds` | Maximum worker-local elapsed time across worker files. |
| `min_worker_elapsed_seconds` | Minimum worker-local elapsed time across worker files. |
| `result_file_count` | Number of worker JSON files merged. |
| `timestamp` | Timestamp when the merged row was written. |

When an expected worker count is available, the merge step should validate that
the expected number of worker JSON files exists. Missing worker files should
cause a safe failure by default rather than silently producing misleading
results.

## Summary Statistics CSV Schema

`src/analysis_summary.py` reads merged trial rows and writes grouped summary
statistics. The default output is:

```text
results/summary/summary_statistics_v2.csv
```

Rows are grouped by:

```text
experiment_name, environment, difficulty, k, N
```

Columns:

| Column | Description |
| --- | --- |
| `experiment_name` | Experiment name from merged rows. |
| `environment` | Environment label from merged rows. |
| `difficulty` | Workload difficulty. |
| `k` | Target success count. |
| `N` | Worker count. |
| `trials_completed` | Number of merged trial rows in this condition. |
| `mean_T_real` | Mean observed `T_real` for the condition. |
| `std_T_real` | Sample standard deviation of `T_real`; `0` when only one trial is available. |
| `ci95_low` | Lower 95% confidence interval bound for mean `T_real`. |
| `ci95_high` | Upper 95% confidence interval bound for mean `T_real`. |
| `lambda_hat` | Workload-specific single-worker success-rate estimate. |
| `T_theory` | Ideal modeled completion time, `k / (N * lambda_hat)`. |
| `delta_N` | Empirical overhead, `mean_T_real - T_theory`. Observed values are not clipped. |
| `speedup` | Observed speedup, `mean_T_real(N=1) / mean_T_real(N)`. |
| `parallel_efficiency` | `speedup / N`. |
| `relative_error_percent` | `(mean_T_real - T_theory) / T_theory * 100`. |
| `oracle_N` | Worker count with the lowest mean `T_real` for the workload group. |
| `oracle_T` | Lowest mean `T_real` for the workload group. |
| `core_count_N` | Core-count baseline worker count, default `6`. |
| `core_count_T` | Mean `T_real` at `core_count_N`, if that worker count is available. |
| `core_count_regret_seconds` | `core_count_T - oracle_T`, if Core-count data is available. |
| `core_count_regret_percent` | `core_count_regret_seconds / oracle_T * 100`, if available. |
| `oracle_differs_from_core_count` | Whether `oracle_N` differs from `core_count_N`; `unassessable` when the Core-count worker count is not present. |

## Workload-Specific Lambda

The main Docker Compose analysis uses:

```text
lambda_hat(difficulty, k) = k / mean_T_real(N=1, difficulty, k)
```

In implementation, `experiment_name` and `environment` are also used as grouping
keys to avoid accidentally mixing separate experiment outputs. The interpretation
remains workload-specific: the calibration comes from the `N=1` trials for the
same `difficulty` and `k`, not from a global lambda shared across workloads.

If optional environments are collected later, environment labels can be used to
keep their measurements separate. The main Docker Compose claim should still be
described as workload-specific calibration unless a real environment comparison
is performed.

If `N=1` data is missing for a workload group, theory-based fields such as
`lambda_hat`, `T_theory`, `delta_N`, `speedup`, `parallel_efficiency`, and
`relative_error_percent` should be left empty for that group and a warning should
be printed.

Observed `delta_N` is kept as measured in the summary. In very small smoke tests
or low-trial stochastic measurements, `delta_N` can be negative because the
measured mean may fall below the ideal modeled time by chance. Prediction models
may use conservative nonnegative clipping for fitted overhead, but the observed
summary statistic should not be clipped.

## Core-Count Baseline Fields

The Core-count baseline represents the physical-core heuristic. For the current
single-host setting, the default is:

```text
core_count_N = 6
```

The summary records the observed Core-count runtime when `N=6` exists in the
merged data. It also records regret relative to Oracle for each workload group.
If `N=6` is not available, `core_count_T` and regret fields are left empty, and
`oracle_differs_from_core_count` is marked `unassessable`. The analysis should
not fabricate Core-count results for unmeasured worker counts.

## Modeling Scope

The ideal Poisson/Gamma waiting-time model is used as modeling background:

```text
T_theory(N) = k / (N * lambda_hat)
```

This model is not claimed as a new theoretical contribution. The JIPS v5
contribution is the measurement-driven characterization of container overhead,
workload-adaptive worker allocation, held-out validation, and end-to-end online
evaluation.

# JIPS v5 Extension Plan

## 1. Original KIPS Experiment Goal

The original experiment evaluates Docker-based parallel SHA-256 prefix search using
multiple independent workers. Its main purpose is to compare measured completion
time against the ideal probabilistic model:

```text
E[T] = k / (N * lambda)
```

and to measure the empirical overhead:

```text
delta(N) = E[T_real] - k / (N * lambda)
```

The current implementation is useful as a reproducible baseline because it
already defines the worker logic, Docker Compose execution path, trial merging
procedure, and basic post-experiment analysis.

## 2. v5 Paper Positioning

The JIPS v5 paper should be positioned as:

```text
measurement-driven worker allocation for independent-trial probabilistic parallel search
```

The target contribution is not a full control system and not a new derivation of
the Poisson/Gamma waiting-time model. Instead, the paper studies whether a small
measurement step and an empirical overhead model can select an efficient worker
count for a new workload without exhaustively testing every candidate worker
count.

## 3. SHA-256 Prefix Search as a Controlled Workload

SHA-256 prefix matching is the controlled workload used for measurement. It is
not the whole problem definition.

The broader problem class is independent-trial probabilistic parallel search,
where each worker repeatedly samples candidate solutions with a low and
approximately independent success probability. SHA-256 prefix search is useful
because difficulty can be controlled, trials can be repeated, and success events
are easy to observe.

Related workload families include:

- randomized search
- Monte Carlo sampling
- distributed random search
- distributed grid/random hyperparameter search
- proof-of-work-like candidate search
- rare-event candidate generation
- embarrassingly parallel probabilistic search

## 4. Revised Contributions

C1. Empirically characterize how container overhead shifts the efficient worker
count across workload regimes in independent-trial probabilistic parallel search.

C2. Propose a measurement-driven worker allocation procedure that uses a
workload-specific N=1 calibration run and an empirical overhead model to select
a worker count without exhaustive sweeps.

C3. Evaluate the procedure through held-out workload validation and an
end-to-end online scenario against practical baselines, including Core-count,
Saturation-stop, Pilot-sweep, Max-worker, and Oracle.

The ideal Poisson/Gamma waiting-time model should be presented as modeling
background, not as a new theoretical contribution.

## 5. Research Questions

RQ1. How does empirical container overhead change the efficient worker count
across difficulty and target-count regimes?

RQ2. Under which workload regimes does a simple Core-count heuristic fail, and
can a measurement-driven allocation method adapt to those regimes?

RQ3. Can the proposed policy select a near-Oracle worker count under held-out
workload conditions without exhaustive worker-count sweeps?

RQ4. Does an end-to-end online scenario using only N=1 calibration select an
efficient worker count compared with practical heuristics?

## 6. Workload-Specific Lambda Calibration

The main Docker Compose experiment should use workload-specific calibration:

```text
lambda_hat(difficulty, k) = k / mean_T_real(N=1, difficulty, k)
```

This avoids a global lambda estimate and keeps the model aligned with workload
difficulty and target count. The notation should not be environment-specific in
the main experiment unless optional environment comparisons are actually run.

If optional Kubernetes single-node data is later collected, the notation may be
extended to:

```text
lambda_hat(environment, difficulty, k)
```

## 7. Discovery Grid

The discovery grid is used to find regimes where the efficient worker count
differs from simple static choices, especially Core-count.

```text
environment = docker-compose
difficulty = [4, 5]
k = [1, 3, 5, 10, 20, 40]
N = [1, 2, 4, 6, 8, 12, 16]
trials = 10
```

The goal is to identify:

- regimes where Core-count is near-Oracle
- regimes where Core-count has meaningful regret
- regimes where the proposed method can adaptively select a better or similarly
  efficient worker count

Core-count failure should not be fabricated. If Core-count is strong across most
regimes, that result should be reported honestly as a measurement finding.

## 8. Confirmatory Grid

The confirmatory grid should repeat the most important regimes found during
discovery. It should include low-k, mid-k, high-k, and harder-workload cases when
the runtime budget permits.

Recommended starting point:

```text
environment = docker-compose
difficulty = [5, 6]
k = [1, 3, 10, 40]
N = [1, 2, 4, 6, 8, 12, 16]
trials = 20 to 30
```

If runtime is limited, priority should go to regimes that test the paper's main
defense:

- where Core-count is expected to be strong
- where Core-count may fail
- where startup overhead may favor smaller N
- where saturation or rare-event behavior may favor larger N

## 9. Held-Out Validation

Held-out validation is the main defense against post-hoc fitting. It is more
important for the main claim than adding more execution environments.

Validation modes:

```text
k hold-out:
  train k = [1, 3, 10]
  test k = [20, 40]

difficulty hold-out:
  train difficulty = [4, 5]
  test difficulty = [6]

leave-one-N-out:
  train all N except one
  predict the held-out N
```

Main performance claims should use held-out results. The proposed method should
not be evaluated only on the same grid used to fit the empirical overhead model.

## 10. End-to-End Online Scenario

At least one end-to-end online scenario should be included. This scenario should
demonstrate worker-count selection before observing all worker counts for the
test workload.

Procedure:

```text
1. Select a held-out workload, such as difficulty=6, k=20 or difficulty=5, k=40.
2. Use only N=1 calibration for the held-out workload to estimate lambda_hat.
3. Use delta_hat trained from training regimes.
4. Predict T_pred(N) for candidate worker counts:
   T_pred(N) = k / (N * lambda_hat) + delta_hat(N)
5. Select the smallest N within 5% of the minimum predicted time.
6. Run or evaluate the selected_N as the proposed policy action.
7. Compare against practical baselines and Oracle.
```

Oracle is for evaluation only. The proposed method must not use observed test
results for all N when choosing selected_N.

## 11. Baselines

The evaluation should include realistic policies:

- Static-8
- Max-worker
- Core-count, with default N=6 on the current host
- optional Thread-count, for example N=12 if logical threads are reported
- Saturation-stop
- Pilot-sweep
- Oracle
- Proposed

Core-count should be treated as a strong practical baseline. The paper should
report where Core-count is near-Oracle and where it fails. The proposed method
should be described as workload-adaptive, not as a method that always beats
Core-count.

Pilot-sweep is useful because it represents a practical but more expensive
baseline. The proposed method can be contrasted with it by emphasizing that the
proposed policy uses only N=1 calibration for the new workload.

## 12. Main Experimental Scope

Docker Compose is the main environment for the JIPS v5 claim:

```text
Main paper = Docker Compose + held-out workload validation + online scenario
```

Kubernetes single-node support is optional and should only be considered after
the Docker Compose pipeline, held-out validation, online scenario, and policy
evaluation are working.

kind and cloud experiments are not required for the main claim. They may be left
as future work or appendix candidates.

## 13. N-BEATS and Black-Box Forecasting

Do not implement N-BEATS for this version. Black-box forecasting models can be
left as future work because this study focuses on interpretable
measurement-driven allocation.

## 14. Limitations

The planned study is intentionally scoped. Important limitations should be
reported clearly:

- The main evaluation uses Docker Compose on a single host.
- SHA-256 prefix search is a controlled workload, not a complete representation
  of all randomized or Monte Carlo search workloads.
- Core-count may be near-Oracle in some or many regimes.
- The proposed method depends on the quality of N=1 calibration and empirical
  overhead modeling.
- Oracle is an offline upper bound and is not deployable.
- Optional Kubernetes, kind, cloud, physical multi-node clusters, and black-box
  forecasting models are outside the main JIPS v5 claim.

## 15. Conservative Claim Template

A safe claim for the paper is:

```text
Core-count is a strong and practical baseline on a single host, but it is not
workload-adaptive. We therefore examine regimes in which the efficient worker
count deviates from the physical core count and evaluate whether a
measurement-driven allocation method can adapt to those regimes using only N=1
calibration for a new workload.
```

# Parallel Search Experiment

Empirical study for the paper:

**"확률 모델(포아송·감마 분포)을 활용한 분산 병렬 탐색 알고리즘의 평균 시간 복잡도 분석"**
*(Average-Case Time Complexity Analysis of Distributed Parallel Search Algorithms using Stochastic Models: A Docker-based Empirical Study)*

Submitted to: **2026 ASK Workshop** (Deadline: 2026-04-06)

---

## Overview

This project validates the hypothesis that N independent worker nodes searching in parallel reduce expected search time according to:

```
E[T] = k / (N * λ)
```

and measures the real overhead:

```
δ(N) = E[T_real] - k / (N * λ)
```

using SHA-256 hash prefix matching as the search task, running in Docker containers.

---

## Project Structure

```
parallel-search-experiment/
├── worker/
│   └── worker.py          # Hash search worker (SHA-256 prefix matching)
├── Dockerfile             # Worker container image
├── docker-compose.yml     # Multi-worker orchestration
├── merge_trial.py         # Merge per-worker JSONs → T_real → CSV
├── run_experiment.sh      # Full experiment automation (N=1,2,4,8,16 × 30 trials)
├── calibrate.py           # Find suitable DIFFICULTY value before experiment
├── analyze.py             # Statistical analysis and figure generation
├── results/               # Experiment output (gitignored)
└── figures/               # Generated plots (gitignored)
```

---

## Requirements

- Docker Desktop
- Python 3.11+
- Python packages (for analysis only):
  ```bash
  pip install pandas matplotlib scipy numpy
  ```

---

## Usage

### Step 0 — Calibrate difficulty

```bash
python calibrate.py
```

Find a `DIFFICULTY` value where N=1 takes 30–120 seconds.

### Step 1 — Run the full experiment

```bash
bash run_experiment.sh --difficulty 6 --k 10 --trials 30
```

Quick test mode (N=1, 2 trials, DIFFICULTY=4):

```bash
bash run_experiment.sh --test
```

Resume after interruption: re-run the same command. Already-completed trials are skipped automatically.

### Step 2 — Analyze results

```bash
python analyze.py
```

Outputs:
- `figures/theory_vs_measured.png` — Measured vs. theoretical E[T]
- `figures/delta_N.png` — Synchronization overhead δ(N)
- `figures/speedup.png` — Measured speedup vs. ideal linear speedup
- `results/summary_statistics.csv` — Per-N statistics table

---

## Experiment Parameters

| Parameter   | Value            |
|-------------|------------------|
| N (workers) | 1, 2, 4, 8, 16   |
| k (targets) | 10               |
| Trials      | 30 per N         |
| Environment | Docker Compose (single host, multi-container) |
| Search task | SHA-256 prefix matching (`"0" * DIFFICULTY`) |

---

## Worker ID Assignment

Each worker atomically claims a unique ID (1, 2, 3, …) by creating a `.claimed_id_N` file with `O_CREAT|O_EXCL`, which is atomic on both Linux (Docker) and NTFS (Windows). This guarantees independent search spaces: `seed = "worker_{id}_{trial}"`.

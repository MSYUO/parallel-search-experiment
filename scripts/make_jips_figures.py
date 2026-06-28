import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

summary_path = Path("results/summary/confirmatory_d6_key_regimes_30_summary_v2.csv")
policy_path = Path("results/summary/probe_to_confirmatory_policy_validation.csv")
out_dir = Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)

if not summary_path.exists():
    raise SystemExit(f"[ERROR] missing: {summary_path}")

if not policy_path.exists():
    raise SystemExit(f"[ERROR] missing: {policy_path}")

summary = pd.read_csv(summary_path)
policy = pd.read_csv(policy_path)

# -----------------------------
# Figure 1: Latency by worker count
# -----------------------------
plt.figure(figsize=(8, 5))

for k, g in summary.sort_values(["k", "N"]).groupby("k"):
    plt.plot(g["N"], g["mean_T_real"], marker="o", label=f"k={k}")
    plt.fill_between(g["N"], g["ci95_low"], g["ci95_high"], alpha=0.15)

plt.xlabel("Number of Workers (N)")
plt.ylabel("Mean Completion Time (seconds)")
plt.title("Latency by Worker Count")
plt.xticks(sorted(summary["N"].unique()))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "figure_1_latency_by_worker_count.png", dpi=300)
plt.close()

# -----------------------------
# Figure 2: Parallel efficiency by worker count
# -----------------------------
plt.figure(figsize=(8, 5))

for k, g in summary.sort_values(["k", "N"]).groupby("k"):
    plt.plot(g["N"], g["parallel_efficiency"], marker="o", label=f"k={k}")

plt.xlabel("Number of Workers (N)")
plt.ylabel("Parallel Efficiency")
plt.title("Parallel Efficiency by Worker Count")
plt.xticks(sorted(summary["N"].unique()))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "figure_2_parallel_efficiency_by_worker_count.png", dpi=300)
plt.close()

# -----------------------------
# Figure 3: Core-count vs selected vs max-worker latency
# -----------------------------
labels = [f"k={int(k)}" for k in policy["k"]]
x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(8, 5))
plt.bar(x - width, policy["core_count_T"], width, label="Core-count N=6")
plt.bar(x, policy["selected_T"], width, label="Probe-selected N")
plt.bar(x + width, policy["max_worker_T"], width, label="Max-worker N=16")

plt.xlabel("Workload")
plt.ylabel("Mean Completion Time (seconds)")
plt.title("Latency Comparison: Core-count vs Probe-selected vs Max-worker")
plt.xticks(x, labels)
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "figure_3_latency_policy_comparison.png", dpi=300)
plt.close()

# -----------------------------
# Figure 4: Efficiency comparison selected vs max-worker
# -----------------------------
plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, policy["selected_efficiency"], width, label="Probe-selected N")
plt.bar(x + width / 2, policy["max_worker_efficiency"], width, label="Max-worker N=16")

plt.xlabel("Workload")
plt.ylabel("Parallel Efficiency")
plt.title("Efficiency Comparison: Probe-selected vs Max-worker")
plt.xticks(x, labels)
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "figure_4_efficiency_selected_vs_max_worker.png", dpi=300)
plt.close()

print("[OK] figures saved to:", out_dir)
for p in sorted(out_dir.glob("*.png")):
    print(" -", p)

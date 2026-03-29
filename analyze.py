"""
analyze.py — Experiment result analysis and figure generation
Usage: python analyze.py [--csv results/experiment_results.csv] [--results-dir results]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless (no display required)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

# ── style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  11,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
})

N_ORDER = [1, 2, 4, 8, 16]


# ── helpers ────────────────────────────────────────────────────────

def mean_ci(series: pd.Series, confidence: float = 0.95):
    """Return (mean, ci_lower, ci_upper) using t-distribution."""
    n = len(series)
    m = series.mean()
    if n < 2:
        return m, m, m
    se = stats.sem(series)
    h  = se * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return m, m - h, m + h


def lambda_from_n1(df: pd.DataFrame, k: int):
    """Estimate λ from N=1 data: λ = k / mean(T_real)."""
    n1 = df[df["n_workers"] == 1]["T_real"]
    mean_t, ci_lo, ci_hi = mean_ci(n1)
    lam      = k / mean_t
    lam_hi   = k / ci_lo   # CI inverts because λ = k/T
    lam_lo   = k / ci_hi
    return lam, lam_lo, lam_hi


def theory_t(n: int, k: int, lam: float) -> float:
    return k / (n * lam)


# ── per-N statistics ────────────────────────────────────────────────

def build_summary(df: pd.DataFrame, k: int, lam: float) -> pd.DataFrame:
    rows = []
    mean_t_n1 = df[df["n_workers"] == 1]["T_real"].mean()

    for n in N_ORDER:
        sub = df[df["n_workers"] == n]["T_real"]
        if sub.empty:
            continue
        m, lo, hi = mean_ci(sub)
        th        = theory_t(n, k, lam)
        delta     = m - th
        speedup   = mean_t_n1 / m
        rows.append({
            "N":        n,
            "mean_T":   round(m,      6),
            "std_T":    round(sub.std(ddof=1), 6) if len(sub) > 1 else 0.0,
            "ci_lower": round(lo,     6),
            "ci_upper": round(hi,     6),
            "theory_T": round(th,     6),
            "delta_N":  round(delta,  6),
            "speedup":  round(speedup, 4),
        })
    return pd.DataFrame(rows)


# ── figure 1: theory vs measured ──────────────────────────────────

def plot_theory_vs_measured(summary: pd.DataFrame, out_path: str):
    ns   = summary["N"].values
    meas = summary["mean_T"].values
    lo   = summary["mean_T"].values - summary["ci_lower"].values
    hi   = summary["ci_upper"].values - summary["mean_T"].values
    th   = summary["theory_T"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(ns, meas, yerr=[lo, hi],
                fmt="o-", color="#1f77b4", linewidth=2, markersize=7,
                capsize=5, label="Measured mean ± 95% CI")
    ax.plot(ns, th,
            "r--", linewidth=1.8, marker="s", markersize=6,
            label=r"Theory $k/(N\lambda)$")

    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Number of Workers (N)")
    ax.set_ylabel("Average Search Time E[T]  (s)")
    ax.set_title("Measured vs. Theoretical Average Search Time")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[analyze] Saved {out_path}")


# ── figure 2: δ(N) overhead ────────────────────────────────────────

def plot_delta(summary: pd.DataFrame, out_path: str):
    ns    = summary["N"].values
    delta = summary["delta_N"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta]
    ax.bar(range(len(ns)), delta, color=colors, alpha=0.8, width=0.5)
    ax.plot(range(len(ns)), delta, "ko-", linewidth=1.5, markersize=6, zorder=5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Number of Workers (N)")
    ax.set_ylabel(r"$\delta(N) = E[T_{real}] - k/(N\lambda)$  (s)")
    ax.set_title(r"Synchronization Overhead $\delta(N)$ vs. Worker Count")
    ax.legend(handles=[
        matplotlib.patches.Patch(color="#2ca02c", alpha=0.8, label="Positive overhead"),
        matplotlib.patches.Patch(color="#d62728", alpha=0.8, label="Negative (faster than theory)"),
    ], loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[analyze] Saved {out_path}")


# ── figure 3: speedup ──────────────────────────────────────────────

def plot_speedup(summary: pd.DataFrame, out_path: str):
    ns      = summary["N"].values
    speedup = summary["speedup"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(ns, speedup, "o-", color="#1f77b4", linewidth=2,
            markersize=8, label="Measured speedup")
    ax.plot(ns, ns,
            "--", color="gray", linewidth=1.5, label="Ideal linear speedup (y = N)")

    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Number of Workers (N)")
    ax.set_ylabel("Speedup  $S(N) = E[T_1] / E[T_N]$")
    ax.set_title("Measured Speedup vs. Ideal Linear Speedup")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[analyze] Saved {out_path}")


# ── terminal table ──────────────────────────────────────────────────

def print_summary_table(summary: pd.DataFrame, lam: float, lam_lo: float, lam_hi: float):
    print()
    print("=" * 74)
    print(f"  lambda = {lam:.6f}  (95% CI: [{lam_lo:.6f}, {lam_hi:.6f}])")
    print("=" * 74)
    header = f"{'N':>4}  {'mean_T':>9}  {'std_T':>9}  {'CI_lower':>9}  {'CI_upper':>9}  {'theory_T':>9}  {'delta_N':>9}  {'speedup':>7}"
    print(header)
    print("-" * 74)
    for _, row in summary.iterrows():
        print(
            f"{int(row['N']):>4}  "
            f"{row['mean_T']:>9.4f}  "
            f"{row['std_T']:>9.4f}  "
            f"{row['ci_lower']:>9.4f}  "
            f"{row['ci_upper']:>9.4f}  "
            f"{row['theory_T']:>9.4f}  "
            f"{row['delta_N']:>9.4f}  "
            f"{row['speedup']:>7.3f}"
        )
    print("=" * 74)
    print()


# ── main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze parallel search experiment results")
    parser.add_argument("--csv",         default="results/experiment_results.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    # load
    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    required = {"trial", "n_workers", "k", "difficulty", "T_real"}
    missing  = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    k = int(df["k"].iloc[0])
    print(f"[analyze] Loaded {len(df)} rows  k={k}  "
          f"difficulty={df['difficulty'].iloc[0]}")
    print(f"[analyze] N values present: "
          f"{sorted(df['n_workers'].unique().tolist())}")

    os.makedirs(args.figures_dir, exist_ok=True)

    # λ
    lam, lam_lo, lam_hi = lambda_from_n1(df, k)
    print(f"\nlambda = {lam:.6f}  (95% CI: [{lam_lo:.6f}, {lam_hi:.6f}])")

    # summary table
    summary = build_summary(df, k, lam)
    print_summary_table(summary, lam, lam_lo, lam_hi)

    # save summary CSV
    summary_path = os.path.join(args.results_dir, "summary_statistics.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[analyze] Saved {summary_path}")

    # figures
    plot_theory_vs_measured(summary,
        os.path.join(args.figures_dir, "theory_vs_measured.png"))
    plot_delta(summary,
        os.path.join(args.figures_dir, "delta_N.png"))
    plot_speedup(summary,
        os.path.join(args.figures_dir, "speedup.png"))

    print("\n[analyze] Done.")


if __name__ == "__main__":
    main()

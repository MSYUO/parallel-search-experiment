from pathlib import Path
import pandas as pd
import re

summary_path = Path("results/summary/confirmatory_d6_key_regimes_30_summary_v2.csv")
policy_path = Path("results/summary/probe_to_confirmatory_policy_validation.csv")
out_dir = Path("results/tables")
out_dir.mkdir(parents=True, exist_ok=True)

if not summary_path.exists():
    raise SystemExit(f"[ERROR] missing: {summary_path}")
if not policy_path.exists():
    raise SystemExit(f"[ERROR] missing: {policy_path}")

summary = pd.read_csv(summary_path)
policy = pd.read_csv(policy_path)


def fmt_float(x, digits=3):
    try:
        return round(float(x), digits)
    except Exception:
        return x


def write_table(df, name):
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"
    tex_path = out_dir / f"{name}.tex"

    df.to_csv(csv_path, index=False)

    # dependency-free markdown table
    with open(md_path, "w", encoding="utf-8") as f:
        cols = list(df.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in df.iterrows():
            f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")

    df.to_latex(tex_path, index=False, escape=True)

    print(f"[OK] {name}")
    print(f" - {csv_path}")
    print(f" - {md_path}")
    print(f" - {tex_path}")


# ============================================================
# Table 1. Full confirmatory summary by worker count
# ============================================================
rows = []

for (d, k), g in summary.groupby(["difficulty", "k"]):
    g = g.sort_values("N")
    oracle_T = g["mean_T_real"].min()
    oracle_N = int(g.loc[g["mean_T_real"].idxmin(), "N"])

    selected_N = int(policy[(policy["difficulty"] == d) & (policy["k"] == k)]["probe_selected_N"].iloc[0])

    for _, r in g.iterrows():
        N = int(r["N"])
        role = []
        if N == 6:
            role.append("Core-count")
        if N == selected_N:
            role.append("Probe-selected")
        if N == 16:
            role.append("Max-worker")
        if N == oracle_N:
            role.append("Oracle")

        rows.append({
            "Workload": f"d={int(d)}, k={int(k)}",
            "N": N,
            "Role": ", ".join(role) if role else "-",
            "Mean T (s)": fmt_float(r["mean_T_real"]),
            "95% CI (s)": f"[{fmt_float(r['ci95_low'])}, {fmt_float(r['ci95_high'])}]",
            "Speedup": fmt_float(r["speedup"]),
            "Efficiency": fmt_float(r["parallel_efficiency"]),
            "Regret vs Oracle (%)": fmt_float((r["mean_T_real"] / oracle_T - 1) * 100),
        })

table1 = pd.DataFrame(rows)
write_table(table1, "table_1_confirmatory_worker_count_summary")


# ============================================================
# Table 2. Main probe-guided policy validation
# ============================================================
rows = []

for _, r in policy.iterrows():
    selected_N = int(r["probe_selected_N"])
    max_N = 16
    worker_saving = (1 - selected_N / max_N) * 100
    efficiency_ratio = float(r["selected_efficiency"]) / float(r["max_worker_efficiency"])

    rows.append({
        "Workload": f"d={int(r['difficulty'])}, k={int(r['k'])}",
        "Discovery Oracle N": int(r["discovery_oracle_N"]),
        "Probe-selected N": selected_N,
        "Confirmatory Oracle N": int(r["confirmatory_oracle_N"]),
        "Selected T (s)": fmt_float(r["selected_T"]),
        "Selected Regret (%)": fmt_float(r["selected_vs_oracle_percent"]),
        "Selected Efficiency": fmt_float(r["selected_efficiency"]),
        "Core-count Regret (%)": fmt_float(r["core_regret_percent"]),
        "Max-worker Efficiency": fmt_float(r["max_worker_efficiency"]),
        "Worker Saving vs Max (%)": fmt_float(worker_saving),
        "Efficiency Gain vs Max (x)": fmt_float(efficiency_ratio),
    })

table2 = pd.DataFrame(rows)
write_table(table2, "table_2_probe_guided_policy_validation")


# ============================================================
# Table 3. Practical policy comparison
# ============================================================
rows = []

for _, r in policy.iterrows():
    d = int(r["difficulty"])
    k = int(r["k"])
    g = summary[(summary["difficulty"] == d) & (summary["k"] == k)]
    oracle_T = float(g["mean_T_real"].min())

    policies = [
        ("Core-count", 6),
        ("Probe-selected", int(r["probe_selected_N"])),
        ("Max-worker", 16),
    ]

    for policy_name, N in policies:
        row = g[g["N"] == N].iloc[0]
        T = float(row["mean_T_real"])
        eff = float(row["parallel_efficiency"])

        rows.append({
            "Workload": f"d={d}, k={k}",
            "Policy": policy_name,
            "N": N,
            "Mean T (s)": fmt_float(T),
            "Regret vs Oracle (%)": fmt_float((T / oracle_T - 1) * 100),
            "Efficiency": fmt_float(eff),
        })

table3 = pd.DataFrame(rows)
write_table(table3, "table_3_practical_policy_comparison")


# ============================================================
# Table 4. Optional online prediction / limitation table
# ============================================================
online_files = [
    ("quadratic", "results/summary/online_d6_train_k3_k10_test_k40.csv"),
    ("quadratic", "results/summary/online_d6_train_k3_k40_test_k10.csv"),
    ("quadratic", "results/summary/online_d6_train_k10_k40_test_k3.csv"),
    ("linear", "results/summary/online_linear_d6_train_k3_k10_test_k40.csv"),
    ("linear", "results/summary/online_linear_d6_train_k3_k40_test_k10.csv"),
    ("linear", "results/summary/online_linear_d6_train_k10_k40_test_k3.csv"),
]

rows = []

for model, file in online_files:
    p = Path(file)
    if not p.exists():
        continue

    df = pd.read_csv(p)
    if df.empty:
        continue

    r = df.iloc[0]
    test_setting = str(r["test_setting"])
    m = re.search(r"k=(\d+)", test_setting)
    test_k = m.group(1) if m else "?"

    rows.append({
        "Model": model,
        "Test workload": f"d=6, k={test_k}",
        "Selected N": int(r["selected_N"]),
        "Oracle N": int(r["oracle_N"]),
        "Observed T (s)": fmt_float(r["observed_T"]),
        "Oracle T (s)": fmt_float(r["oracle_T"]),
        "Regret (%)": fmt_float(r["regret_percent"]),
        "Interpretation": "success" if float(r["regret_percent"]) <= 5 else "unstable",
    })

if rows:
    table4 = pd.DataFrame(rows)
    write_table(table4, "table_4_online_prediction_ablation_optional")
else:
    print("[SKIP] Table 4 online files not found")

print("\n[DONE] Tables saved under results/tables/")

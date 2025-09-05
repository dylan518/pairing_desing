import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"


def parse_task_info(task: str):
    """Return (family, n_bits) for labels like 'nk_32' or 'trap_100'."""
    if task.startswith("nk_"):
        return "nk", int(task.split("_")[1])
    if task.startswith("trap_"):
        return "trap", int(task.split("_")[1])
    raise ValueError(f"Unrecognized task label: {task}")


def load_all_equal_csvs() -> pd.DataFrame:
    files = sorted(OUT_DIR.glob("equal_*.csv"))
    # exclude combined
    files = [f for f in files if f.name != "equal_all_tasks.csv"]
    if not files:
        raise FileNotFoundError("No per-task CSVs found in outputs/. Run test.py first.")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    # Ensure columns exist
    expected = {"task","embed_dim","tau","final_best"}
    missing = expected - set(all_df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSVs: {missing}")
    # Derive task family and n_bits
    fam_bits = all_df["task"].apply(parse_task_info)
    all_df["family"] = fam_bits.apply(lambda x: x[0])
    all_df["n_bits"] = fam_bits.apply(lambda x: x[1])
    return all_df


def summarize_best_k(df: pd.DataFrame) -> pd.DataFrame:
    """For each task, pick (k, tau) with max final_best; compute k/n_bits.

    Returns columns: task, family, n_bits, best_k, best_tau, best_final_best, k_over_bits
    """
    rows = []
    for task, g in df.groupby("task"):
        idx = g["final_best"].idxmax()
        row = g.loc[idx]
        best_k = int(row["embed_dim"])
        best_tau = float(row["tau"])
        best_val = float(row["final_best"])
        family, n_bits = parse_task_info(task)
        rows.append({
            "task": task,
            "family": family,
            "n_bits": n_bits,
            "best_k": best_k,
            "best_tau": best_tau,
            "best_final_best": best_val,
            "k_over_bits": best_k / n_bits
        })
    return pd.DataFrame(rows).sort_values(["family","n_bits"]).reset_index(drop=True)


def plot_percent_vs_bits(summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6.5, 4.2))
    for fam, color, marker in [("nk", "#2563EB", "o"), ("trap", "#10B981", "s")]:
        sub = summary[summary["family"] == fam]
        if len(sub) == 0:
            continue
        plt.scatter(sub["n_bits"], 100*sub["k_over_bits"], c=color, marker=marker, label=fam.upper(), s=70, edgecolor="white")

    # naive trendline across all points
    x = summary["n_bits"].values.astype(float)
    y = (100*summary["k_over_bits"].values).astype(float)
    if len(x) >= 2 and np.ptp(x) > 0:
        coef = np.polyfit(x, y, deg=1)
        xs = np.linspace(x.min()-2, x.max()+2, 100)
        ys = coef[0]*xs + coef[1]
        plt.plot(xs, ys, color="#6B7280", linestyle="--", linewidth=1.5, label="linear trend")

    plt.xlabel("genome length (bits)")
    plt.ylabel("best embedding as % of genome (100·k/D)")
    plt.title("Heuristic: best k as a percentage of genome size")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    df = load_all_equal_csvs()
    summary = summarize_best_k(df)
    out_csv = OUT_DIR / "analysis_k_ratio_summary.csv"
    summary.to_csv(out_csv, index=False)

    scatter_path = OUT_DIR / "analysis_k_ratio_scatter.png"
    plot_percent_vs_bits(summary, scatter_path)

    # Print quick console summary
    print("Saved:")
    print(f"- summary CSV: {out_csv}")
    print(f"- scatter:     {scatter_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()



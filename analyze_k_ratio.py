import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import erf, sqrt


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"


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
    # trait_dim is optional (older runs may lack it). If missing, try to infer from filename suffix _D{D}.
    expected = {"task","embed_dim","tau","final_best"}
    missing = expected - set(all_df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSVs: {missing}")
    if "trait_dim" not in all_df.columns:
        dims = []
        for f, n in zip(files, range(len(files))):
            m = re.search(r"_D(\d+)\.csv$", f.name)
            dims.append(int(m.group(1)) if m else np.nan)
        # broadcast per-file trait_dim across rows from that file
        parts = []
        row_offset = 0
        for f, dim in zip(files, dims):
            df = pd.read_csv(f)
            df["trait_dim"] = dim
            parts.append(df)
        all_df = pd.concat(parts, ignore_index=True)
    # Derive task family and n_bits
    return all_df


def summarize_best_k(df: pd.DataFrame) -> pd.DataFrame:
    """For each task×trait_dim, pick (k, tau) with max final_best; compute k/trait_dim.

    Returns columns: task, trait_dim, best_k, best_tau, best_final_best, k_over_bits
    """
    rows = []
    for (task, D), g in df.groupby(["task","trait_dim"], dropna=False):
        idx = g["final_best"].idxmax()
        row = g.loc[idx]
        best_k = int(row["embed_dim"])
        best_tau = float(row["tau"])
        best_val = float(row["final_best"])
        rows.append({
            "task": task,
            "trait_dim": int(D) if not np.isnan(D) else np.nan,
            "best_k": best_k,
            "best_tau": best_tau,
            "best_final_best": best_val,
            "k_over_bits": (best_k / D) if (D and not np.isnan(D) and D>0) else np.nan
        })
    return pd.DataFrame(rows).sort_values(["task","trait_dim"]).reset_index(drop=True)


def plot_percent_vs_bits(summary: pd.DataFrame, out_path: Path) -> None:
    # (Deprecated in simplified run: keep stub for compatibility.)
    pass


def main():
    df = load_all_equal_csvs()
    summary = summarize_best_k(df)
    out_csv = OUT_DIR / "analysis_k_ratio_summary.csv"
    summary.to_csv(out_csv, index=False)
    # Global linear fit with intercept: k = a·D + b
    all_sub = summary.dropna(subset=["trait_dim","best_k"]) 
    x_all = all_sub["trait_dim"].values.astype(float)
    y_all = all_sub["best_k"].values.astype(float)
    n_all = len(x_all)
    xm, ym = x_all.mean(), y_all.mean()
    sxx = np.sum((x_all - xm)**2)
    sxy = np.sum((x_all - xm) * (y_all - ym))
    a = sxy / sxx if sxx > 0 else 0.0
    b = ym - a * xm
    yhat_all = a * x_all + b
    ss_res = np.sum((y_all - yhat_all)**2)
    ss_tot = np.sum((y_all - ym)**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
    se = np.sqrt(ss_res / (n_all - 2)) if n_all > 2 else np.inf
    se_a = se / np.sqrt(sxx) if sxx > 0 else np.inf
    t_stat = a / se_a if se_a > 0 else np.nan
    p_val = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))) if np.isfinite(t_stat) else np.nan
    # Standalone global linear figure (scatter per task + line k=a·D+b)
    fig2, ax2 = plt.subplots(figsize=(6.4, 4.2))
    cmap = plt.get_cmap("tab10")
    for i, t in enumerate(sorted(summary["task"].unique())):
        sub = summary[summary["task"] == t].dropna(subset=["trait_dim","best_k"])
        if len(sub) == 0:
            continue
        ax2.scatter(sub["trait_dim"], sub["best_k"], s=36, c=[cmap(i%10)], edgecolor="white", label=t)
    xs2 = np.linspace(x_all.min(), x_all.max(), 200)
    ax2.plot(xs2, a * xs2 + b, color="#111827", linestyle="-", linewidth=2, label=f"k = {a:.3f}·D + {b:.2f}")
    ax2.set_xlabel("genome length D (bits)")
    ax2.set_ylabel("optimal embedding size k (bits)")
    ax2.set_title(f"Global linear fit (R²={r2:.3f}, p={p_val:.2g})")
    ax2.legend(frameon=False, ncol=2)
    fig2.tight_layout()
    global_fig = OUT_DIR / "analysis_k_global_linear.png"
    fig2.savefig(global_fig, dpi=200)
    plt.close(fig2)

    # Save global linear stats
    global_csv = OUT_DIR / "analysis_k_global_linear.csv"
    pd.DataFrame([{"slope_a": a, "intercept_b": b, "r2": r2, "p": p_val, "n": int(n_all)}]).to_csv(global_csv, index=False)

    # Print quick console summary
    print("Saved:")
    print(f"- summary CSV: {out_csv}")
    print(f"- global linear: {global_csv}")
    print(f"- global fig:  {global_fig}")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Global linear fit: k = a·D + b with a={a:.4f}, b={b:.3f}, R2={r2:.3f}, p={p_val:.4g}, n={n_all}")

    # New: distribution of best k and best tau overall and per task
    dist_k = summary.groupby("best_k").size().reset_index(name="count").sort_values("best_k")
    dist_tau = summary.groupby("best_tau").size().reset_index(name="count").sort_values("best_tau")
    dist_task_k = summary.groupby(["task","best_k"]).size().reset_index(name="count")
    dist_task_tau = summary.groupby(["task","best_tau"]).size().reset_index(name="count")
    dist_k.to_csv(OUT_DIR / "best_k_distribution.csv", index=False)
    dist_tau.to_csv(OUT_DIR / "best_tau_distribution.csv", index=False)
    dist_task_k.to_csv(OUT_DIR / "best_k_distribution_by_task.csv", index=False)
    dist_task_tau.to_csv(OUT_DIR / "best_tau_distribution_by_task.csv", index=False)

    # Plots: bar charts for global distributions
    def _bar(df_plot, xcol, ycol, title, out_name):
        plt.figure(figsize=(6,3.2))
        plt.bar(df_plot[xcol].astype(str), df_plot[ycol].values, color="#4F46E5")
        plt.xlabel(xcol)
        plt.ylabel("count")
        plt.title(title)
        plt.tight_layout()
        p = OUT_DIR / out_name
        plt.savefig(p, dpi=200)
        plt.close()

    _bar(dist_k, "best_k", "count", "Distribution of best embedding k (winners)", "best_k_distribution.png")
    _bar(dist_tau, "best_tau", "count", "Distribution of best tau (winners)", "best_tau_distribution.png")


if __name__ == "__main__":
    main()



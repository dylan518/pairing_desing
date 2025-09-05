# Rerun: embed_dim × τ grids across multiple sizes for NK and Trap,
# generating labeled tables in the format `nk_[n_bits]` and `trap_[n_bits]`.
# Equal-offspring GA only (fecundity-weighted variant removed).
#
# Config kept light for speed: seeds=[0,1], gens=40, embed_dims=[0,1,2,3,5,8], taus=[0.0,0.1,0.4,0.7,1.0]
# Tasks: NK sizes n_bits ∈ {32,48,64} with K=8; Trap sizes n_bits ∈ {50,100}.

# Standard libs
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Third-party
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

# Equal-offspring GA (moved to separate module)
from equal_ga import FWSMConfigEq, FWSMGAEqualOffspring
from problems import make_problem

# ---------- Utility: output & display helpers ----------

# Create an `outputs` directory next to this script for any CSVs we save.
_SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def save_heatmap_png(title: str, df: pd.DataFrame, path: Path) -> None:
    """Save a tau × embed_dim heatmap of final_best as a PNG."""
    pivot = (
        df.pivot(index="embed_dim", columns="tau", values="final_best")
          .sort_index(axis=0)
          .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(y) for y in pivot.index])
    ax.set_xlabel("tau")
    ax.set_ylabel("embed_dim")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("final_best")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- Core helpers ----------
def _row_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / (n + eps)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def _zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (x - x.mean()) / (x.std() + eps)

# (fecundity-weighted GA removed)

# (equal-offspring GA is defined in equal_ga.py)

# ---------- Tasks now imported from problems.py ----------

# ---------- Runner ----------
def _run_single(task_label: str, k: int, tau: float, seed: int, gens: int = 40):
    """Run one GA trial for the given (task, k, tau, seed).

    Rebuilds the task deterministically inside the worker to avoid pickling
    closures across processes.
    """
    trait_dim, fitness_fn = make_problem(task_label)
    cfg = FWSMConfigEq(
        pop_size=60, gens=gens, trait_dim=trait_dim,
        embed_dim=k, tau=max(tau, 1e-8),
        sigma_traits=0.22, sigma_embed=0.05,
        child_budget=60, seed=seed,
    )
    ga = FWSMGAEqualOffspring(cfg)
    hist = ga.fit(fitness_fn)
    return {
        "final_best": float(hist["best"][-1]),
        "curve": hist["best"].tolist(),
    }


def run_grid(task_label: str, embed_dims, taus, seeds, gens=40, max_workers: int | None = None):
    trait_dim, fitness_fn = make_problem(task_label)
    records = []
    # Choose worker count
    if max_workers is None:
        max_workers = max(1, min(os.cpu_count() or 1, 8))
    for k in embed_dims:
        for tau in taus:
            futures = []
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                for seed in seeds:
                    futures.append(ex.submit(_run_single, task_label, int(k), float(tau), int(seed), int(gens)))
                results = []
                for fut in as_completed(futures):
                    results.append(fut.result())
            finals = np.array([r["final_best"] for r in results], dtype=float)
            curves = np.vstack([np.asarray(r["curve"], dtype=float) for r in results])
            mean_curve = np.mean(curves, axis=0)
            records.append({
                "task": task_label,
                "embed_dim": k,
                "tau": tau,
                "final_best": float(finals.mean()),
                "best@10": float(mean_curve[10]),
                "best@20": float(mean_curve[20]),
                "best@40": float(mean_curve[-1]),
            })
    return pd.DataFrame.from_records(records)

embed_dims = [0,1,2,3,5,8]
taus = [0.0, 0.1, 0.4, 0.7, 1.0]
seeds = [0,1]
gens = 40
tasks = ["nk_32","nk_48","nk_64","trap_50","trap_100"]

def main():
    all_dfs = []
    for task in tasks:
        df = run_grid(task, embed_dims, taus, seeds, gens=gens)
        all_dfs.append(df)
        title = f"Equal-offspring {task} (final_best)"
        csv_path = OUTPUT_DIR / f"equal_{task}.csv"
        df.to_csv(csv_path, index=False)
        heatmap_path = OUTPUT_DIR / f"equal_{task}_heatmap.png"
        save_heatmap_png(title, df, heatmap_path)

    # Combined summary
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_sorted = combined.sort_values(["task","embed_dim","tau"]).reset_index(drop=True)

    # Save combined
    combined_path = OUTPUT_DIR / "equal_all_tasks.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Saved per-task CSVs and heatmaps to {OUTPUT_DIR}. Combined CSV: {combined_path}")


if __name__ == "__main__":
    main()

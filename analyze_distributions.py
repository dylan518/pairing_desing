import os
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from equal_ga import FWSMConfigEq, FWSMGAEqualOffspring
from problems import make_problem


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


# Configuration — pick ONE problem and ONE trait size
TASK = "Levy"         # e.g., "Rastrigin", "Ackley", "Rosenbrock", "Sphere"
TRAIT_DIM = 110              # D to study
EMBED_DIMS = [0, 1, 2, 3, 5, 8]
TAUS = [0.0, 0.5]
REPS = 10                   # number of independent runs per (k, tau)
POPSIZE = 100
GENS = 250
MAX_WORKERS = max(1, min(os.cpu_count() or 1, 8))


def derive_crn_seed(task_label: str, trait_dim: int, rep: int) -> int:
    """Common-random-numbers across (k, tau) for replicate rep.

    Each replicate uses one RNG seed shared across all grid cells to reduce
    between-setting noise while still measuring distribution across reps.
    """
    payload = f"{task_label}|D={trait_dim}|rep={rep}".encode()
    h = hashlib.md5(payload).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False)


def run_single(task_label: str, trait_dim: int, k: int, tau: float, rep: int) -> dict:
    dim, fitness_fn = make_problem(task_label, trait_dim=trait_dim)
    seed = derive_crn_seed(task_label, trait_dim, rep)
    cfg = FWSMConfigEq(
        pop_size=POPSIZE, gens=GENS, trait_dim=dim,
        embed_dim=k, tau=max(tau, 1e-8),
        sigma_traits=0.22, sigma_embed=0.05,
        child_budget=POPSIZE, seed=seed,
    )
    ga = FWSMGAEqualOffspring(cfg)
    hist = ga.fit(fitness_fn)
    return {
        "task": task_label,
        "trait_dim": trait_dim,
        "embed_dim": k,
        "tau": tau,
        "rep": rep,
        "final_best": float(hist["best"][-1]),
    }


def collect_distributions():
    futures = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for k in EMBED_DIMS:
            for tau in TAUS:
                for rep in range(REPS):
                    futures.append(ex.submit(run_single, TASK, TRAIT_DIM, k, tau, rep))
        rows = []
        for f in as_completed(futures):
            rows.append(f.result())
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"dist_{TASK.lower()}_D{TRAIT_DIM}.csv"
    df.to_csv(out_csv, index=False)
    return df, out_csv


def plot_violin_grid(df: pd.DataFrame, out_path: Path):
    ks = EMBED_DIMS
    taus = TAUS
    fig, axes = plt.subplots(len(taus), len(ks), figsize=(2.4*len(ks), 1.8*len(taus)), sharey=True)
    for i, tau in enumerate(taus):
        for j, k in enumerate(ks):
            ax = axes[i, j] if len(taus) > 1 else (axes[j] if len(ks) > 1 else axes)
            vals = df[(df["embed_dim"]==k) & (df["tau"]==tau)]["final_best"].values
            if len(vals) == 0:
                ax.axis('off'); continue
            parts = ax.violinplot(vals, showmeans=True, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor('#60A5FA'); pc.set_alpha(0.7)
            parts['cmeans'].set_color('#111827')
            ax.set_title(f"k={k}, τ={tau}", fontsize=9)
            if j==0:
                ax.set_ylabel("final_best")
            ax.set_xticks([])
    fig.suptitle(f"{TASK} (D={TRAIT_DIM}) — distribution across {REPS} reps", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_winner_frequency(df: pd.DataFrame, out_path: Path):
    # For each rep, find winning (k,tau)
    wins = df.sort_values("final_best").groupby("rep").tail(1)
    table = wins.pivot_table(index="embed_dim", columns="tau", values="final_best", aggfunc=len, fill_value=0)
    ks = EMBED_DIMS; taus = TAUS
    table = table.reindex(index=ks, columns=taus, fill_value=0)
    plt.figure(figsize=(1.6*len(taus), 1.2*len(ks)))
    im = plt.imshow(table.values, cmap="viridis")
    plt.xticks(range(len(taus)), taus)
    plt.yticks(range(len(ks)), ks)
    plt.xlabel("tau"); plt.ylabel("embed_dim (k)")
    plt.title(f"Winner frequency across {REPS} reps (higher = more consistent)")
    for i in range(len(ks)):
        for j in range(len(taus)):
            plt.text(j, i, int(table.values[i,j]), va='center', ha='center', color='white' if table.values[i,j]>REPS/3 else 'black', fontsize=8)
    plt.colorbar(im, label="wins")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    df, csv_path = collect_distributions()
    fig1 = OUT_DIR / f"dist_{TASK.lower()}_D{TRAIT_DIM}_violin.png"
    plot_violin_grid(df, fig1)
    fig2 = OUT_DIR / f"dist_{TASK.lower()}_D{TRAIT_DIM}_wins.png"
    plot_winner_frequency(df, fig2)
    print("Saved:")
    print(f"- distributions CSV: {csv_path}")
    print(f"- violin grid:       {fig1}")
    print(f"- winner frequency:  {fig2}")


if __name__ == "__main__":
    main()



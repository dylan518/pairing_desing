from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"


def load_combined() -> pd.DataFrame:
    p = OUT_DIR / "equal_all_tasks.csv"
    if not p.exists():
        raise FileNotFoundError("Run test.py first to produce outputs/equal_all_tasks.csv")
    df = pd.read_csv(p)
    # Handle old runs without trait_dim
    if "trait_dim" not in df.columns:
        df["trait_dim"] = np.nan
    return df


def pivot_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    sub = df[df["task"] == task]
    pv = sub.pivot_table(index="embed_dim", columns="tau", values="final_best", aggfunc=np.mean)
    return pv


def _zscore_grid(values: np.ndarray) -> np.ndarray:
    v = values.astype(float)
    m = np.nanmean(v)
    s = np.nanstd(v)
    return (v - m) / s if s > 0 else (v * 0)


def _vectorize_grid(pv: pd.DataFrame, ks: list, taus: list, standardize: bool, rank: bool) -> np.ndarray:
    sub = pv.loc[ks, taus].values
    if rank:
        # Spearman: rank within grid before vectorizing
        r = pd.DataFrame(sub).rank(method="average").values
        return _zscore_grid(r).flatten() if standardize else r.flatten()
    return _zscore_grid(sub).flatten() if standardize else sub.flatten()


def corr_between_tasks(df: pd.DataFrame, tasks: list[str], standardize: bool = True, spearman: bool = False) -> pd.DataFrame:
    # Build vectorized performance per (k,tau)
    grids = {}
    for t in tasks:
        pv = pivot_task(df, t)
        # Align grids
        pv = pv.sort_index(axis=0).sort_index(axis=1)
        grids[t] = pv
    # Intersect (k,tau) support across tasks
    common_ks = sorted(set.intersection(*(set(g.index) for g in grids.values())))
    common_taus = sorted(set.intersection(*(set(g.columns) for g in grids.values())))
    if not common_ks or not common_taus:
        raise ValueError("No common (k,tau) grid across tasks.")
    vecs = {}
    for t,g in grids.items():
        vecs[t] = _vectorize_grid(g, common_ks, common_taus, standardize=standardize, rank=spearman)
    # Correlation matrix
    mat = np.corrcoef(np.vstack([vecs[t] for t in tasks]))
    corr_df = pd.DataFrame(mat, index=tasks, columns=tasks)
    return corr_df, common_ks, common_taus


def plot_corr_heatmap(corr_df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(4.8, 4.2))
    # Use a widely available diverging colormap
    im = plt.imshow(corr_df.values, cmap="RdBu", vmin=-1, vmax=1)
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr_df.index)), corr_df.index)
    plt.colorbar(im, label="Pearson r")
    plt.title("Cross-problem correlation of (k,τ) performance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def corr_across_dims_for_task(df: pd.DataFrame, task: str, standardize: bool = True, spearman: bool = False) -> tuple[pd.DataFrame, list[int]]:
    sub = df[df["task"] == task]
    dims = sorted([d for d in sub["trait_dim"].unique() if pd.notna(d)])
    if not dims:
        raise ValueError(f"No trait_dim values for task {task}. Re-run test.py to include trait_dim.")
    # Build grid vectors for each D over common (k,tau)
    pv_by_D = {}
    for D in dims:
        g = sub[sub["trait_dim"] == D]
        pv = g.pivot_table(index="embed_dim", columns="tau", values="final_best", aggfunc=np.mean)
        pv_by_D[D] = pv.sort_index(axis=0).sort_index(axis=1)
    common_ks = sorted(set.intersection(*(set(p.index) for p in pv_by_D.values())))
    common_taus = sorted(set.intersection(*(set(p.columns) for p in pv_by_D.values())))
    if not common_ks or not common_taus:
        raise ValueError(f"No common (k,tau) grid across dims for task {task}.")
    vecs = []
    for D in dims:
        pv = pv_by_D[D]
        vecs.append(_vectorize_grid(pv, common_ks, common_taus, standardize=standardize, rank=spearman))
    mat = np.corrcoef(np.vstack(vecs))
    corr_df = pd.DataFrame(mat, index=dims, columns=dims)
    return corr_df, dims


def main():
    df = load_combined()
    tasks = sorted(df["task"].unique())
    # Between-problem correlations (z-scored grids, Pearson)
    corr_df, ks, taus = corr_between_tasks(df, tasks, standardize=True, spearman=False)
    corr_csv = OUT_DIR / "transfer_corr_matrix.csv"
    corr_df.to_csv(corr_csv)
    fig_path = OUT_DIR / "transfer_corr_heatmap.png"
    plot_corr_heatmap(corr_df, fig_path)
    # Between-problem correlations (rank-based Spearman)
    corr_df_s, _, _ = corr_between_tasks(df, tasks, standardize=True, spearman=True)
    corr_csv_s = OUT_DIR / "transfer_corr_matrix_spearman.csv"
    corr_df_s.to_csv(corr_csv_s)
    fig_path_s = OUT_DIR / "transfer_corr_heatmap_spearman.png"
    plot_corr_heatmap(corr_df_s, fig_path_s)
    # Per-task correlations across trait sizes (z-scored Pearson)
    for t in tasks:
        try:
            cdf, dims = corr_across_dims_for_task(df, t, standardize=True, spearman=False)
        except Exception:
            continue
        out_csv = OUT_DIR / f"transfer_corr_by_dim_{t.lower().replace(' ','_')}.csv"
        out_fig = OUT_DIR / f"transfer_corr_by_dim_{t.lower().replace(' ','_')}.png"
        cdf.to_csv(out_csv)
        plot_corr_heatmap(cdf, out_fig)
        # Spearman version
        try:
            cdf_s, _ = corr_across_dims_for_task(df, t, standardize=True, spearman=True)
            out_csv_s = OUT_DIR / f"transfer_corr_by_dim_{t.lower().replace(' ','_')}_spearman.csv"
            out_fig_s = OUT_DIR / f"transfer_corr_by_dim_{t.lower().replace(' ','_')}_spearman.png"
            cdf_s.to_csv(out_csv_s)
            plot_corr_heatmap(cdf_s, out_fig_s)
        except Exception:
            pass
    # Also compute within-task rank-correlation (Spearman) between D splits if available
    print("Saved:")
    print(f"- correlation matrix: {corr_csv}")
    print(f"- heatmap (Pearson): {fig_path}")
    print(f"- heatmap (Spearman): {fig_path_s}")
    print("Common grid:")
    print("embed_dim:", ks)
    print("taus:", taus)
    print("Per-task D-by-D correlation files created (if dims available).")


if __name__ == "__main__":
    main()



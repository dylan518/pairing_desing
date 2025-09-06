# Pairing Design Experiments

## Quick start

```bash
python3 test.py
```
- Generates per-problem CSVs/heatmaps in `outputs/` and a combined CSV.

## Multiprocessing
- `test.py` uses multiple processes per (task, k, tau) to parallelize seeds.
- Seeds are derived per-setting to ensure fairness across grid points.

## Rigorous single-problem analysis
- For a high-replication sweep on one benchmark (default: Rastrigin):

```bash
python3 rigorous_single.py
```

Outputs in `outputs/`:
- `rigorous_<task>_grid_stats.csv` — mean/std/SE/CI for each (D, k, tau)
- `rigorous_<task>_best_k_by_D.csv` — winners per trait dimension
- `rigorous_<task>_best_k_linear.png` — best k vs D with linear fit
- `rigorous_<task>_linear_stats.csv` — fit coefficients and R²

## Global k heuristic analysis
```bash
python3 analyze_k_ratio.py
```
Produces:
- `analysis_k_ratio_summary.csv` — winners across all runs
- `analysis_k_global_linear.png/.csv` — global linear fit k = a·D + b
- `best_k_distribution*.{csv,png}`, `best_tau_distribution*.{csv,png}` — winner histograms

## Requirements
Install minimal dependencies:

```bash
pip3 install -r requirements.txt
```

- Python 3.10+ recommended.
- No external ML libs required.

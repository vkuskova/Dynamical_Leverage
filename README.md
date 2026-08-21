# Dynamical Leverage in Temporal Networks: Reproducibility 

Code and artifacts for the paper "Dynamical Leverage in Temporal Networks: \\
When Does Node Importance Depend on the Trajectory?" (Paper under review).

## Layout

```
code/       reference implementation of the leverage measure (E_j) and the
            average-controllability baseline
scripts/    figure and experiment scripts
results/    committed outputs: sweep artifacts and the synthetic grid, plus
            the paper's figures regenerated from them
```

## Path A — verify (seconds)

The committed CSVs in `results/` are the artifacts every number in the paper
is read from. The figure scripts regenerate the paper's figures from them
with no recomputation:

```
python scripts/make_fig_trade_scatter.py
python scripts/make_fig_departure_vs_advantage.py
```

- `results/clean_sweep.csv` — tgbn-trade 8-seed departure sweep
  (pooled gap 0.037 ± 0.012, realized 0.015 ± 0.007)
- `results/clean_rankings.csv` — per-seed and mean nation ranks
  (mean pairwise Spearman 0.844)
- `results/clean_validation.csv` — leverage vs. strength / volume /
  static-AC / perturbation impact per nation
- `results/phase_diagram_results.csv` — the 144-cell synthetic grid
  (Section 3.4); `1 - ej_acp` is the pooled departure, `d_pool` the
  fidelity advantage

## Path B — regenerate (minutes to hours)

`code/` contains the leverage measure (`dynamical_leverage.py`) and the
neural VAR estimator (`neural_var.py`) both sweeps use.

`scripts/make_phase_grid.py` regenerates the synthetic grid from scratch
(~5 min): a frozen two-regime nonlinear DGP over 4 structures × 3
nonlinearity × 3 persistence × 4 amplitude levels, 20 seeds per cell.
Point-estimate columns reproduce exactly across machines; the bootstrap CI
columns (`*_lo`, `*_hi`) use an unseeded resampler and reproduce to within
about ±0.005. No number in the paper derives from the CI columns.

The real-system sweeps regenerate every trade and polecat number:

```
pip install py-tgb torch
python scripts/fetch_data.py          # downloads both TGB edgelists to data/
python scripts/run_trade_sweep.py     # ~10 min: clean_sweep / clean_rankings /
                                      #   clean_validation (8 seeds)
python scripts/run_polecat_sweep.py   # ~15-20 min on GPU: valenced run
                                      #   (8 seeds x 2 valence codings)
```

The polecat script writes one accumulating artifact,
`results/polecat_sweep_results.csv`, with a `variant` column. Setting
`ABLATION = "role_split"` or `ABLATION = "collapsed"` at the top of the
script runs the valence-removal ablations; re-running a variant replaces
its rows and preserves the others, so the three invocations build a single
committed file.

The edgelists are not committed (the polecat edgelist has 3.56M rows);
`fetch_data.py` materializes them from the Temporal Graph Benchmark
distribution. Sweeps are seeded; departure numbers reproduce to the
precision reported in the paper.

## Environment

Python 3 with `numpy`, `scipy`, `pandas`, `matplotlib`; the real-system
sweeps additionally require `torch` and `py-tgb`.

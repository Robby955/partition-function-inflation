# Paper Results — Local Backup
All experimental data for "Partition Function Inflation in Hashed N-Gram Caches."

## Data Files
- **exp1_captured.csv** — 36 configurations (4 buckets × 9 alpha values), full precision
- **exp1b_captured.csv** — 27 per-order profile configurations (6 profiles × 4 buckets + 2 diagnostic)
- **exp3_captured.csv** — Real vs random vs clean collision control (3 conditions)
- **remap_multiseed.csv** — 8-seed remap control (mean=0.16401, std=0.00004)
- **z_measurements.csv** — Empirical partition function (E[log2 Z], Z_max) at 4 bucket sizes
- **stepwise_results.csv** — Step-wise normalization at 1M (orders 2-3 through 2-7)
- **bonus_captured.csv** — Supplementary experiments (131K/262K buckets, alpha=1000/5000, etc.)

## Additional Data Files
- **lambda_results.csv** — Lambda sweep results for normalization penalty estimation
- **multiseed_results.csv** — Multi-seed remap control experiment results
- **all_results_unified.csv** — Unified table of all experimental results

## Notes
- Original exp3 seed 54321 result from pod2 (in exp3_captured.csv)
- 8 remap seeds: 54321, 42, 12345, 99999, 2026, 314159, 7777, 415
- Z measurement logs use MEASURE_Z_EVERY=10 (~11% subsample)

## Sanity Checks (from dedicated runs)
- cache_off: 1.12963 BPB (neural-only baseline)
- exp4 identity: 0.75420131 BPB (Dirichlet and interpolation forms match to 8 dp)
- hash_sensitivity at 4M/alpha=50: default=0.75422, alt=0.74721 (gap=0.007)

## Provenance
- All CSVs extracted from pod logs with full precision
- Figure generation: `generate_paper_figures.py`
- Paper source: `latex/paper.tex`

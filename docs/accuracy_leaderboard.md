# Step 3: accuracy leaderboard

Use `scripts/build_accuracy_leaderboard.py` to turn detector/HMR and full-data motion-stratified evaluation artifacts into paper-facing accuracy tables.

## One-command path from a detector/HMR config

This runs the detector-measurement evaluator once per transition model, then writes a ranked leaderboard:

```bash
python scripts/build_accuracy_leaderboard.py \
  --eval-config configs/hmr_measurements.local.json \
  --method gaussian=gaussian_rw:numpy \
  --method adaptive=adaptive_gaussian_rw:numpy \
  --method cv=constant_velocity:numpy \
  --method mlp=mlp_delta:numpy \
  --method hist=history_mlp_delta:numpy \
  --method gru=gru_delta:numpy \
  --detector-dataset HMR-KIT-or-ACCAD \
  --output-dir runs/accuracy_leaderboard
```

To include a proposed model later, add another method spec, for example:

```bash
--method proposed=learned_proposal:numpy
```

## Aggregate already-computed runs

Use this when each transition model has already been evaluated into its own output directory:

```bash
python scripts/build_accuracy_leaderboard.py \
  --detector-run gaussian=runs/hmr_gaussian_rw \
  --detector-run adaptive=runs/hmr_adaptive_gaussian_rw \
  --detector-run cv=runs/hmr_constant_velocity \
  --detector-run mlp=runs/hmr_mlp_delta \
  --detector-run hist=runs/hmr_history_mlp_delta \
  --detector-run gru=runs/hmr_gru_delta \
  --motion-run ACCAD=runs/full_data_accad_artifact \
  --motion-run KIT=runs/full_data_kit_artifact \
  --output-dir runs/accuracy_leaderboard
```

## Outputs

The script writes:

- `accuracy_leaderboard.md` - paper-facing Markdown table.
- `accuracy_leaderboard.tex` - `booktabs` LaTeX table for `main.tex`.
- `accuracy_leaderboard.csv` - complete machine-readable table.
- `accuracy_leaderboard.json` - JSON table with row count, paper summary, sanity report, and comparison rows.
- `accuracy_leaderboard_paper_summary.*` - aggregate paper-facing summary across noise and occlusion conditions.
- `accuracy_leaderboard_sanity_report.*` - baseline coverage and duplicate-row checks.
- `accuracy_leaderboard_method_comparisons.csv` - paired method-vs-baseline comparisons over matched conditions.
- `accuracy_leaderboard_class_comparisons.csv` - paired method-class comparisons, such as causal online filters versus offline smoothers.
- `accuracy_leaderboard_comparison_report.json` - method and method-class comparisons with win rates, exact paired sign-test p-values, and bootstrap intervals.
- `accuracy_leaderboard_comparison_report.md` - readable comparison report, including causal-online versus offline-smoother comparisons.
- `accuracy_leaderboard_claim_candidates.*` - cautious within-benchmark claim checks derived from the class comparisons.
- `accuracy_leaderboard_selector_headroom.*` - diagnostic comparison of fixed persistence, fixed Gaussian RW, the noise-adaptive selector, the best post-hoc noise threshold, and an oracle persistence/Gaussian selector.

The ranking metric is `tracking_error_deg`; lower is better. For real detector/HMR outputs, the script adds `raw_measurement` and `persistence` baseline rows from the detector evaluation summary, then one filter row per transition-model run. The main leaderboard and paper summary include `method_class` so causal online filters, causal baselines, raw measurements, and offline smoothers remain visually distinct. Positive `improvement_vs_raw_deg` and `improvement_vs_persistence_deg` mean the method improved over those baselines. Positive `mean_improvement_deg` in the comparison report means the target method beat the stated baseline on matched conditions. The sign-test p-value is an exact two-sided paired sign test over matched conditions and ignores ties. Claim candidates are evidence labels for the current benchmark only; they should not be read as state-of-the-art claims without an external comparison.

Motion-stratified synthetic artifacts may also include `noise_adaptive_selector`, which uses the known synthetic measurement noise level to select either the Gaussian RW filter or deterministic persistence. This row is intended as a diagnostic policy for synthetic robustness grids, not as a detector/HMR transition model.

The `Accuracy Leaderboard` workflow validates these files, uploads them as a sanitized artifact,
and writes the claim candidates, comparison report, and sanity report into the GitHub Actions job
summary for quick inspection.

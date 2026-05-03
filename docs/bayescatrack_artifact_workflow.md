# BayesCaTrack artifact workflow

Long-running BayesCaTrack experiments should run in this public code repository. The private paper repository should consume frozen result artifacts and should not rerun the benchmark.

## Motion-stratified ACCAD artifact

After the motion-stratified evaluation runner is present, dispatch the **ACCAD Motion-Stratified Benchmark** workflow. For GitHub-hosted sample runs, leave `download_sample=true`. For a full private ACCAD checkout mounted on a self-hosted runner, set:

```text
runs_on_json = ["self-hosted", "<your-runner-label>"]
download_sample = false
data_root = /path/to/ACCAD
```

The workflow uploads an artifact named:

```text
bayescatrack-accad-motion-stratified-<run-number>-<sha>
```

The artifact contains:

```text
run_manifest.json
motion_stratified_validation.json
motion_stratified_private_accad_eval_summary.json
motion_stratified_private_accad_eval_summary.md
window_selection_report.json
window_manifest.csv
aggregate_benchmark_metrics_by_motion_bin.csv
aggregate_transition_metrics_by_motion_bin.csv
aggregate_method_means_by_motion_bin.csv
aggregate_method_means_by_noise_occlusion_motion.csv
benchmarks/**/first_results_summary.json
benchmarks/**/benchmark_metrics.csv
benchmarks/**/transition_metrics.csv
benchmarks/**/plots/*.svg
```

`run_manifest.json` records the source SHA, workflow run metadata, config hash/content, selected runtime package versions, and output file hashes. Keep this manifest beside every paper result snapshot.

## Repository boundary

The paper repository should:

1. download one selected artifact bundle from this workflow,
2. unpack it under `results/accad-motion-stratified/<snapshot>/`,
3. generate publication figures from the CSV/JSON files, and
4. commit the frozen result snapshot plus figure scripts.

It should not execute the AMASS/ACCAD filtering benchmark.

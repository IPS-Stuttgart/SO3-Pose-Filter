# Reproducing SO(3)^K pose-filtering results

This guide keeps generated data, trained checkpoints, and benchmark outputs outside git while recording enough metadata to rerun and audit an experiment.

## 1. Create an isolated environment

Use Python 3.11 or newer. The repository CI matrix currently exercises Python 3.11, 3.12, and 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2. Run the deterministic toy smoke experiment

The toy dataset is synthetic and intentionally small, so this path is suitable for CI, packaging checks, and new contributors.

```bash
python scripts/make_toy_amass.py --output data/tiny_amass --sequences 6 --frames 80
python scripts/run_experiment.py --config configs/example.json
python scripts/run_model_sweep.py --config configs/example.json --output runs/sweep
```

The smoke experiment writes `runs/example/summary.json`, CSV metrics, SVG plots, and `trajectory_preview.csv`. The sweep writes `runs/sweep/comparison_metrics.csv` and `runs/sweep/comparison_summary.json`.

## 3. Record a run manifest

After an experiment, write a manifest that records the config hash, git SHA, runtime package versions, and hashes for generated output files.

```bash
python scripts/write_experiment_manifest.py \
  --experiment-name toy-smoke \
  --config configs/example.json \
  --result-root runs/example \
  --output runs/example/run_manifest.json \
  --source-data-root data/tiny_amass
```

The manifest intentionally records local paths before publication. Use the packaging step below to redact private paths.

## 4. Create a public-safe artifact package

```bash
python scripts/package_paper_artifact.py \
  --result-root runs/example \
  --output-dir results/toy-smoke-public \
  --output-zip results/toy-smoke-public.zip
```

The package step copies public JSON, CSV, and SVG outputs; redacts path-like JSON fields and CSV columns; skips raw `.npz` motion data; and writes `paper_artifact_package_manifest.json` with SHA-256 hashes for copied files. The optional zip archive is written with deterministic entry timestamps and ordering.

## 5. Reproduce the bounded ACCAD benchmark path

For the bounded ACCAD workflow, provide an AMASS/SMPL-style ACCAD checkout and keep outputs under `runs/` or `results/`.

```bash
python scripts/prepare_amass_windows.py \
  --data-root /path/to/ACCAD \
  --output-dir data/accad_dynamic_segments \
  --report runs/accad_dynamic_segments_report.json \
  --manifest runs/accad_dynamic_segments_manifest.csv \
  --frame-rate 20 \
  --segment-frames 80 \
  --stride-frames 40 \
  --max-segments 48 \
  --selection balanced-motion \
  --max-per-file 2

python scripts/run_model_sweep.py \
  --config configs/accad_dynamic.example.json \
  --output runs/accad_dynamic_sweep \
  --models persistence gaussian_rw learned_delta mlp_delta history_mlp_delta gru_delta

python scripts/run_first_results_benchmark.py \
  --config configs/accad_dynamic_benchmark.example.json \
  --output runs/accad_dynamic_first_results
```

Then package the public subset:

```bash
python scripts/write_experiment_manifest.py \
  --experiment-name accad-dynamic-first-results \
  --config configs/accad_dynamic_benchmark.example.json \
  --result-root runs/accad_dynamic_first_results \
  --output runs/accad_dynamic_first_results/run_manifest.json \
  --source-data-root /path/to/ACCAD

python scripts/package_paper_artifact.py \
  --result-root runs/accad_dynamic_first_results \
  --output-dir results/accad-dynamic-first-results-public \
  --output-zip results/accad-dynamic-first-results-public.zip
```

## 6. Acceptance checks

For the toy smoke path, check that:

- `runs/example/summary.json` exists.
- `runs/sweep/comparison_metrics.csv` exists.
- `summary.json` reports 23 joints and 6 sequences.
- The filtered mean error is below both the raw observed and persistence mean errors.
- The public artifact package contains no raw `.npz` files and no local data paths.

For private-data runs, always inspect `paper_artifact_package_manifest.json` and the copied CSV/JSON files before publishing.

# SO(3)^K Motion Filtering Prototype

This workspace contains a small prototype for early AMASS/SMPL motion-filtering results.
It reads SMPL-style AMASS `.npz` files, converts local body joint axis-angle poses into SO(3) rotation
matrices, creates synthetic noisy/occluded measurements, and evaluates transition baselines plus a particle
filter. It also reports cheap smoothing baselines so the particle filter is compared against simple temporal
methods rather than only raw measurements.
Measurements can carry per-joint confidence scores, so detector outputs with soft joint reliability can
downweight uncertain observations instead of dropping them with a hard mask only.

The core SO(3) numerics use NumPy, and quaternion product-state distributions use PyRecEst as their
backend. The smoothing baselines also reuse PyRecEst utilities. The package source follows a
`src/pose_filter` layout.

## Quick Start

Generate a tiny synthetic AMASS-like dataset and run a smoke experiment:

```powershell
python -m pip install -e .
python scripts\make_toy_amass.py --output data\tiny_amass --sequences 6 --frames 80
python scripts\run_experiment.py --config configs\example.json
```

Outputs are written to `runs/example/`:

- `summary.json`
- `transition_metrics.csv`
- `filter_metrics.csv`
- `per_joint_metrics.csv`
- `temporal_metrics.csv`
- `ablation_metrics.csv`
- `robustness_metrics.csv`
- `plots/*.svg`
- `trajectory_preview.csv`

Run tests:

```powershell
python -m unittest discover -s tests
```

Compare all transition models from one config:

```powershell
python scripts\run_model_sweep.py --config configs\example.json --output runs\sweep
```

The sweep writes:

- `runs/sweep/comparison_metrics.csv`
- `runs/sweep/comparison_summary.json`
- one full experiment folder per transition model

Limit a sweep to selected transition models:

```powershell
python scripts\run_model_sweep.py --config configs\example.json --output runs\sweep --models persistence gaussian_rw
```

Run the first-results benchmark wrapper, which reports raw observations, deterministic persistence,
Gaussian random-walk particle filters, and the nonlinear MLP transition particle filters on one
noise/occlusion grid:

```powershell
python scripts\run_first_results_benchmark.py `
  --config configs\accad_dynamic_benchmark.example.json `
  --output runs\accad_dynamic_first_results
```

The benchmark writes:

- `benchmark_metrics.csv`
- `first_results_summary.json`
- `transition_metrics.csv`
- `plots/tracking_error_heatmap.svg`
- `plots/filter_vs_baselines.svg`

Benchmark outputs are local generated artifacts and are excluded from the repository.
Use `runs/` or `results/` for local result snapshots.

## Real AMASS Data

Point `data_root` in a config JSON file at an AMASS/SMPL-style directory containing `.npz` files with:

- `poses`: shape `[T, >=72]`, axis-angle SMPL pose parameters
- `mocap_framerate` or `mocap_frame_rate`: source frame rate

The prototype uses the 23 local body joints in `poses[:, 3:72]`, excluding global root orientation,
global translation, hands, and face.

Copy `configs/amass_small.example.json` to a local config and replace `data_root` with the real AMASS
directory. Keep generated real-data outputs under `runs/` or `results/`; they are ignored by git.

The ACCAD first-results benchmark workflow uses `configs/accad_first_results.example.json` and
`configs/accad_first_results_benchmark.example.json`. It downloads the OwnCloud ACCAD sample URL from the
`AMASS_ACCAD_SAMPLE` secret, falling back to the public share URL used for this project, chunks the motion
file into train/validation/test sequence windows, runs the PyRecEst-backed Gaussian random-walk filter, runs
a compact transition-model sweep, and runs the first-results benchmark wrapper. It uploads CSV, JSON, and
SVG artifacts. The workflow asserts that the filter beats raw synthetic observations and reports the
persistence rollout baseline under the configured moderate noise/occlusion setting. The benchmark workflow
is intentionally bounded to one downloaded sample, at most six selected windows, and four benchmark grid
points so pull requests do not scan or evaluate a full AMASS dataset.

For a local ACCAD run on a full dataset checkout, first select a bounded set of dynamic windows and then
run the dynamic benchmark config:

```powershell
python scripts\prepare_amass_windows.py `
  --data-root D:\Uni-Data\ACCAD `
  --output-dir data\accad_dynamic_segments `
  --report runs\accad_dynamic_segments_report.json `
  --manifest runs\accad_dynamic_segments_manifest.csv `
  --frame-rate 20 `
  --segment-frames 80 `
  --stride-frames 40 `
  --max-segments 48 `
  --selection top-motion `
  --max-per-file 2

python scripts\run_model_sweep.py `
  --config configs\accad_dynamic.example.json `
  --output runs\accad_dynamic_sweep `
  --models persistence gaussian_rw learned_delta mlp_delta

python scripts\run_first_results_benchmark.py `
  --config configs\accad_dynamic_benchmark.example.json `
  --output runs\accad_dynamic_first_results

python scripts\run_first_results_benchmark.py `
  --config configs\accad_dynamic_benchmark.example.json `
  --output runs\accad_dynamic_mlp_single_point `
  --methods raw persistence gaussian_rw pyrecest_pf mlp_delta `
  --noise-deg 10 `
  --occlusion-prob 0.25
```

`prepare_amass_windows.py` records `motion_deg_per_frame` for every selected segment so results can be
stratified by motion intensity. CI deliberately uses the same selector with `--max-files 1` and
`--max-segments 6`, so pull requests exercise the benchmark path without scanning or evaluating a full
AMASS dataset.

## PyRecEst Backend

The default experiment path uses the NumPy SO(3)^K particle filter. Set `"filter_backend": "pyrecest"`
in an experiment config to store particles in PyRecEst's `SO3ProductParticleFilter` while keeping the
same transition models, synthetic measurements, and output metrics. This backend uses scalar-last unit
quaternion states `(x, y, z, w)` on the upper hyperhemisphere `S^3_+` and converts back to rotation
matrices for evaluation:

```python
from pose_filter.quaternion import rotations_to_quaternions
from pose_filter.pyrecest_filter import run_pyrecest_particle_filter

quaternions = rotations_to_quaternions(rotations)  # [N, 23, 4], w >= 0
result = run_pyrecest_particle_filter(observations, mask, model, noise_sigma, 128, rng)
```

PyRecEst is a runtime dependency because the quaternion product-state distributions and the optional
PyRecEst particle filter backend are part of the package backend rather than an external script.

## Config Fields

Required fields:

- `data_root`
- `dataset_subset`
- `frame_rate`
- `num_joints`
- `noise_deg`
- `occlusion_prob`
- `num_particles`
- `transition_model`: `persistence`, `gaussian_rw`, `learned_delta`, or `mlp_delta`

Useful optional fields:

- `output_dir`
- `seed`
- `max_sequences`
- `min_frames`
- `train_fraction`
- `val_fraction`
- `rollout_horizon`
- `robustness_noise_deg`
- `robustness_occlusion_prob`
- `process_noise_deg`
- `filter_backend`: `numpy` or `pyrecest`
- `proposal_gain`
- `confidence_noise_std`
- `min_confidence`
- `smoother_ema_alpha`
- `smoother_chordal_window`
- `factorized_update`
- `resample_threshold`
- `ablation_particle_counts`
- `ablation_proposal_gains`
- `ablation_factorized_updates`
- `ablation_resample_thresholds`
- `mlp_hidden_dim`
- `mlp_epochs`
- `mlp_learning_rate`
- `mlp_weight_decay`
- `mlp_batch_size`
- `transition_checkpoint`
- `transition_load_checkpoint`
- `transition_save_checkpoint`

## Notes

`learned_delta` is a NumPy ridge-regression transition baseline. It predicts tangent-space SO(3) deltas
from the current pose and estimates residual noise for sampling. This keeps the first prototype runnable
without PyTorch while preserving the `sample_next` / `log_prob_next` interface expected by later neural
models.

`mlp_delta` is a nonlinear NumPy MLP transition baseline. It standardizes the current pose log-map, trains
a compact one-hidden-layer tanh network to predict the next tangent-space delta, estimates residual
per-joint variance, and supports `.npz` checkpoint save/load through `transition_checkpoint`. This keeps the
learned baseline CI-friendly while providing a stronger target than the linear ridge model before adding a
full PyTorch GRU.

Synthetic confidence values default to the original binary mask behavior when `confidence_noise_std` is
zero. Setting `confidence_noise_std > 0` samples observed-joint confidences in `[min_confidence, 1]`; these
scores scale both the guided proposal correction and the measurement likelihood.

The smoothing baselines are deterministic references:

- `smoother_ema`: causal per-joint exponential smoothing in the tangent space of the previous SO(3) estimate.
- `smoother_chordal`: offline centered-window chordal mean over visible observations.

The experiment outputs include research-oriented diagnostics beyond aggregate pose error:
observed-vs-occluded joint errors, per-joint errors, and temporal acceleration/jerk metrics for the raw
measurements, filtered estimate, persistence baseline, and ground truth.

`ablation_metrics.csv` varies one filter setting at a time around the configured baseline. It reports
particle-count, proposal-gain, factorized-update, and resampling-threshold rows so experiments can compare
the guided/factorized particle filter against simpler bootstrap settings such as `proposal_gain=0` or
`factorized_update=false`.

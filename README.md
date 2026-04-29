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

## Real AMASS Data

Point `data_root` in a config JSON file at an AMASS/SMPL-style directory containing `.npz` files with:

- `poses`: shape `[T, >=72]`, axis-angle SMPL pose parameters
- `mocap_framerate` or `mocap_frame_rate`: source frame rate

The prototype uses the 23 local body joints in `poses[:, 3:72]`, excluding global root orientation,
global translation, hands, and face.

Copy `configs/amass_small.example.json` to a local config and replace `data_root` with the real AMASS
directory. Keep generated real-data outputs under `runs/`; they are ignored by git.

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
- `transition_model`: `persistence`, `gaussian_rw`, or `learned_delta`

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

## Notes

`learned_delta` is a NumPy ridge-regression transition baseline. It predicts tangent-space SO(3) deltas
from the current pose and estimates residual noise for sampling. This keeps the first prototype runnable
without PyTorch while preserving the `sample_next` / `log_prob_next` interface expected by later neural
models.

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

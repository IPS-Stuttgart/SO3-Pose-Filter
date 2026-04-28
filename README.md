# SO(3)^K Motion Filtering Prototype

This workspace contains a small, dependency-light prototype for early AMASS/SMPL motion-filtering results.
It reads SMPL-style AMASS `.npz` files, converts local body joint axis-angle poses into SO(3) rotation
matrices, creates synthetic noisy/occluded measurements, and evaluates transition baselines plus a particle
filter.

The code intentionally uses only NumPy and the Python standard library. The package source follows a
PyRecEst-style `src/pose_filter` layout.

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
- `proposal_gain`

## Notes

`learned_delta` is a NumPy ridge-regression transition baseline. It predicts tangent-space SO(3) deltas
from the current pose and estimates residual noise for sampling. This keeps the first prototype runnable
without PyTorch while preserving the `sample_next` / `log_prob_next` interface expected by later neural
models.

The experiment outputs include research-oriented diagnostics beyond aggregate pose error:
observed-vs-occluded joint errors, per-joint errors, and temporal acceleration/jerk metrics for the raw
measurements, filtered estimate, persistence baseline, and ground truth.

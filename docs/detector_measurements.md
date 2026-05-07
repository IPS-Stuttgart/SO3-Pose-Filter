# Real detector and SMPL-fitting measurements

The synthetic experiment path is still useful for controlled noise and occlusion sweeps. For real detector or SMPL-fitting outputs, use the detector-measurement importer and evaluation path instead. This keeps real observations separate from synthetic corruption while reusing the same SO(3)^K particle-filter backends and transition models.

## Supported input layouts

The importer accepts `.npz` and `.json` files. A file should contain one pose sequence and may use any of these common pose encodings:

- rotation matrices: `observations`, `rotations`, `body_rotations`, `body_pose_rotmat`, `pose_rotmat`, `pred_rotmat`, `full_pose_rotmat`
- axis-angle: `poses`, `body_pose`, `smpl_body_pose`, `pred_body_pose`, `pose`, `full_pose`, `axis_angle`
- quaternions: `quaternions`, `body_quaternions`, `body_quat`, `pose_quat`, `pred_quat`

Full SMPL pose arrays with a root joint, such as `poses[:, :72]` or `[T, 24, 3]`, skip the global root orientation and keep the configured local body joints. Body-only arrays, such as `body_pose` shaped `[T, 69]` for 23 joints, are used directly.

Optional arrays:

- `mask`, `visible`, `visibility`, `valid`, or `valid_mask`: per-frame/per-joint visibility
- `confidence`, `joint_confidence`, `body_pose_confidence`, `scores`, or `keypoint_scores`: detector confidence in `[0, 1]`
- `joint_noise_sigma_rad` or `joint_noise_sigma_deg`: per-frame/per-joint measurement noise
- `mocap_framerate`, `mocap_frame_rate`, `fps`, or `frame_rate`: source frame rate used for downsampling

Confidence values outside `[0, 1]` are rejected unless you pass `--confidence-scale`; for example, use `--confidence-scale 100` for percentage scores.

## Standardize detector files

Convert raw detector or fitting outputs into standardized measurement bundles:

```powershell
python scripts\import_detector_measurements.py `
  --input data\raw_detector_outputs `
  --output-dir data\detector_measurements `
  --frame-rate 20 `
  --num-joints 23 `
  --noise-deg 10
```

Each output `.npz` contains:

- `observations`: `[T, J, 3, 3]` local SO(3) rotations
- `mask`: `[T, J]` visible/valid joints
- `confidence`: `[T, J]` detector confidence, zero for inactive joints
- `noise_sigma_rad`: scalar fallback measurement noise
- optional `joint_noise_sigma_rad`: `[T, J]` measurement-noise overrides

Use explicit keys when an upstream file uses non-standard names:

```powershell
python scripts\import_detector_measurements.py `
  --input data\smplifyx_outputs `
  --output-dir data\detector_measurements `
  --pose-key body_pose `
  --confidence-key joint_scores `
  --mask-key valid_mask
```

## Evaluate against ground-truth AMASS/SMPL sequences

Create a config from `configs/detector_measurements.example.json`, then point `data_root` at the ground-truth AMASS/SMPL files and `measurement_data_root` at either raw detector files or standardized bundles. File stems must match sequence names, for example `seq_000.npz` in both roots.

```powershell
python scripts\run_detector_measurement_eval.py `
  --config configs\detector_measurements.example.json
```

The evaluation writes:

- `detector_filter_metrics.csv`
- `detector_measurement_eval_summary.json`

The metrics include observed detector error, confidence-weighted observed error, filtered error, observed-vs-occluded joint errors, persistence baseline error, effective sample size, resampling rate, and particle spread.

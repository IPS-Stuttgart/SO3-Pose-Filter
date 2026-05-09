# HMR/HPS measurement pipeline

This pipeline converts real human-mesh-recovery or human-pose-and-shape outputs into the standardized SO(3)^K measurement bundles consumed by `run_detector_measurement_eval.py`.

Use it when the upstream method writes nested or non-flat outputs, for example WHAM/GVHMR dictionaries with `smpl_params_global.body_pose` or `smpl_params_incam.body_pose`, TokenHMR/HMR-style `pred_rotmat`, or AMASS-like `poses` arrays.

## Convert HMR outputs

```powershell
python scripts\import_hmr_measurements.py `
  --input data\raw_hmr_outputs `
  --output-dir data\hmr_measurements `
  --pose-frame global `
  --frame-rate 20 `
  --num-joints 23 `
  --noise-deg 8
```

Supported input suffixes are `.npz`, `.json`, `.pkl`, `.pickle`, `.pt`, and `.pth`. Loading `.pt` / `.pth` requires the optional torch extra. Pickle files and legacy or full-pickle PyTorch checkpoints are disabled by default because deserializing them can execute code; add `--allow-unsafe-deserialization` only for trusted local HMR outputs.

The converter accepts these pose layouts:

- body axis-angle: `body_pose`, `pred_body_pose`, `smpl_body_pose`, `pose_body`, `bodypose`
- full/root-including axis-angle: `poses`, `pose`, `pred_pose`, `full_pose`, `fullpose`, `smpl_pose`
- rotation matrices: `body_pose_rotmat`, `pred_body_rotmat`, `pred_rotmat`, `rotmat`, `rotmats`

If both camera-space and world/global branches are present, `--pose-frame global` prefers keys such as `smpl_params_global.body_pose`; `--pose-frame incam` prefers keys such as `smpl_params_incam.body_pose`. `--pose-frame auto` defaults to global/world branches when it can identify them.

Many current HMR systems export 21 SMPL-X-style body joints, while this project evaluates 23 local SMPL body joints. By default, `import_hmr_measurements.py` pads missing joints with identity rotations and marks those padded joints inactive in `mask` and `confidence`. Use `--no-pad-missing-joints` if you want such files to fail instead.

## Evaluate the standardized measurements

After conversion, point the detector-measurement evaluator at the standardized directory:

```powershell
python scripts\run_detector_measurement_eval.py `
  --config configs\hmr_measurements.example.json
```

The file stems must match the ground-truth AMASS sequence names. If an HMR output contains several person tracks, the converter writes names like `sequence_0`, `sequence_1`, etc.; rename or configure your ground-truth segments accordingly before evaluation.

The output metric table reports raw HMR measurement error, filtered error, persistence error, observed-vs-occluded joint error, effective sample size, resampling rate, and particle spread. For paper-facing accuracy claims, compare at least raw HMR, persistence, Gaussian random-walk PF, history MLP/GRU PF, and the proposed model on the same converted measurement directory.

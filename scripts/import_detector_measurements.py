from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_filter.detector_import import (  # noqa: E402
    find_detector_measurement_files,
    load_detector_measurement,
    save_imported_measurements,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert detector or SMPL-fitting outputs to standardized SO(3)^K measurement .npz bundles."
    )
    parser.add_argument("--input", required=True, type=Path, help="Detector output file or directory containing .npz/.json files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for standardized measurement .npz files.")
    parser.add_argument("--dataset-subset", default="", help="Optional substring filter when --input is a directory.")
    parser.add_argument("--frame-rate", type=int, default=20, help="Target frame rate after source-fps downsampling.")
    parser.add_argument("--num-joints", type=int, default=23, help="Number of local SMPL body joints to import.")
    parser.add_argument("--noise-deg", type=float, default=10.0, help="Fallback global measurement noise in degrees.")
    parser.add_argument("--pose-key", help="Override pose key, e.g. body_pose, poses, pred_rotmat, or quaternions.")
    parser.add_argument("--mask-key", help="Override visibility/mask key.")
    parser.add_argument("--confidence-key", help="Override detector confidence key.")
    parser.add_argument("--joint-noise-key", help="Override per-joint measurement-noise key.")
    parser.add_argument("--confidence-scale", type=float, default=1.0, help="Divide confidence values by this value before validation.")
    parser.add_argument("--quaternion-order", choices=("xyzw", "wxyz"), default="xyzw", help="Quaternion component order.")
    args = parser.parse_args()

    files = find_detector_measurement_files(args.input, args.dataset_subset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in files:
        measurement = load_detector_measurement(
            source,
            frame_rate=args.frame_rate,
            num_joints=args.num_joints,
            noise_deg=args.noise_deg,
            pose_key=args.pose_key,
            mask_key=args.mask_key,
            confidence_key=args.confidence_key,
            joint_noise_key=args.joint_noise_key,
            confidence_scale=args.confidence_scale,
            quaternion_order=args.quaternion_order,
        )
        target = args.output_dir / f"{measurement.name}.npz"
        save_imported_measurements(target, measurement)
        rows.append(
            {
                "source": str(source),
                "output": str(target),
                "sequence": measurement.name,
                "frames": int(measurement.observations.shape[0]),
                "joints": int(measurement.observations.shape[1]),
                "observed_joint_fraction": float(measurement.mask.mean()),
                "mean_confidence": float(measurement.confidence[measurement.mask].mean()) if measurement.mask.any() else float("nan"),
                "measurement_source": measurement.source,
            }
        )
    print(json.dumps({"converted": len(rows), "files": rows}, indent=2))


if __name__ == "__main__":
    main()

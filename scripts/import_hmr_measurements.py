from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_filter.detector_import import save_imported_measurements  # noqa: E402
from pose_filter.hmr_measurements import find_hmr_measurement_files, load_hmr_measurements  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert WHAM/GVHMR/TokenHMR-style HMR outputs to standardized SO(3)^K measurement .npz bundles."
    )
    parser.add_argument("--input", required=True, type=Path, help="HMR output file or directory containing .npz/.json/.pkl/.pt files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for standardized measurement .npz files.")
    parser.add_argument("--dataset-subset", default="", help="Optional substring filter when --input is a directory.")
    parser.add_argument("--frame-rate", type=int, default=20, help="Target frame rate after source-fps downsampling.")
    parser.add_argument("--num-joints", type=int, default=23, help="Number of local body joints expected by the SO(3)^K filter.")
    parser.add_argument("--noise-deg", type=float, default=10.0, help="Fallback global measurement noise in degrees.")
    parser.add_argument(
        "--pose-frame",
        choices=("auto", "global", "incam"),
        default="auto",
        help="Prefer world/global or camera/local HMR pose branches when both are present.",
    )
    parser.add_argument("--confidence-scale", type=float, default=1.0, help="Divide confidence values by this value before validation.")
    parser.add_argument(
        "--no-pad-missing-joints",
        action="store_true",
        help="Reject 21-joint HMR body-pose outputs instead of padding missing local joints as inactive identity rotations.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for quick smoke conversions.")
    args = parser.parse_args()

    files = find_hmr_measurement_files(args.input, args.dataset_subset)
    if args.max_files is not None:
        files = files[: args.max_files]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in files:
        for measurement in load_hmr_measurements(
            source,
            frame_rate=args.frame_rate,
            num_joints=args.num_joints,
            noise_deg=args.noise_deg,
            pose_frame=args.pose_frame,
            confidence_scale=args.confidence_scale,
            pad_missing_joints=not args.no_pad_missing_joints,
        ):
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

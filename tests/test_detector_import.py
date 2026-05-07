from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from _path import SRC  # noqa: F401
from pose_filter.detector_evaluation import load_detector_eval_config, run_detector_measurement_eval
from pose_filter.detector_import import (
    load_detector_measurement,
    load_detector_measurement_dataset,
    save_imported_measurements,
)


def _toy_poses(frames: int = 48) -> np.ndarray:
    poses = np.zeros((frames, 156), dtype=np.float64)
    t = np.linspace(0.0, 1.0, frames)
    for joint in range(23):
        start = 3 + joint * 3
        poses[:, start] = 0.08 * np.sin(2.0 * np.pi * t * (joint % 4 + 1))
        poses[:, start + 1] = 0.04 * np.cos(2.0 * np.pi * t)
    return poses


def _write_truth(path: Path, frames: int = 48, fps: float = 60.0) -> np.ndarray:
    poses = _toy_poses(frames)
    np.savez(path, poses=poses, mocap_framerate=np.asarray(fps))
    return poses


def _write_detector(path: Path, poses: np.ndarray, fps: float = 60.0) -> None:
    body_pose = poses[:, 3:72].copy()
    body_pose[:, 0] += 0.02
    confidence = np.ones((poses.shape[0], 23), dtype=np.float64)
    confidence[6, 3] = 0.0
    confidence[:, 7] = 0.5
    mask = np.ones((poses.shape[0], 23), dtype=bool)
    mask[6, 5] = False
    np.savez(
        path,
        body_pose=body_pose,
        confidence=confidence,
        mask=mask,
        mocap_framerate=np.asarray(fps),
        joint_noise_sigma_deg=np.full((poses.shape[0], 23), 6.0, dtype=np.float64),
    )


class DetectorImportTests(unittest.TestCase):
    def test_loads_smpl_body_pose_confidence_mask_and_standardized_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            poses = _toy_poses(48)
            detector_path = root / "seq.npz"
            _write_detector(detector_path, poses)

            measurement = load_detector_measurement(detector_path, frame_rate=20, num_joints=23, noise_deg=7.0)

            self.assertEqual(measurement.observations.shape, (16, 23, 3, 3))
            self.assertEqual(measurement.mask.shape, (16, 23))
            self.assertEqual(measurement.confidence.shape, (16, 23))
            self.assertEqual(measurement.joint_noise_sigma_rad.shape, (16, 23))
            self.assertEqual(measurement.source, "detector:body_pose")
            self.assertFalse(bool(measurement.mask[2, 3]))
            self.assertFalse(bool(measurement.mask[2, 5]))
            self.assertAlmostEqual(float(measurement.confidence[0, 7]), 0.5)

            standardized = root / "standardized" / "seq.npz"
            save_imported_measurements(standardized, measurement)
            reloaded = load_detector_measurement(standardized, frame_rate=20, num_joints=23, noise_deg=1.0)

            self.assertEqual(reloaded.observations.shape, measurement.observations.shape)
            self.assertTrue(np.array_equal(reloaded.mask, measurement.mask))
            self.assertTrue(np.allclose(reloaded.confidence, measurement.confidence))
            self.assertAlmostEqual(float(np.degrees(reloaded.noise_sigma_rad)), 7.0)

            dataset = load_detector_measurement_dataset(root / "standardized", "", 20, 23, noise_deg=1.0)
            self.assertEqual(set(dataset), {"seq"})

    def test_detector_measurement_eval_runs_against_truth_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            truth_root = root / "truth"
            detector_root = root / "detector"
            truth_root.mkdir()
            detector_root.mkdir()
            for idx in range(4):
                poses = _write_truth(truth_root / f"seq_{idx}.npz", frames=48 + idx * 3)
                _write_detector(detector_root / f"seq_{idx}.npz", poses)

            config_path = root / "detector_eval.json"
            output_dir = root / "runs"
            config_path.write_text(
                json.dumps(
                    {
                        "data_root": str(truth_root),
                        "dataset_subset": "",
                        "measurement_data_root": str(detector_root),
                        "frame_rate": 20,
                        "num_joints": 23,
                        "measurement_noise_deg": 7.0,
                        "num_particles": 8,
                        "transition_model": "persistence",
                        "filter_backend": "numpy",
                        "output_dir": str(output_dir),
                        "seed": 4,
                        "max_sequences": 4,
                        "min_frames": 8,
                        "train_fraction": 0.5,
                        "val_fraction": 0.25,
                        "proposal_gain": 0.1,
                        "factorized_update": True,
                        "resample_threshold": 0.5,
                    }
                ),
                encoding="utf-8",
            )

            summary = run_detector_measurement_eval(load_detector_eval_config(config_path))

            self.assertEqual(summary["row_count"], 1)
            self.assertEqual(summary["measurement_count"], 4)
            self.assertIn("filter_error_deg", summary["means"])
            metrics_path = output_dir / "detector_filter_metrics.csv"
            self.assertTrue(metrics_path.exists())
            with metrics_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["measurement_source"], "detector:body_pose")
            self.assertLess(float(rows[0]["observed_joint_fraction"]), 1.0)
            self.assertTrue((output_dir / "detector_measurement_eval_summary.json").exists())


if __name__ == "__main__":
    unittest.main()

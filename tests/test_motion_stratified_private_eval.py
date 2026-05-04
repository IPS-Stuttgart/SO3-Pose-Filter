from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from _path import SRC  # noqa: F401
from run_motion_stratified_private_accad_eval import run_motion_stratified_private_accad_eval


def _write_scaled_motion(path: Path, scale: float, frames: int = 90, fps: float = 60.0) -> None:
    poses = np.zeros((frames, 156), dtype=np.float64)
    t = np.linspace(0.0, 1.0, frames)
    for joint in range(23):
        start = 3 + joint * 3
        poses[:, start] = scale * np.sin(2.0 * np.pi * t * (joint % 3 + 1))
        poses[:, start + 1] = 0.5 * scale * np.cos(2.0 * np.pi * t)
    np.savez(path, poses=poses, mocap_framerate=np.asarray(fps))


class MotionStratifiedPrivateEvalTests(unittest.TestCase):
    def test_motion_stratified_private_eval_writes_motion_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            _write_scaled_motion(raw_dir / "low.npz", scale=0.002)
            _write_scaled_motion(raw_dir / "medium.npz", scale=0.05)
            _write_scaled_motion(raw_dir / "high.npz", scale=0.2)

            output_dir = root / "motion_stratified_eval"
            summary = run_motion_stratified_private_accad_eval(
                {
                    "source_data_root": str(raw_dir),
                    "output_dir": str(output_dir),
                    "frame_rate": 20,
                    "num_joints": 23,
                    "segment_frames": 12,
                    "stride_frames": 6,
                    "max_segments": 3,
                    "selection": "balanced-motion",
                    "max_per_file": 1,
                    "noise_deg": 5.0,
                    "occlusion_prob": 0.0,
                    "num_particles": 8,
                    "benchmark_num_particles": [8],
                    "benchmark_seeds": [3],
                    "benchmark_methods": [
                        "raw",
                        "persistence",
                        "constant_velocity",
                        "gaussian_rw",
                    ],
                    "benchmark_noise_deg": [5.0],
                    "benchmark_occlusion_prob": [0.0],
                    "transition_model": "gaussian_rw",
                    "max_sequences": 3,
                    "min_frames": 8,
                    "train_fraction": 0.5,
                    "val_fraction": 0.25,
                    "rollout_horizon": 5,
                    "process_noise_deg": 3.0,
                    "proposal_gain": 0.2,
                    "factorized_update": True,
                    "resample_threshold": 0.5,
                }
            )

            self.assertTrue(summary["motion_bin_counts"])
            self.assertIn("low_motion", summary["motion_bin_counts"])
            self.assertIn("medium_motion", summary["motion_bin_counts"])
            self.assertIn("high_motion", summary["motion_bin_counts"])
            self.assertTrue(summary["method_means_by_motion_bin"])
            self.assertTrue(summary["robustness_summary_by_motion_bin"])
            self.assertTrue(summary["transition_tracking_diagnostics_by_motion_bin"])
            self.assertTrue(
                (output_dir / "aggregate_benchmark_metrics_by_motion_bin.csv").exists()
            )
            self.assertTrue(
                (output_dir / "aggregate_transition_metrics_by_motion_bin.csv").exists()
            )
            self.assertTrue(
                (output_dir / "aggregate_transition_means_by_motion_bin.csv").exists()
            )
            self.assertTrue(
                (output_dir / "aggregate_method_means_by_motion_bin.csv").exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "aggregate_method_means_by_noise_occlusion_motion.csv"
                ).exists()
            )
            self.assertTrue(
                (output_dir / "robustness_summary_by_motion_bin.csv").exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "transition_tracking_diagnostics_by_motion_bin.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "motion_stratified_private_accad_eval_summary.md"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()

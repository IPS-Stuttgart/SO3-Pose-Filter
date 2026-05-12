from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from _path import SRC  # noqa: F401
from run_first_results_benchmark import run_first_results_benchmark

from pose_filter.data import PoseSequence
from pose_filter.so3 import axis_angle_to_matrix, mean_joint_distance_deg
from pose_filter.transitions import AdaptiveGaussianRandomWalkTransition


def _linear_rotation_sequence(
    frames: int = 12,
    num_joints: int = 23,
    step_rad: float = 0.05,
) -> np.ndarray:
    base_velocity = np.array([step_rad, 0.0, 0.0], dtype=np.float64)
    joint_scale = np.ones(num_joints, dtype=np.float64)
    times = np.arange(frames, dtype=np.float64)[:, None, None]
    axis_angle = times * joint_scale[None, :, None] * base_velocity[None, None, :]
    return axis_angle_to_matrix(axis_angle)


def _write_toy(path: Path, frames: int = 45, fps: float = 60.0) -> None:
    poses = np.zeros((frames, 156), dtype=np.float64)
    t = np.linspace(0.0, 1.0, frames)
    for joint in range(23):
        start = 3 + joint * 3
        poses[:, start] = 0.1 * np.sin(2.0 * np.pi * t * (joint % 3 + 1))
        poses[:, start + 1] = 0.05 * np.cos(2.0 * np.pi * t)
    np.savez(path, poses=poses, mocap_framerate=np.asarray(fps))


class AdaptiveGaussianRandomWalkTests(unittest.TestCase):
    def test_low_motion_gate_uses_persistence_mean(self) -> None:
        rotations = _linear_rotation_sequence(step_rad=0.05)
        seq = PoseSequence(
            name="linear",
            rotations=rotations,
            source_fps=20.0,
            frame_rate=20,
        )
        model = AdaptiveGaussianRandomWalkTransition.fit(
            [seq],
            motion_threshold_deg=1.5,
            low_motion_process_noise_deg=0.25,
        )

        pred = model.deterministic_next_from_history([seq.rotations[0], seq.rotations[0]])

        self.assertLess(mean_joint_distance_deg(seq.rotations[0], pred), 1e-9)

    def test_high_motion_gate_uses_gaussian_rw_mean(self) -> None:
        rotations = _linear_rotation_sequence(step_rad=0.05)
        seq = PoseSequence(
            name="linear",
            rotations=rotations,
            source_fps=20.0,
            frame_rate=20,
        )
        model = AdaptiveGaussianRandomWalkTransition.fit(
            [seq],
            motion_threshold_deg=1.5,
            low_motion_process_noise_deg=0.25,
        )

        pred = model.deterministic_next_from_history([seq.rotations[1], seq.rotations[2]])

        self.assertLess(mean_joint_distance_deg(seq.rotations[3], pred), 1e-6)
        samples = model.sample_next_from_history(
            [seq.rotations[1], seq.rotations[2]],
            np.random.default_rng(1),
        )
        self.assertEqual(samples.shape, (23, 3, 3))

    def test_first_results_benchmark_accepts_adaptive_method(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for idx in range(4):
                _write_toy(root / f"seq_{idx}.npz", frames=60 + idx * 3)

            summary = run_first_results_benchmark(
                {
                    "data_root": str(root),
                    "dataset_subset": "",
                    "frame_rate": 20,
                    "num_joints": 23,
                    "noise_deg": 5.0,
                    "occlusion_prob": 0.25,
                    "num_particles": 12,
                    "transition_model": "gaussian_rw",
                    "seed": 13,
                    "max_sequences": 4,
                    "min_frames": 10,
                    "process_noise_deg": 3.0,
                    "adaptive_motion_threshold_deg": 1.5,
                    "adaptive_low_motion_process_noise_deg": 0.25,
                    "proposal_gain": 0.2,
                    "factorized_update": True,
                    "resample_threshold": 0.5,
                },
                root / "benchmark",
                methods=("raw", "persistence", "adaptive_gaussian_rw", "gaussian_rw"),
                noise_grid=[5.0],
                occlusion_grid=[0.0],
            )

            self.assertIn("adaptive_gaussian_rw", summary["means_by_method"])
            self.assertEqual(summary["row_count"], 4)

    def test_noise_adaptive_selector_switches_by_noise_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for idx in range(4):
                _write_toy(root / f"seq_{idx}.npz", frames=60 + idx * 3)

            output_dir = root / "benchmark"
            summary = run_first_results_benchmark(
                {
                    "data_root": str(root),
                    "dataset_subset": "",
                    "frame_rate": 20,
                    "num_joints": 23,
                    "noise_deg": 5.0,
                    "occlusion_prob": 0.25,
                    "num_particles": 12,
                    "transition_model": "gaussian_rw",
                    "seed": 13,
                    "max_sequences": 4,
                    "min_frames": 10,
                    "process_noise_deg": 3.0,
                    "noise_adaptive_selector_threshold_deg": 10.0,
                    "proposal_gain": 0.2,
                    "factorized_update": True,
                    "resample_threshold": 0.5,
                },
                output_dir,
                methods=("raw", "persistence", "gaussian_rw", "noise_adaptive_selector"),
                noise_grid=[5.0, 20.0],
                occlusion_grid=[0.0],
            )

            self.assertIn("noise_adaptive_selector", summary["means_by_method"])
            self.assertEqual(summary["row_count"], 8)
            with (output_dir / "benchmark_metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            selector_rows = {float(row["noise_deg"]): row for row in rows if row["method"] == "noise_adaptive_selector"}
            self.assertEqual(selector_rows[5.0]["source_metric"], "gaussian_rw:filter_error_deg")
            self.assertEqual(selector_rows[20.0]["source_metric"], "persistence:persistence_error_deg")


if __name__ == "__main__":
    unittest.main()

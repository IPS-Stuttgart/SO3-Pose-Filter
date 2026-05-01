from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from _path import SRC  # noqa: F401
from pose_filter.constant_velocity import ConstantVelocityTransition
from pose_filter.data import PoseSequence
from pose_filter.so3 import axis_angle_to_matrix, mean_joint_distance_deg
from run_first_results_benchmark import run_first_results_benchmark


def _linear_rotation_sequence(frames: int = 12, num_joints: int = 23) -> np.ndarray:
    base_velocity = np.array([0.025, -0.01, 0.015], dtype=np.float64)
    joint_scale = np.linspace(0.5, 1.5, num_joints, dtype=np.float64)
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


class ConstantVelocityTests(unittest.TestCase):
    def test_constant_velocity_uses_previous_delta(self) -> None:
        rotations = _linear_rotation_sequence()
        seq = PoseSequence(
            name="linear",
            rotations=rotations,
            source_fps=20.0,
            frame_rate=20,
        )
        model = ConstantVelocityTransition.fit([seq])

        pred = model.deterministic_next_from_history(
            [seq.rotations[1], seq.rotations[2]]
        )

        self.assertLess(mean_joint_distance_deg(seq.rotations[3], pred), 1e-6)
        self.assertEqual(model.sample_next(seq.rotations[0], np.random.default_rng(1), n_samples=3).shape, (3, 23, 3, 3))

    def test_first_results_benchmark_accepts_constant_velocity_method(self) -> None:
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
                    "proposal_gain": 0.2,
                    "factorized_update": True,
                    "resample_threshold": 0.5,
                },
                root / "benchmark",
                methods=("raw", "persistence", "constant_velocity", "gaussian_rw"),
                noise_grid=[5.0],
                occlusion_grid=[0.0],
            )

            self.assertIn("constant_velocity", summary["means_by_method"])
            self.assertEqual(summary["row_count"], 4)


if __name__ == "__main__":
    unittest.main()

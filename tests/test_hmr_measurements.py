from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from _path import SRC  # noqa: F401
from pose_filter.hmr_measurements import load_hmr_measurements
from pose_filter.so3 import axis_angle_to_matrix


def _body_pose(frames: int, joints: int) -> np.ndarray:
    pose = np.zeros((frames, joints, 3), dtype=np.float64)
    t = np.linspace(0.0, 1.0, frames)
    for joint in range(joints):
        pose[:, joint, 0] = 0.05 * np.sin(2.0 * np.pi * t * (joint % 3 + 1))
        pose[:, joint, 1] = 0.03 * np.cos(2.0 * np.pi * t)
    return pose


class HMRMeasurementTests(unittest.TestCase):
    def test_loads_nested_gvhmr_style_body_pose_and_pads_21_joints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "seq.pkl"
            body_pose = _body_pose(7, 21).reshape(7, 63)
            confidence = np.ones((7, 21), dtype=np.float64)
            confidence[2, 5] = 0.0
            payload = {
                "fps": 30.0,
                "confidence": confidence,
                "smpl_params_global": {
                    "global_orient": np.zeros((7, 3), dtype=np.float64),
                    "body_pose": body_pose,
                    "transl": np.zeros((7, 3), dtype=np.float64),
                },
                "smpl_params_incam": {
                    "body_pose": np.zeros((7, 63), dtype=np.float64),
                },
            }
            with path.open("wb") as handle:
                pickle.dump(payload, handle)

            [measurement] = load_hmr_measurements(path, frame_rate=10, num_joints=23, pose_frame="global", noise_deg=8.0)

            self.assertEqual(measurement.observations.shape, (3, 23, 3, 3))
            self.assertEqual(measurement.mask.shape, (3, 23))
            self.assertEqual(measurement.confidence.shape, (3, 23))
            self.assertTrue(np.all(measurement.mask[:, :21] == (measurement.confidence[:, :21] > 0.0)))
            self.assertTrue(np.all(~measurement.mask[:, 21:]))
            self.assertTrue(np.allclose(measurement.confidence[:, 21:], 0.0))
            self.assertIn("smpl_params_global.body_pose", measurement.source)
            self.assertAlmostEqual(float(np.degrees(measurement.noise_sigma_rad)), 8.0)

    def test_loads_full_24_joint_rotmat_and_skips_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tokenhmr_like.npz"
            axis_angle = _body_pose(5, 24)
            axis_angle[:, 0, 0] = 1.2  # root; should not become local body joint 0
            axis_angle[:, 1, 0] = 0.4
            rotmat = axis_angle_to_matrix(axis_angle)
            np.savez(path, pred_rotmat=rotmat, frame_rate=np.asarray(20.0), scores=np.ones((5, 24), dtype=np.float64))

            [measurement] = load_hmr_measurements(path, frame_rate=20, num_joints=23, pose_frame="auto")

            self.assertEqual(measurement.observations.shape, (5, 23, 3, 3))
            expected_first_body_joint = axis_angle_to_matrix(axis_angle[:, 1:2])[:, 0]
            self.assertTrue(np.allclose(measurement.observations[:, 0], expected_first_body_joint))
            self.assertTrue(np.all(measurement.mask))
            self.assertIn("pred_rotmat", measurement.source)


if __name__ == "__main__":
    unittest.main()

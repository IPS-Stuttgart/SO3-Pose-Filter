from __future__ import annotations

import unittest

import numpy as np

from pose_filter.smoothing import (
    run_baseline_smoothers,
    sliding_chordal_mean_smoother,
    tangent_exponential_smoother,
)
from pose_filter.so3 import axis_angle_to_matrix, geodesic_distance


class SmoothingTests(unittest.TestCase):
    def test_tangent_ema_alpha_one_tracks_visible_observations(self) -> None:
        rotations = axis_angle_to_matrix(
            np.array(
                [
                    [[0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.2]],
                    [[0.0, 0.0, 0.4]],
                ],
                dtype=np.float64,
            )
        )
        mask = np.array([[True], [True], [False]])

        smoothed = tangent_exponential_smoother(rotations, mask, alpha=1.0)

        self.assertLess(float(geodesic_distance(smoothed[0, 0], rotations[0, 0])), 1e-8)
        self.assertLess(float(geodesic_distance(smoothed[1, 0], rotations[1, 0])), 1e-8)
        self.assertLess(float(geodesic_distance(smoothed[2, 0], rotations[1, 0])), 1e-8)

    def test_sliding_chordal_smoother_returns_valid_shape(self) -> None:
        rng = np.random.default_rng(2)
        rotations = axis_angle_to_matrix(rng.normal(0.0, 0.2, size=(6, 3, 3)))
        mask = np.ones((6, 3), dtype=bool)
        mask[2:4, 1] = False

        smoothed = sliding_chordal_mean_smoother(rotations, mask, window=3)

        self.assertEqual(smoothed.shape, rotations.shape)
        identity = np.broadcast_to(np.eye(3), smoothed.shape)
        determinants = np.linalg.det(smoothed)
        self.assertTrue(np.allclose(smoothed @ np.swapaxes(smoothed, -1, -2), identity))
        self.assertTrue(np.allclose(determinants, 1.0))

    def test_run_baseline_smoothers_reports_expected_keys(self) -> None:
        rotations = axis_angle_to_matrix(np.zeros((4, 2, 3)))
        mask = np.ones((4, 2), dtype=bool)

        outputs = run_baseline_smoothers(rotations, mask)

        self.assertEqual(set(outputs), {"smoother_ema", "smoother_chordal"})
        for estimates in outputs.values():
            self.assertEqual(estimates.shape, rotations.shape)


if __name__ == "__main__":
    unittest.main()

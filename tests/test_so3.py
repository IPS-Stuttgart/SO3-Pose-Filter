from __future__ import annotations

import unittest

import numpy as np
from _path import SRC  # noqa: F401
from pose_filter.so3 import (
    axis_angle_to_matrix,
    geodesic_distance,
    matrix_to_axis_angle,
)


class SO3Tests(unittest.TestCase):
    def test_axis_angle_roundtrip(self) -> None:
        rng = np.random.default_rng(1)
        rotvec = rng.normal(0.0, 0.4, size=(32, 23, 3))
        rotations = axis_angle_to_matrix(rotvec)
        recovered = matrix_to_axis_angle(rotations)
        self.assertLess(float(np.max(np.abs(rotvec - recovered))), 1e-6)

    def test_axis_angle_roundtrip_for_tiny_rotations(self) -> None:
        rotvec = np.asarray(
            [
                [1e-8, 0.0, 0.0],
                [0.0, -1e-7, 0.0],
                [0.0, 0.0, 5e-7],
            ],
            dtype=np.float64,
        )
        rotations = axis_angle_to_matrix(rotvec)
        recovered = matrix_to_axis_angle(rotations)
        self.assertLess(float(np.max(np.abs(rotvec - recovered))), 1e-12)

    def test_axis_angle_roundtrip_near_pi_rotation(self) -> None:
        rotvec = np.array([0.0, 0.0, np.pi - 1e-6])
        rotations = axis_angle_to_matrix(rotvec)
        recovered = matrix_to_axis_angle(rotations)
        recovered_rotations = axis_angle_to_matrix(recovered)
        self.assertLess(float(geodesic_distance(rotations, recovered_rotations)), 1e-8)

    def test_geodesic_identity_zero(self) -> None:
        rotations = axis_angle_to_matrix(np.zeros((5, 23, 3)))
        dist = geodesic_distance(rotations, rotations)
        self.assertLess(float(np.max(dist)), 1e-8)

    def test_geodesic_known_angle(self) -> None:
        a = axis_angle_to_matrix(np.zeros((3,)))
        b = axis_angle_to_matrix(np.array([0.0, 0.0, np.pi / 2.0]))
        self.assertAlmostEqual(float(geodesic_distance(a, b)), np.pi / 2.0, places=7)


if __name__ == "__main__":
    unittest.main()

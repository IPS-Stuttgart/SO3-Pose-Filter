from __future__ import annotations

import unittest

import numpy as np
from pyrecest.distributions.cart_prod.hyperhemisphere_cart_prod_dirac_distribution import (
    HyperhemisphereCartProdDiracDistribution,
)

from pose_filter.quaternion import (
    canonicalize_quaternions,
    pyrecest_hyperhemisphere_dirac_to_quaternions,
    pyrecest_hyperhemisphere_dirac_to_rotations,
    quaternions_to_rotations,
    rotations_to_pyrecest_hyperhemisphere_dirac,
    rotations_to_quaternions,
)
from pose_filter.so3 import axis_angle_to_matrix, geodesic_distance


class QuaternionBridgeTests(unittest.TestCase):
    def test_rotation_quaternion_roundtrip(self) -> None:
        rng = np.random.default_rng(12)
        rotvec = rng.normal(0.0, 0.6, size=(8, 23, 3))
        rotations = axis_angle_to_matrix(rotvec)

        quaternions = rotations_to_quaternions(rotations)
        recovered = quaternions_to_rotations(quaternions)

        self.assertEqual(quaternions.shape, (8, 23, 4))
        self.assertTrue(bool(np.all(quaternions[..., 3] >= 0.0)))
        self.assertLess(float(np.max(np.abs(np.linalg.norm(quaternions, axis=-1) - 1.0))), 1e-10)
        self.assertLess(float(np.max(geodesic_distance(rotations, recovered))), 1e-7)

    def test_antipodal_quaternions_are_canonicalized(self) -> None:
        rotation = axis_angle_to_matrix(np.array([0.2, -0.3, 0.4]))
        quaternion = rotations_to_quaternions(rotation)

        canonical = canonicalize_quaternions(quaternion)
        antipodal = canonicalize_quaternions(-quaternion)

        self.assertTrue(bool(np.allclose(canonical, antipodal)))
        self.assertLess(
            float(np.max(geodesic_distance(rotation, quaternions_to_rotations(-quaternion)))),
            1e-10,
        )

    def test_pyrecest_dirac_bridge_roundtrip(self) -> None:
        rng = np.random.default_rng(4)
        rotations = axis_angle_to_matrix(rng.normal(0.0, 0.4, size=(5, 23, 3)))
        weights = np.arange(1, 6, dtype=np.float64)
        weights = weights / np.sum(weights)

        distribution = rotations_to_pyrecest_hyperhemisphere_dirac(rotations, weights=weights)
        self.assertIsInstance(distribution, HyperhemisphereCartProdDiracDistribution)
        quaternions, recovered_weights = pyrecest_hyperhemisphere_dirac_to_quaternions(distribution)
        recovered_rotations, recovered_rotation_weights = pyrecest_hyperhemisphere_dirac_to_rotations(distribution)

        self.assertEqual(distribution.d.shape, (5, 92))
        self.assertEqual(distribution.dim_hemisphere, 3)
        self.assertEqual(distribution.n_hemispheres, 23)
        self.assertEqual(quaternions.shape, (5, 23, 4))
        self.assertTrue(bool(np.all(quaternions[..., 3] >= 0.0)))
        self.assertTrue(bool(np.allclose(recovered_weights, weights)))
        self.assertTrue(bool(np.allclose(recovered_rotation_weights, weights)))
        self.assertLess(float(np.max(geodesic_distance(rotations, recovered_rotations))), 1e-7)


if __name__ == "__main__":
    unittest.main()

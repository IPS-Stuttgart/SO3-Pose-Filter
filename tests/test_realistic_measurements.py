from __future__ import annotations

import unittest

import numpy as np
from _path import SRC  # noqa: F401
from pose_filter.measurements import (
    confidence_to_noise_sigma,
    log_likelihood,
    make_occlusion_mask,
    make_synthetic_measurements,
)
from pose_filter.particle_filter import run_particle_filter
from pose_filter.so3 import axis_angle_to_matrix
from pose_filter.transitions import PersistenceTransition


class RealisticMeasurementTests(unittest.TestCase):
    def test_markov_occlusion_confidence_noise_and_outliers(self) -> None:
        rng = np.random.default_rng(123)
        truth = axis_angle_to_matrix(np.zeros((80, 4, 3), dtype=np.float64))

        measurements = make_synthetic_measurements(
            truth,
            noise_deg=4.0,
            occlusion_prob=0.65,
            rng=rng,
            confidence_noise_std=0.35,
            min_confidence=0.1,
            occlusion_model="markov",
            occlusion_recovery_prob=0.08,
            outlier_prob=0.35,
            confidence_calibrated_noise=True,
            confidence_noise_min_deg=2.0,
            confidence_noise_max_deg=20.0,
            confidence_noise_gamma=2.0,
        )
        joint_noise_sigma_rad = measurements.joint_noise_sigma_rad
        outlier_mask = measurements.outlier_mask
        self.assertIsNotNone(joint_noise_sigma_rad)
        self.assertIsNotNone(outlier_mask)
        assert joint_noise_sigma_rad is not None
        assert outlier_mask is not None

        self.assertEqual(measurements.mask.shape, (80, 4))
        self.assertEqual(measurements.confidence.shape, measurements.mask.shape)
        self.assertEqual(joint_noise_sigma_rad.shape, measurements.mask.shape)
        self.assertEqual(outlier_mask.shape, measurements.mask.shape)
        self.assertTrue(bool(np.all(measurements.mask[0])))
        self.assertTrue(bool(np.all(outlier_mask <= measurements.mask)))
        self.assertTrue(bool(np.any((~measurements.mask[:-1]) & (~measurements.mask[1:]))))
        self.assertGreaterEqual(
            float(np.min(joint_noise_sigma_rad[measurements.mask])),
            np.radians(2.0) - 1e-12,
        )
        self.assertLessEqual(
            float(np.max(joint_noise_sigma_rad)),
            np.radians(20.0) + 1e-12,
        )

        visible_conf = measurements.confidence[measurements.mask]
        visible_sigma = joint_noise_sigma_rad[measurements.mask]
        low_conf_idx = int(np.argmin(visible_conf))
        high_conf_idx = int(np.argmax(visible_conf))
        self.assertGreaterEqual(visible_sigma[low_conf_idx], visible_sigma[high_conf_idx])

    def test_confidence_to_noise_sigma_is_monotone(self) -> None:
        confidence = np.array([1.0, 0.5, 0.0])
        sigma = confidence_to_noise_sigma(
            confidence,
            np.radians(2.0),
            np.radians(20.0),
            gamma=1.5,
        )

        self.assertAlmostEqual(float(sigma[0]), np.radians(2.0))
        self.assertAlmostEqual(float(sigma[-1]), np.radians(20.0))
        self.assertTrue(bool(np.all(np.diff(sigma) > 0.0)))

    def test_outlier_mixture_likelihood_limits_particle_collapse(self) -> None:
        state = axis_angle_to_matrix(np.zeros((1, 3)))
        observation = axis_angle_to_matrix(np.array([[0.0, 0.0, np.pi]]))
        mask = np.array([True])

        gaussian_only = log_likelihood(
            observation,
            state,
            mask,
            np.radians(5.0),
            outlier_prob=0.0,
        )
        mixture = log_likelihood(
            observation,
            state,
            mask,
            np.radians(5.0),
            outlier_prob=0.2,
        )

        self.assertGreater(float(mixture), float(gaussian_only))

    def test_particle_filter_accepts_confidence_calibrated_noise(self) -> None:
        rng = np.random.default_rng(21)
        truth = axis_angle_to_matrix(np.zeros((8, 3, 3), dtype=np.float64))
        measurements = make_synthetic_measurements(
            truth,
            noise_deg=5.0,
            occlusion_prob=0.4,
            rng=rng,
            confidence_noise_std=0.25,
            occlusion_model="markov",
            outlier_prob=0.2,
            confidence_calibrated_noise=True,
            confidence_noise_min_deg=2.0,
            confidence_noise_max_deg=15.0,
        )

        result = run_particle_filter(
            measurements.observations,
            measurements.mask,
            PersistenceTransition(),
            measurements.noise_sigma_rad,
            num_particles=8,
            rng=rng,
            confidence=measurements.confidence,
            joint_noise_sigma_rad=measurements.joint_noise_sigma_rad,
        )

        self.assertEqual(result.estimates.shape, truth.shape)
        self.assertTrue(bool(np.all(np.isfinite(result.effective_sample_size))))

    def test_make_occlusion_mask_preserves_iid_mode(self) -> None:
        rng = np.random.default_rng(3)
        mask = make_occlusion_mask((6, 2), 0.5, rng, occlusion_model="iid")

        self.assertEqual(mask.shape, (6, 2))
        self.assertTrue(bool(np.all(mask[0])))


if __name__ == "__main__":
    unittest.main()

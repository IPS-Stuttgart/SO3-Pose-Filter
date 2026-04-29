from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from _path import SRC  # noqa: F401
from pose_filter.data import load_dataset, split_sequences
from pose_filter.evaluation import (
    ablation_rows,
    evaluate_filter_sequence,
    evaluate_filter_sequence_artifacts,
    temporal_metrics,
)
from pose_filter.measurements import log_likelihood, make_synthetic_measurements
from pose_filter.particle_filter import run_filter, run_particle_filter
from pose_filter.pyrecest_filter import is_pyrecest_filter_available
from pose_filter.so3 import axis_angle_to_matrix
from pose_filter.transitions import (
    GaussianRandomWalkTransition,
    LearnedDeltaTransition,
    PersistenceTransition,
)


def _write_toy(path: Path, frames: int = 45, fps: float = 60.0) -> None:
    poses = np.zeros((frames, 156), dtype=np.float64)
    t = np.linspace(0.0, 1.0, frames)
    for joint in range(23):
        start = 3 + joint * 3
        poses[:, start] = 0.1 * np.sin(2.0 * np.pi * t * (joint % 3 + 1))
        poses[:, start + 1] = 0.05 * np.cos(2.0 * np.pi * t)
    np.savez(path, poses=poses, mocap_framerate=np.asarray(fps))


class PipelineTests(unittest.TestCase):
    def test_preprocess_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_toy(root / "seq.npz")
            seqs = load_dataset(root, "", frame_rate=20, num_joints=23)
            self.assertEqual(seqs[0].rotations.shape[1:], (23, 3, 3))
            self.assertEqual(seqs[0].rotations.shape[0], 15)

    def test_transition_models_and_filter_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for idx in range(4):
                _write_toy(root / f"seq_{idx}.npz", frames=45 + idx * 3)
            seqs = load_dataset(root, "", frame_rate=20, num_joints=23)
            train, _, test = split_sequences(seqs, seed=3)
            models = [
                PersistenceTransition(),
                GaussianRandomWalkTransition.fit(train),
                LearnedDeltaTransition.fit(train),
            ]
            rng = np.random.default_rng(9)
            meas = make_synthetic_measurements(
                test[0].rotations,
                10.0,
                0.2,
                rng,
                confidence_noise_std=0.15,
                min_confidence=0.4,
            )
            self.assertEqual(meas.confidence.shape, meas.mask.shape)
            self.assertTrue(np.all(meas.confidence[~meas.mask] == 0.0))
            self.assertTrue(np.all(meas.confidence[meas.mask] >= 0.4))
            for model in models:
                pred = model.sample_next(test[0].rotations[0], rng, n_samples=3)
                self.assertEqual(pred.shape, (3, 23, 3, 3))
                result = run_particle_filter(
                    meas.observations,
                    meas.mask,
                    model,
                    meas.noise_sigma_rad,
                    num_particles=16,
                    rng=rng,
                    confidence=meas.confidence,
                    factorized_update=False,
                    resample_threshold=0.75,
                )
                self.assertEqual(result.estimates.shape, test[0].rotations.shape)
            rows = ablation_rows(
                test,
                models[1],
                noise_deg=10.0,
                occlusion_prob=0.2,
                base_num_particles=16,
                seed=17,
                base_proposal_gain=0.2,
                base_factorized_update=True,
                base_resample_threshold=0.5,
                particle_counts=[8],
                proposal_gains=[0.0],
                factorized_updates=[False],
                resample_thresholds=[0.25],
            )
            self.assertTrue(
                any(
                    row["ablation"] == "proposal_gain" and row["value"] == "0"
                    for row in rows
                )
            )
            self.assertTrue(any(row["ablation"] == "factorized_update" for row in rows))

    def test_filter_evaluation_reports_smoother_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_toy(root / "seq.npz")
            seqs = load_dataset(root, "", frame_rate=20, num_joints=23)
            rng = np.random.default_rng(12)

            row = evaluate_filter_sequence(
                seqs[0],
                PersistenceTransition(),
                noise_deg=8.0,
                occlusion_prob=0.2,
                num_particles=16,
                rng=rng,
            )

            self.assertIn("smoother_ema_error_deg", row)
            self.assertIn("smoother_chordal_error_deg", row)

    def test_evaluation_reports_research_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for idx in range(4):
                _write_toy(root / f"seq_{idx}.npz", frames=45 + idx * 3)
            seqs = load_dataset(root, "", frame_rate=20, num_joints=23)
            train, _, test = split_sequences(seqs, seed=4)
            model = GaussianRandomWalkTransition.fit(train)

            artifacts = evaluate_filter_sequence_artifacts(
                test[0],
                model,
                noise_deg=8.0,
                occlusion_prob=0.5,
                num_particles=16,
                rng=np.random.default_rng(10),
            )

            self.assertIn("observed_joint_error_deg", artifacts.metrics)
            self.assertIn("filter_observed_joint_error_deg", artifacts.metrics)
            self.assertIn("filter_occluded_joint_error_deg", artifacts.metrics)
            self.assertIn("filter_acceleration_deg", artifacts.metrics)
            self.assertIn("filter_jerk_error_deg", artifacts.metrics)
            self.assertEqual(len(artifacts.per_joint_rows), 23)
            self.assertEqual(
                {row["estimate"] for row in artifacts.temporal_rows},
                {"truth", "observed", "filter", "persistence"},
            )

    def test_pyrecest_filter_backend_smoke(self) -> None:
        if not is_pyrecest_filter_available():
            self.skipTest("PyRecEst SO3ProductParticleFilter is not available")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_toy(root / "seq.npz", frames=45)
            seq = load_dataset(root, "", frame_rate=20, num_joints=23)[0]

            rng = np.random.default_rng(23)
            meas = make_synthetic_measurements(seq.rotations, 8.0, 0.25, rng)
            result = run_filter(
                meas.observations,
                meas.mask,
                PersistenceTransition(),
                meas.noise_sigma_rad,
                num_particles=12,
                rng=rng,
                factorized_update=False,
                resample_threshold=0.75,
                backend="pyrecest",
            )

            self.assertEqual(result.estimates.shape, seq.rotations.shape)
            self.assertTrue(bool(np.all(np.isfinite(result.effective_sample_size))))
            self.assertTrue(bool(np.all(result.effective_sample_size > 0.0)))

    def test_temporal_metrics_are_zero_for_constant_pose(self) -> None:
        rotations = np.broadcast_to(np.eye(3), (5, 23, 3, 3)).copy()

        metrics = temporal_metrics(rotations, truth=rotations)

        self.assertAlmostEqual(metrics["acceleration_deg"], 0.0)
        self.assertAlmostEqual(metrics["jerk_deg"], 0.0)
        self.assertAlmostEqual(metrics["acceleration_error_deg"], 0.0)
        self.assertAlmostEqual(metrics["jerk_error_deg"], 0.0)

    def test_confidence_downweights_measurement_likelihood(self) -> None:
        state = axis_angle_to_matrix(np.zeros((1, 3)))
        observation = axis_angle_to_matrix(np.array([[0.0, 0.0, 1.0]]))
        mask = np.array([True])

        high_confidence = log_likelihood(
            observation,
            state,
            mask,
            np.radians(10.0),
            confidence=np.array([1.0]),
        )
        low_confidence = log_likelihood(
            observation,
            state,
            mask,
            np.radians(10.0),
            confidence=np.array([0.1]),
        )
        zero_confidence = log_likelihood(
            observation,
            state,
            mask,
            np.radians(10.0),
            confidence=np.array([0.0]),
        )

        self.assertGreater(float(low_confidence), float(high_confidence))
        self.assertAlmostEqual(float(zero_confidence), 0.0)

        small_sigma = log_likelihood(
            observation,
            state,
            mask,
            np.radians(10.0),
            joint_noise_sigma_rad=np.array([0.1]),
        )
        large_sigma = log_likelihood(
            observation,
            state,
            mask,
            np.radians(10.0),
            joint_noise_sigma_rad=np.array([1.0]),
        )
        self.assertGreater(float(large_sigma), float(small_sigma))


if __name__ == "__main__":
    unittest.main()

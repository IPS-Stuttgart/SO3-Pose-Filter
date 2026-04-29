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
    HistoryMLPDeltaTransition,
    LearnedDeltaTransition,
    MLPDeltaTransition,
    PersistenceTransition,
)
from run_first_results_benchmark import run_first_results_benchmark
from prepare_amass_windows import prepare_windows


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

    def test_prepare_amass_windows_selects_motion_segments(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_toy(root / "dynamic.npz", frames=180)
            np.savez(
                root / "static.npz",
                poses=np.zeros((180, 156), dtype=np.float64),
                mocap_framerate=np.asarray(60.0),
            )

            report = prepare_windows(
                data_root=root,
                output_dir=root / "segments",
                report_path=root / "report.json",
                manifest_path=root / "manifest.csv",
                frame_rate=20,
                num_joints=23,
                segment_frames=20,
                stride_frames=10,
                max_segments=2,
                selection="top-motion",
                max_per_file=1,
            )

            self.assertEqual(report["selected_count"], 2)
            self.assertTrue((root / "manifest.csv").exists())
            self.assertGreater(report["selected"][0]["motion_deg_per_frame"], 0.0)
            seqs = load_dataset(root / "segments", "", frame_rate=20, num_joints=23)
            self.assertEqual(len(seqs), 2)
            self.assertEqual(seqs[0].rotations.shape[1:], (23, 3, 3))

    def test_first_results_benchmark_writes_summary_and_plots(self) -> None:
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
                    "occlusion_prob": 0.5,
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
                output_dir,
                methods=("raw", "persistence", "gaussian_rw"),
                noise_grid=[5.0],
                occlusion_grid=[0.0, 0.5],
            )

            self.assertEqual(summary["row_count"], 6)
            self.assertEqual(
                set(summary["means_by_method"]), {"raw", "persistence", "gaussian_rw"}
            )
            self.assertTrue((output_dir / "benchmark_metrics.csv").exists())
            self.assertTrue((output_dir / "first_results_summary.json").exists())
            self.assertTrue(
                (output_dir / "plots" / "tracking_error_heatmap.svg").exists()
            )
            self.assertTrue(
                (output_dir / "plots" / "filter_vs_baselines.svg").exists()
            )

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
                MLPDeltaTransition.fit(train, hidden_dim=12, epochs=3, seed=5),
                HistoryMLPDeltaTransition.fit(
                    train, history_length=2, hidden_dim=12, epochs=3, seed=6
                ),
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

    def test_mlp_delta_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for idx in range(3):
                _write_toy(root / f"seq_{idx}.npz", frames=48 + idx * 3)
            train = load_dataset(root, "", frame_rate=20, num_joints=23)
            model = MLPDeltaTransition.fit(
                train,
                hidden_dim=10,
                epochs=4,
                learning_rate=0.002,
                batch_size=16,
                seed=7,
            )
            checkpoint = root / "mlp_transition.npz"
            model.save_npz(checkpoint)
            loaded = MLPDeltaTransition.load_npz(checkpoint)

            pred = model.deterministic_next(train[0].rotations[:2])
            loaded_pred = loaded.deterministic_next(train[0].rotations[:2])

            self.assertTrue(np.allclose(pred, loaded_pred))
            self.assertEqual(
                loaded.sample_next(
                    train[0].rotations[0],
                    np.random.default_rng(1),
                    n_samples=2,
                ).shape,
                (2, 23, 3, 3),
            )

    def test_history_mlp_delta_uses_history_and_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for idx in range(3):
                _write_toy(root / f"seq_{idx}.npz", frames=54 + idx * 3)
            train = load_dataset(root, "", frame_rate=20, num_joints=23)
            model = HistoryMLPDeltaTransition.fit(
                train,
                history_length=2,
                hidden_dim=10,
                epochs=4,
                learning_rate=0.002,
                batch_size=16,
                seed=11,
            )
            history = [train[0].rotations[0], train[0].rotations[1]]
            checkpoint = root / "history_mlp_transition.npz"
            model.save_npz(checkpoint)
            loaded = HistoryMLPDeltaTransition.load_npz(checkpoint)

            pred = model.deterministic_next_from_history(history)
            loaded_pred = loaded.deterministic_next_from_history(history)

            self.assertTrue(np.allclose(pred, loaded_pred))
            self.assertEqual(model.history_length, 2)
            self.assertEqual(
                loaded.sample_next_from_history(
                    history,
                    np.random.default_rng(2),
                ).shape,
                (23, 3, 3),
            )

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

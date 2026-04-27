from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pose_filter.data import load_dataset, split_sequences
from pose_filter.evaluation import ablation_rows
from pose_filter.measurements import make_synthetic_measurements
from pose_filter.particle_filter import run_particle_filter
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
            meas = make_synthetic_measurements(test[0].rotations, 10.0, 0.2, rng)
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
            self.assertTrue(any(row["ablation"] == "proposal_gain" and row["value"] == "0" for row in rows))
            self.assertTrue(any(row["ablation"] == "factorized_update" for row in rows))


if __name__ == "__main__":
    unittest.main()

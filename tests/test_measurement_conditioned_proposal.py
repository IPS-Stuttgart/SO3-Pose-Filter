from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from _path import SRC  # noqa: F401
from pose_filter.data import load_dataset, split_sequences
from pose_filter.measurements import make_synthetic_measurements
from pose_filter.proposal_particle_filter import run_particle_filter_with_proposal
from pose_filter.proposals import MeasurementConditionedMLPProposal
from pose_filter.transitions import GaussianRandomWalkTransition


def _write_toy(path: Path, frames: int = 45, fps: float = 60.0) -> None:
    poses = np.zeros((frames, 156), dtype=np.float64)
    t = np.linspace(0.0, 1.0, frames)
    for joint in range(23):
        start = 3 + joint * 3
        poses[:, start] = 0.1 * np.sin(2.0 * np.pi * t * (joint % 3 + 1))
        poses[:, start + 1] = 0.05 * np.cos(2.0 * np.pi * t)
    np.savez(path, poses=poses, mocap_framerate=np.asarray(fps))


class MeasurementConditionedProposalTests(unittest.TestCase):
    def test_measurement_conditioned_proposal_trains_roundtrips_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for idx in range(4):
                _write_toy(root / f"seq_{idx}.npz", frames=45 + idx * 3)
            seqs = load_dataset(root, "", frame_rate=20, num_joints=23)
            train, _, test = split_sequences(seqs, seed=5)
            transition = GaussianRandomWalkTransition.fit(train)
            proposal = MeasurementConditionedMLPProposal.fit(
                train,
                transition,
                history_length=2,
                hidden_dim=12,
                epochs=2,
                batch_size=16,
                seed=7,
                training_noise_deg=5.0,
                training_occlusion_prob=0.25,
                noise_scale=0.0,
                max_correction_rad=np.radians(5.0),
            )
            checkpoint = root / "measurement_proposal.npz"
            proposal.save_npz(checkpoint)
            loaded = MeasurementConditionedMLPProposal.load_npz(checkpoint)
            self.assertEqual(loaded.history_length, 2)
            self.assertEqual(loaded.hidden_dim, 12)
            self.assertAlmostEqual(loaded.max_correction_rad, np.radians(5.0))

            rng = np.random.default_rng(11)
            meas = make_synthetic_measurements(
                test[0].rotations,
                5.0,
                0.25,
                rng,
                confidence_noise_std=0.1,
                min_confidence=0.5,
            )
            result = run_particle_filter_with_proposal(
                meas.observations,
                meas.mask,
                transition,
                meas.noise_sigma_rad,
                num_particles=12,
                rng=rng,
                proposal_model=loaded,
                confidence=meas.confidence,
                factorized_update=True,
                proposal_gain=0.1,
            )
            self.assertEqual(result.estimates.shape, test[0].rotations.shape)
            self.assertTrue(np.all(np.isfinite(result.effective_sample_size)))
            self.assertTrue(np.all(result.effective_sample_size > 0.0))


if __name__ == "__main__":
    unittest.main()

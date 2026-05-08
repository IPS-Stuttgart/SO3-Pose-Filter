from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from _path import SRC  # noqa: F401
from pose_filter.block_particle_filter import (
    DEFAULT_SMPL_BODY_BLOCKS,
    format_particle_blocks,
    resolve_particle_blocks,
    run_block_particle_filter,
)
from pose_filter.data import load_dataset, split_sequences
from pose_filter.measurements import make_synthetic_measurements
from pose_filter.particle_filter import run_filter
from pose_filter.transitions import GaussianRandomWalkTransition


def _write_toy(path: Path, frames: int = 45, fps: float = 60.0) -> None:
    poses = np.zeros((frames, 156), dtype=np.float64)
    t = np.linspace(0.0, 1.0, frames)
    for joint in range(23):
        start = 3 + joint * 3
        poses[:, start] = 0.1 * np.sin(2.0 * np.pi * t * (joint % 3 + 1))
        poses[:, start + 1] = 0.05 * np.cos(2.0 * np.pi * t)
    np.savez(path, poses=poses, mocap_framerate=np.asarray(fps))


class BlockParticleFilterTests(unittest.TestCase):
    def test_smpl_body_blocks_are_strict_partition(self) -> None:
        self.assertEqual(resolve_particle_blocks(23, "smpl_body"), DEFAULT_SMPL_BODY_BLOCKS)
        self.assertEqual(resolve_particle_blocks(23, "auto"), DEFAULT_SMPL_BODY_BLOCKS)
        self.assertEqual(resolve_particle_blocks(3, "joint"), ((0,), (1,), (2,)))
        self.assertEqual(resolve_particle_blocks(5, "contiguous"), ((0, 1, 2, 3), (4,)))
        self.assertEqual(format_particle_blocks([[0, 2], [1, 3]]), "0,2;1,3")
        with self.assertRaises(ValueError):
            resolve_particle_blocks(23, [[0, 1], [1, 2]])
        with self.assertRaises(ValueError):
            resolve_particle_blocks(22, "smpl_body")

    def test_block_particle_filter_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for idx in range(4):
                _write_toy(root / f"seq_{idx}.npz", frames=45 + idx * 3)
            seqs = load_dataset(root, "", frame_rate=20, num_joints=23)
            train, _, test = split_sequences(seqs, seed=8)
            model = GaussianRandomWalkTransition.fit(train)
            rng = np.random.default_rng(19)
            meas = make_synthetic_measurements(
                test[0].rotations,
                8.0,
                0.4,
                rng,
                occlusion_model="markov",
                outlier_prob=0.05,
            )

            result = run_block_particle_filter(
                meas.observations,
                meas.mask,
                model,
                meas.noise_sigma_rad,
                num_particles=16,
                rng=rng,
                confidence=meas.confidence,
                joint_noise_sigma_rad=meas.joint_noise_sigma_rad,
                particle_blocks="smpl_body",
                resample_threshold=0.75,
            )

            self.assertEqual(result.estimates.shape, test[0].rotations.shape)
            self.assertTrue(bool(np.all(np.isfinite(result.effective_sample_size))))
            self.assertTrue(bool(np.all(result.effective_sample_size > 0.0)))
            self.assertEqual(result.resampled.shape, result.effective_sample_size.shape)

    def test_run_filter_routes_particle_blocks_to_numpy_block_filter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for idx in range(4):
                _write_toy(root / f"seq_{idx}.npz", frames=45 + idx * 3)
            seqs = load_dataset(root, "", frame_rate=20, num_joints=23)
            train, _, test = split_sequences(seqs, seed=9)
            model = GaussianRandomWalkTransition.fit(train)
            rng = np.random.default_rng(23)
            meas = make_synthetic_measurements(test[0].rotations, 8.0, 0.25, rng)

            result = run_filter(
                meas.observations,
                meas.mask,
                model,
                meas.noise_sigma_rad,
                num_particles=12,
                rng=rng,
                confidence=meas.confidence,
                particle_blocks="smpl_body",
                backend="numpy",
            )

            self.assertEqual(result.estimates.shape, test[0].rotations.shape)
            with self.assertRaises(ValueError):
                run_filter(
                    meas.observations,
                    meas.mask,
                    model,
                    meas.noise_sigma_rad,
                    num_particles=12,
                    rng=rng,
                    particle_blocks="smpl_body",
                    backend="pyrecest",
                )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from _path import SRC  # noqa: F401

from pose_filter.data import load_dataset
from pose_filter.measurements import make_synthetic_measurements
from pose_filter.particle_filter import run_filter
from pose_filter.pyrecest_filter import (
    is_pyrecest_filter_available,
    is_pyrecest_partitioned_filter_available,
)
from pose_filter.transitions import PersistenceTransition


def _write_toy(path: Path, frames: int = 45, fps: float = 60.0) -> None:
    poses = np.zeros((frames, 156), dtype=np.float64)
    t = np.linspace(0.0, 1.0, frames)
    for joint in range(23):
        start = 3 + joint * 3
        poses[:, start] = 0.1 * np.sin(2.0 * np.pi * t * (joint % 3 + 1))
        poses[:, start + 1] = 0.05 * np.cos(2.0 * np.pi * t)
    np.savez(path, poses=poses, mocap_framerate=np.asarray(fps))


class PyRecEstBackendTests(unittest.TestCase):
    def test_pyrecest_global_filter_backend_smoke(self) -> None:
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

    def test_pyrecest_partitioned_filter_backend_smoke(self) -> None:
        if not is_pyrecest_partitioned_filter_available():
            self.skipTest("PyRecEst PartitionedSO3ProductParticleFilter is not available")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_toy(root / "seq.npz", frames=45)
            seq = load_dataset(root, "", frame_rate=20, num_joints=23)[0]

            rng = np.random.default_rng(29)
            meas = make_synthetic_measurements(seq.rotations, 8.0, 0.25, rng)
            result = run_filter(
                meas.observations,
                meas.mask,
                PersistenceTransition(),
                meas.noise_sigma_rad,
                num_particles=12,
                rng=rng,
                factorized_update=True,
                particle_blocks="smpl_body",
                resample_threshold=0.75,
                backend="pyrecest",
            )

            self.assertEqual(result.estimates.shape, seq.rotations.shape)
            self.assertTrue(bool(np.all(np.isfinite(result.effective_sample_size))))
            self.assertTrue(bool(np.all(result.effective_sample_size > 0.0)))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_sequence(rng: np.random.Generator, frames: int, dims: int = 156) -> np.ndarray:
    poses = np.zeros((frames, dims), dtype=np.float64)
    t = np.linspace(0.0, 1.0, frames)
    for joint in range(23):
        start = 3 + joint * 3
        amp = rng.uniform(0.03, 0.35, size=3)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
        freq = rng.integers(1, 4, size=3)
        trend = rng.normal(0.0, 0.02, size=3)
        poses[:, start : start + 3] = (
            amp[None, :]
            * np.sin(2.0 * np.pi * t[:, None] * freq[None, :] + phase[None, :])
            + trend[None, :] * t[:, None]
        )
    return poses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create tiny AMASS-like .npz files for smoke tests."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequences", type=int, default=6)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for idx in range(args.sequences):
        poses = make_sequence(rng, args.frames + idx * 3)
        np.savez(
            out / f"toy_{idx:03d}.npz",
            poses=poses,
            mocap_framerate=np.asarray(args.fps),
        )
    print(f"wrote {args.sequences} toy sequences to {out}")


if __name__ == "__main__":
    main()

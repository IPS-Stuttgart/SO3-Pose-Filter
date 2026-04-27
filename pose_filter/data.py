"""AMASS/SMPL preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .so3 import axis_angle_to_matrix


@dataclass(frozen=True)
class PoseSequence:
    name: str
    rotations: np.ndarray
    source_fps: float
    frame_rate: int


def find_amass_files(data_root: str | Path, dataset_subset: str = "") -> list[Path]:
    """Find AMASS-style `.npz` files, optionally filtered by substring/path part."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")
    files = sorted(root.rglob("*.npz"))
    if dataset_subset:
        needle = dataset_subset.lower()
        files = [p for p in files if needle in str(p.relative_to(root)).lower()]
    return files


def _source_fps(npz: np.lib.npyio.NpzFile) -> float:
    for key in ("mocap_framerate", "mocap_frame_rate", "fps"):
        if key in npz:
            value = np.asarray(npz[key]).reshape(-1)[0]
            return float(value)
    return 60.0


def load_amass_sequence(path: str | Path, frame_rate: int = 20, num_joints: int = 23) -> PoseSequence:
    """Load one AMASS/SMPL `.npz` file as `[T, J, 3, 3]` local body rotations."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        if "poses" not in npz:
            raise ValueError(f"{path} has no 'poses' array")
        poses = np.asarray(npz["poses"], dtype=np.float64)
        fps = _source_fps(npz)
    if poses.ndim != 2:
        raise ValueError(f"{path} poses must be shaped [T, D], got {poses.shape}")
    needed = 3 + num_joints * 3
    if poses.shape[1] < needed:
        raise ValueError(f"{path} poses has {poses.shape[1]} dims, need at least {needed}")

    stride = max(1, int(round(fps / float(frame_rate))))
    body_axis_angle = poses[::stride, 3:needed].reshape(-1, num_joints, 3)
    rotations = axis_angle_to_matrix(body_axis_angle)
    return PoseSequence(
        name=path.stem,
        rotations=rotations,
        source_fps=fps,
        frame_rate=frame_rate,
    )


def load_dataset(
    data_root: str | Path,
    dataset_subset: str,
    frame_rate: int,
    num_joints: int,
    max_sequences: int | None = None,
    min_frames: int = 2,
) -> list[PoseSequence]:
    """Load a list of AMASS sequences with basic length filtering."""
    files = find_amass_files(data_root, dataset_subset)
    if max_sequences is not None:
        files = files[:max_sequences]
    sequences: list[PoseSequence] = []
    errors: list[str] = []
    for path in files:
        try:
            seq = load_amass_sequence(path, frame_rate=frame_rate, num_joints=num_joints)
            if seq.rotations.shape[0] >= min_frames:
                sequences.append(seq)
        except Exception as exc:  # Keep scanning mixed AMASS directories.
            errors.append(f"{path}: {exc}")
    if not sequences:
        detail = "\n".join(errors[:5])
        raise ValueError(f"no usable AMASS sequences found under {data_root}\n{detail}")
    return sequences


def split_sequences(
    sequences: Iterable[PoseSequence],
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[list[PoseSequence], list[PoseSequence], list[PoseSequence]]:
    """Split sequences into train/validation/test lists."""
    sequences = list(sequences)
    if len(sequences) < 3:
        return sequences[:1], sequences[1:2], sequences[2:] or sequences[:1]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sequences))
    shuffled = [sequences[i] for i in order]
    n = len(shuffled)
    n_train = max(1, int(round(n * train_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def sequence_pairs(sequences: Iterable[PoseSequence]) -> tuple[np.ndarray, np.ndarray]:
    """Stack consecutive state pairs from sequences."""
    xs = []
    ys = []
    for seq in sequences:
        if seq.rotations.shape[0] < 2:
            continue
        xs.append(seq.rotations[:-1])
        ys.append(seq.rotations[1:])
    if not xs:
        raise ValueError("need at least one sequence with two frames")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

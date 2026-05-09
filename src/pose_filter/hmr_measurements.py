"""Adapters for real HMR/HPS outputs used as SO(3)^K measurements."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .detector_import import ImportedMeasurements
from .measurements import validate_confidence
from .so3 import axis_angle_to_matrix, project_to_so3

SUPPORTED_HMR_SUFFIXES = {".npz", ".json", ".pkl", ".pickle", ".pt", ".pth"}
BODY_KEYS = {"body_pose", "pred_body_pose", "smpl_body_pose", "pose_body", "bodypose"}
FULL_KEYS = {"poses", "pose", "pred_pose", "full_pose", "fullpose", "smpl_pose", "theta"}
ROTMAT_KEYS = {"body_pose_rotmat", "pred_body_rotmat", "pred_rotmat", "rotmat", "rotmats"}
CONF_KEYS = {"confidence", "confidences", "body_pose_confidence", "joint_confidence", "scores", "score", "keypoint_scores"}
MASK_KEYS = {"mask", "masks", "visible", "visibility", "valid", "valid_mask"}
FPS_KEYS = {"mocap_framerate", "mocap_frame_rate", "fps", "frame_rate", "source_fps"}
RAD_NOISE_KEYS = {"noise_sigma_rad", "measurement_noise_sigma_rad", "joint_noise_sigma_rad"}
DEG_NOISE_KEYS = {"noise_sigma_deg", "measurement_noise_sigma_deg", "joint_noise_sigma_deg"}
FRAME_GROUPS = {
    "auto": (),
    "global": ("smpl_params_global", "pred_smpl_params_global", "global", "world"),
    "incam": ("smpl_params_incam", "pred_smpl_params_incam", "incam", "camera", "local"),
}


def find_hmr_measurement_files(data_root: str | Path, dataset_subset: str = "") -> list[Path]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"HMR data root does not exist: {root}")
    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_HMR_SUFFIXES:
            raise ValueError(f"unsupported HMR output suffix: {root.suffix}")
        return [root]
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_HMR_SUFFIXES)
    if dataset_subset:
        files = [p for p in files if dataset_subset.lower() in str(p.relative_to(root)).lower()]
    return files


def _load_pickle(path: Path) -> Any:
    # Optional loader for trusted local HMR exports.
    import pickle  # nosec B403

    with path.open("rb") as handle:
        # Guarded by allow_unsafe_deserialization.
        return pickle.load(handle)  # nosec B301


def _load_torch(path: Path, allow_unsafe_deserialization: bool) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(f"loading {path.suffix.lower()} HMR outputs requires PyTorch") from exc

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        if not allow_unsafe_deserialization:
            raise ValueError(
                "safe torch loading requires a PyTorch version with weights_only=True; " "pass allow_unsafe_deserialization=True only for trusted local HMR outputs"
            ) from exc
    except Exception as exc:
        if not allow_unsafe_deserialization:
            raise ValueError("safe torch loading failed; pass allow_unsafe_deserialization=True " "only for trusted local HMR outputs that require full pickle loading") from exc
    return torch.load(path, map_location="cpu", weights_only=False)  # nosec B614


def _load(path: Path, allow_unsafe_deserialization: bool = False) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as npz:
            return {k: np.asarray(npz[k]) for k in npz.files}
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".pkl", ".pickle"}:
        if not allow_unsafe_deserialization:
            raise ValueError("pickle HMR outputs can execute code while loading; pass " "allow_unsafe_deserialization=True only for trusted local files")
        return _load_pickle(path)
    if suffix in {".pt", ".pth"}:
        return _load_torch(path, allow_unsafe_deserialization)
    raise ValueError(f"unsupported HMR output suffix: {path.suffix}")


def _to_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy()
    if isinstance(value, (str, bytes)):
        return None
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return None
    return None if arr.dtype == object else arr


def _flatten(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(value, Mapping):
        out: dict[str, np.ndarray] = {}
        for key, child in value.items():
            out.update(_flatten(child, f"{prefix}.{key}" if prefix else str(key)))
        return out
    arr = _to_numpy(value)
    if prefix and arr is not None:
        return {prefix: arr}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
        out = {}
        for idx, child in enumerate(value):
            out.update(_flatten(child, f"{prefix}.{idx}" if prefix else str(idx)))
        if out:
            return out
    return {}


def _leaf(key: str) -> str:
    return key.replace("/", ".").split(".")[-1].lower()


def _score(key: str, pose_frame: str) -> tuple[int, int]:
    lower = key.lower()
    groups = FRAME_GROUPS[pose_frame]
    if groups:
        return (0 if any(g in lower for g in groups) else 1, len(key))
    if any(g in lower for g in FRAME_GROUPS["global"]):
        return (0, len(key))
    if any(g in lower for g in FRAME_GROUPS["incam"]):
        return (1, len(key))
    return (2, len(key))


def _items(flat: Mapping[str, np.ndarray], names: set[str], pose_frame: str = "auto") -> list[tuple[str, np.ndarray]]:
    out = [(k, v) for k, v in flat.items() if _leaf(k) in names]
    return sorted(out, key=lambda item: _score(item[0], pose_frame))


def _scalar(flat: Mapping[str, np.ndarray], names: set[str]) -> float | None:
    for key, value in flat.items():
        if _leaf(key) in names and np.asarray(value).size:
            try:
                val = float(np.asarray(value).reshape(-1)[0])
            except (TypeError, ValueError):
                continue
            if np.isfinite(val):
                return val
    return None


def _axis_body(values: np.ndarray, key: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] % 3 == 0:
        return arr.reshape(arr.shape[0], arr.shape[1] // 3, 3)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    raise ValueError(f"{key} must be [T,J*3] or [T,J,3], got {arr.shape}")


def _axis_full(values: np.ndarray, key: str, num_joints: int) -> np.ndarray:
    arr = _axis_body(values, key)
    return arr[:, 1 : num_joints + 1] if arr.shape[1] >= num_joints + 1 else arr


def _rotmat(values: np.ndarray, key: str, num_joints: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] % 9 == 0:
        arr = arr.reshape(arr.shape[0], arr.shape[1] // 9, 3, 3)
    elif arr.ndim == 3 and arr.shape[-1] == 9:
        arr = arr.reshape(arr.shape[0], arr.shape[1], 3, 3)
    elif arr.ndim == 3 and arr.shape[-2:] == (3, 3):
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[-2:] != (3, 3):
        raise ValueError(f"{key} must be rotation matrices, got {arr.shape}")
    if arr.shape[1] >= num_joints + 1 and "body" not in key.lower():
        arr = arr[:, 1 : num_joints + 1]
    return project_to_so3(arr)


def _pad(values: np.ndarray, num_joints: int, key: str, pad_missing_joints: bool) -> tuple[np.ndarray, np.ndarray, int]:
    source_joints = int(values.shape[1])
    if source_joints >= num_joints:
        return values[:, :num_joints], np.ones(num_joints, dtype=bool), source_joints
    if not pad_missing_joints:
        raise ValueError(f"{key} has {source_joints} joints, need {num_joints}")
    if values.shape[-1:] == (3,):
        padded = np.zeros((values.shape[0], num_joints, 3), dtype=np.float64)
    else:
        padded = np.broadcast_to(np.eye(3), (values.shape[0], num_joints, 3, 3)).copy()
    padded[:, :source_joints] = values
    valid = np.zeros(num_joints, dtype=bool)
    valid[:source_joints] = True
    return padded, valid, source_joints


def _pose(flat: Mapping[str, np.ndarray], num_joints: int, pose_frame: str, pad_missing_joints: bool) -> tuple[np.ndarray, np.ndarray, str, int]:
    for key, value in _items(flat, BODY_KEYS, pose_frame):
        body, valid, source_joints = _pad(_axis_body(value, key), num_joints, key, pad_missing_joints)
        return axis_angle_to_matrix(body), valid, key, source_joints
    for key, value in _items(flat, FULL_KEYS, pose_frame):
        body, valid, source_joints = _pad(_axis_full(value, key, num_joints), num_joints, key, pad_missing_joints)
        return axis_angle_to_matrix(body), valid, key, source_joints
    for key, value in _items(flat, ROTMAT_KEYS, pose_frame):
        mats, valid, source_joints = _pad(_rotmat(value, key, num_joints), num_joints, key, pad_missing_joints)
        return mats, valid, key, source_joints
    raise ValueError("could not find usable HMR pose keys")


def _joint_values(values: np.ndarray, t_steps: int, num_joints: int, valid: np.ndarray, missing: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., -1]
    if arr.ndim == 0:
        out = np.full((t_steps, num_joints), float(arr), dtype=np.float64)
    elif arr.ndim == 1 and arr.shape[0] == t_steps:
        out = np.repeat(arr[:, None], num_joints, axis=1)
    elif arr.ndim == 1:
        src = arr[1 : num_joints + 1] if arr.shape[0] >= num_joints + 1 else arr[:num_joints]
        out = np.repeat(src[None], t_steps, axis=0)
    elif arr.ndim == 2 and arr.shape[0] == t_steps:
        src = arr[:, 1 : num_joints + 1] if arr.shape[1] >= num_joints + 1 else arr[:, :num_joints]
        out = np.full((t_steps, num_joints), missing, dtype=np.float64)
        out[:, : src.shape[1]] = src
    else:
        raise ValueError(f"cannot broadcast values shaped {arr.shape} to [T,J]")
    return np.where(valid[None], out, missing)


def _confidence(flat: Mapping[str, np.ndarray], t_steps: int, num_joints: int, valid: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("confidence_scale must be positive and finite")
    for _, value in _items(flat, CONF_KEYS):
        try:
            return validate_confidence(_joint_values(value, t_steps, num_joints, valid, 0.0) / scale, (t_steps, num_joints))
        except ValueError:
            continue
    return np.repeat(valid[None], t_steps, axis=0).astype(np.float64)


def _mask(flat: Mapping[str, np.ndarray], t_steps: int, num_joints: int, valid: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    for _, value in _items(flat, MASK_KEYS):
        try:
            return _joint_values(value, t_steps, num_joints, valid, 0.0).astype(bool) & (confidence > 0.0)
        except ValueError:
            continue
    return np.repeat(valid[None], t_steps, axis=0) & (confidence > 0.0)


def _noise_sigma(flat: Mapping[str, np.ndarray], fallback_deg: float) -> float:
    val = _scalar(flat, RAD_NOISE_KEYS)
    if val is not None and val > 0.0:
        return val
    val = _scalar(flat, DEG_NOISE_KEYS)
    if val is not None and val > 0.0:
        return float(np.radians(val))
    return float(np.radians(fallback_deg))


def _has_pose(payload: Any, pose_frame: str) -> bool:
    try:
        _pose(_flatten(payload), 23, pose_frame, True)
        return True
    except ValueError:
        return False


def _single_track_mapping(payload: Mapping[Any, Any]) -> bool:
    keys = {str(key).lower() for key in payload}
    return bool(keys & (BODY_KEYS | FULL_KEYS | ROTMAT_KEYS | {g for groups in FRAME_GROUPS.values() for g in groups}))


def _tracks(payload: Any, stem: str, pose_frame: str) -> list[tuple[str, Any]]:
    if isinstance(payload, Mapping) and not _single_track_mapping(payload):
        tracks = [(f"{stem}_{key}", value) for key, value in payload.items() if _has_pose(value, pose_frame)]
        if tracks:
            return tracks
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, np.ndarray)):
        tracks = [(f"{stem}_{idx:03d}", value) for idx, value in enumerate(payload) if _has_pose(value, pose_frame)]
        if tracks:
            return tracks
    return [(stem, payload)] if _has_pose(payload, pose_frame) else []


def load_hmr_measurements(
    path: str | Path,
    *,
    frame_rate: int | None = 20,
    num_joints: int = 23,
    noise_deg: float = 10.0,
    pose_frame: str = "auto",
    confidence_scale: float = 1.0,
    pad_missing_joints: bool = True,
    name: str | None = None,
    allow_unsafe_deserialization: bool = False,
) -> list[ImportedMeasurements]:
    """Load one HMR/HPS result file into standardized detector measurements."""
    if pose_frame not in FRAME_GROUPS:
        raise ValueError(f"pose_frame must be one of {sorted(FRAME_GROUPS)}")
    path = Path(path)
    tracks = _tracks(
        _load(path, allow_unsafe_deserialization=allow_unsafe_deserialization),
        name or path.stem,
        pose_frame,
    )
    if not tracks:
        raise ValueError(f"no usable HMR pose track found in {path}")
    out = []
    for seq_name, payload in tracks:
        flat = _flatten(payload)
        rotations, valid, pose_key, source_joints = _pose(flat, int(num_joints), pose_frame, pad_missing_joints)
        fps = _scalar(flat, FPS_KEYS)
        stride = 1 if fps is None or frame_rate is None else max(1, int(round(float(fps) / float(frame_rate))))
        sigma = _noise_sigma(flat, noise_deg)
        conf_full = _confidence(flat, rotations.shape[0], int(num_joints), valid, confidence_scale)
        mask = _mask(flat, rotations.shape[0], int(num_joints), valid, conf_full)[::stride]
        conf = conf_full[::stride]
        out.append(
            ImportedMeasurements(
                name=seq_name,
                observations=rotations[::stride],
                mask=mask,
                confidence=np.where(mask, conf, 0.0),
                noise_sigma_rad=sigma,
                joint_noise_sigma_rad=None,
                source_fps=fps,
                frame_rate=frame_rate,
                source_path=str(path),
                source=f"hmr:{pose_frame}:{pose_key}:source_joints={source_joints}",
            )
        )
    return out


def load_hmr_measurement_dataset(
    data_root: str | Path,
    dataset_subset: str,
    *,
    frame_rate: int | None = 20,
    num_joints: int = 23,
    noise_deg: float = 10.0,
    pose_frame: str = "auto",
    confidence_scale: float = 1.0,
    pad_missing_joints: bool = True,
    max_files: int | None = None,
    allow_unsafe_deserialization: bool = False,
) -> dict[str, ImportedMeasurements]:
    files = find_hmr_measurement_files(data_root, dataset_subset)
    if max_files is not None:
        files = files[: int(max_files)]
    measurements: dict[str, ImportedMeasurements] = {}
    errors = []
    for path in files:
        try:
            for measurement in load_hmr_measurements(
                path,
                frame_rate=frame_rate,
                num_joints=num_joints,
                noise_deg=noise_deg,
                pose_frame=pose_frame,
                confidence_scale=confidence_scale,
                pad_missing_joints=pad_missing_joints,
                allow_unsafe_deserialization=allow_unsafe_deserialization,
            ):
                measurements[measurement.name] = measurement
        except Exception as exc:  # Keep scanning mixed output directories.
            errors.append(f"{path}: {exc}")
    if not measurements:
        raise ValueError(f"no usable HMR measurement files found under {data_root}\n" + "\n".join(errors[:5]))
    return measurements

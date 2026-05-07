"""Import detector or SMPL-fitting outputs as SO(3)^K measurement sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .measurements import validate_confidence
from .so3 import axis_angle_to_matrix, project_to_so3

SUPPORTED_SUFFIXES = {".npz", ".json"}
ROTATION_MATRIX_KEYS = (
    "observations",
    "rotations",
    "body_rotations",
    "body_pose_rotmat",
    "pose_rotmat",
    "pose_rotmats",
    "pred_rotmat",
    "pred_rotmats",
    "full_pose_rotmat",
)
AXIS_ANGLE_KEYS = (
    "poses",
    "body_pose",
    "smpl_body_pose",
    "pred_body_pose",
    "pose_body",
    "pose",
    "full_pose",
    "axis_angle",
    "axis_angles",
)
QUATERNION_KEYS = (
    "quaternions",
    "body_quaternions",
    "body_quat",
    "pose_quat",
    "pose_quaternions",
    "pred_quat",
    "pred_quaternions",
)
CONFIDENCE_KEYS = (
    "confidence",
    "confidences",
    "joint_confidence",
    "joint_confidences",
    "body_pose_confidence",
    "scores",
    "score",
    "joint_scores",
    "keypoint_scores",
)
MASK_KEYS = ("mask", "masks", "visible", "visibility", "valid", "valid_mask")
JOINT_NOISE_RAD_KEYS = (
    "joint_noise_sigma_rad",
    "measurement_noise_sigma_rad",
    "joint_sigma_rad",
    "sigma_rad",
)
JOINT_NOISE_DEG_KEYS = (
    "joint_noise_sigma_deg",
    "measurement_noise_sigma_deg",
    "joint_sigma_deg",
    "sigma_deg",
)
FPS_KEYS = ("mocap_framerate", "mocap_frame_rate", "fps", "frame_rate", "source_fps")
NOISE_SIGMA_RAD_KEYS = ("noise_sigma_rad", "measurement_noise_sigma_rad")
NOISE_SIGMA_DEG_KEYS = ("noise_sigma_deg", "measurement_noise_sigma_deg")


@dataclass(frozen=True)
class ImportedMeasurements:
    """Detector/SMPL-fitting observations aligned with one pose sequence."""

    name: str
    observations: np.ndarray
    mask: np.ndarray
    confidence: np.ndarray
    noise_sigma_rad: float
    joint_noise_sigma_rad: np.ndarray | None = None
    source_fps: float | None = None
    frame_rate: int | None = None
    source_path: str = ""
    source: str = "detector"


def find_detector_measurement_files(data_root: str | Path, dataset_subset: str = "") -> list[Path]:
    """Find detector/SMPL-fitting files with supported suffixes under a directory."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"measurement data_root does not exist: {root}")
    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported detector output suffix: {root.suffix}")
        return [root]
    files = sorted(path for path in root.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES)
    if dataset_subset:
        needle = dataset_subset.lower()
        files = [path for path in files if needle in str(path.relative_to(root)).lower()]
    return files


def _json_to_arrays(payload: Any) -> dict[str, np.ndarray]:
    if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
        frames = payload["frames"]
        keys = sorted({key for frame in frames if isinstance(frame, dict) for key in frame})
        arrays: dict[str, np.ndarray] = {}
        for key in keys:
            values = [frame.get(key) for frame in frames]
            if all(value is not None for value in values):
                try:
                    arrays[key] = np.asarray(values, dtype=np.float64)
                except (TypeError, ValueError):
                    continue
        for key, value in payload.items():
            if key == "frames":
                continue
            try:
                arrays[key] = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                continue
        return arrays
    if isinstance(payload, dict):
        arrays = {}
        for key, value in payload.items():
            try:
                arrays[key] = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                continue
        return arrays
    raise ValueError("JSON detector output must be an object or an object with a 'frames' list")


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as npz:
            return {key: np.asarray(npz[key]) for key in npz.files}
    if path.suffix.lower() == ".json":
        return _json_to_arrays(json.loads(path.read_text(encoding="utf-8")))
    raise ValueError(f"unsupported detector output suffix: {path.suffix}")


def _first_key(arrays: Mapping[str, np.ndarray], keys: tuple[str, ...], requested: str | None = None) -> str | None:
    if requested:
        if requested not in arrays:
            raise ValueError(f"requested key '{requested}' is not present; available keys: {sorted(arrays)}")
        return requested
    for key in keys:
        if key in arrays:
            return key
    return None


def _source_fps(arrays: Mapping[str, np.ndarray]) -> float | None:
    for key in FPS_KEYS:
        if key in arrays:
            value = float(np.asarray(arrays[key]).reshape(-1)[0])
            if np.isfinite(value) and value > 0.0:
                return value
    return None


def _stride(source_fps: float | None, frame_rate: int | None) -> int:
    if frame_rate is None or source_fps is None:
        return 1
    if int(frame_rate) <= 0:
        raise ValueError("frame_rate must be positive")
    return max(1, int(round(float(source_fps) / float(frame_rate))))


def _scalar_noise_rad(arrays: Mapping[str, np.ndarray], fallback_noise_deg: float) -> float:
    for key in NOISE_SIGMA_RAD_KEYS:
        if key in arrays:
            value = float(np.asarray(arrays[key], dtype=np.float64).reshape(-1)[0])
            if np.isfinite(value) and value > 0.0:
                return value
            raise ValueError(f"{key} must be positive and finite")
    for key in NOISE_SIGMA_DEG_KEYS:
        if key in arrays:
            value = float(np.asarray(arrays[key], dtype=np.float64).reshape(-1)[0])
            if np.isfinite(value) and value > 0.0:
                return float(np.radians(value))
            raise ValueError(f"{key} must be positive and finite")
    return float(np.radians(float(fallback_noise_deg)))


def _is_body_only_key(key: str) -> bool:
    lower = key.lower()
    return "body" in lower and "full" not in lower


def _select_body_joint_axis(values: np.ndarray, num_joints: int, key: str) -> np.ndarray:
    if values.ndim < 2:
        raise ValueError(f"{key} must include a joint axis, got shape {values.shape}")
    joint_count = values.shape[1]
    if joint_count == num_joints:
        return values
    if joint_count > num_joints:
        start = 0 if _is_body_only_key(key) else 1
        if start + num_joints <= joint_count:
            return values[:, start : start + num_joints]
    raise ValueError(f"{key} has {joint_count} joints; need {num_joints} body joints")


def _axis_angle_to_rotations(values: np.ndarray, num_joints: int, key: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim == 3 and values.shape[-1] == 3:
        return axis_angle_to_matrix(_select_body_joint_axis(values, num_joints, key))
    if values.ndim != 2:
        raise ValueError(f"{key} axis-angle array must be shaped [T, D] or [T, J, 3], got {values.shape}")
    body_dims = num_joints * 3
    if values.shape[1] == body_dims:
        body = values.reshape(values.shape[0], num_joints, 3)
    elif values.shape[1] >= body_dims + 3:
        start = 0 if _is_body_only_key(key) else 3
        body = values[:, start : start + body_dims].reshape(values.shape[0], num_joints, 3)
    else:
        raise ValueError(f"{key} has {values.shape[1]} axis-angle dims; need {body_dims} body dims or full pose with root")
    return axis_angle_to_matrix(body)


def _matrix_to_rotations(values: np.ndarray, num_joints: int, key: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] % 9 == 0:
        values = values.reshape(values.shape[0], values.shape[1] // 9, 3, 3)
    elif values.ndim == 3 and values.shape[-1] == 9:
        values = values.reshape(values.shape[0], values.shape[1], 3, 3)
    elif values.ndim == 3 and values.shape[-2:] == (3, 3):
        values = values[None, ...]
    if values.ndim != 4 or values.shape[-2:] != (3, 3):
        raise ValueError(f"{key} rotation-matrix array must be shaped [T, J, 3, 3] or flat 9D blocks, got {values.shape}")
    return _select_body_joint_axis(values, num_joints, key)


def _canonicalize_quaternions(quaternions: np.ndarray, order: str) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float64)
    if order not in {"xyzw", "wxyz"}:
        raise ValueError("quaternion_order must be 'xyzw' or 'wxyz'")
    if q.shape[-1:] != (4,):
        raise ValueError(f"expected quaternions shaped (..., 4), got {q.shape}")
    if order == "wxyz":
        q = q[..., [1, 2, 3, 0]]
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("cannot normalize invalid or zero-length quaternion")
    q = q / norms
    return np.where(q[..., 3:4] < 0.0, -q, q)


def _quaternions_to_rotations(quaternions: np.ndarray, order: str) -> np.ndarray:
    q = _canonicalize_quaternions(quaternions, order)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rotations = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    rotations[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[..., 0, 1] = 2.0 * (x * y - z * w)
    rotations[..., 0, 2] = 2.0 * (x * z + y * w)
    rotations[..., 1, 0] = 2.0 * (x * y + z * w)
    rotations[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[..., 1, 2] = 2.0 * (y * z - x * w)
    rotations[..., 2, 0] = 2.0 * (x * z - y * w)
    rotations[..., 2, 1] = 2.0 * (y * z + x * w)
    rotations[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return project_to_so3(rotations)


def _quaternion_to_rotations(values: np.ndarray, num_joints: int, key: str, order: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] % 4 == 0:
        values = values.reshape(values.shape[0], values.shape[1] // 4, 4)
    elif values.ndim == 2 and values.shape[-1] == 4:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[-1] != 4:
        raise ValueError(f"{key} quaternion array must be shaped [T, J, 4] or flat 4D blocks, got {values.shape}")
    return _quaternions_to_rotations(_select_body_joint_axis(values, num_joints, key), order)


def _load_rotations(
    arrays: Mapping[str, np.ndarray],
    num_joints: int,
    pose_key: str | None,
    quaternion_order: str,
) -> tuple[np.ndarray, str]:
    requested_key = _first_key(arrays, tuple(arrays), pose_key) if pose_key else None
    candidates: tuple[str, ...]
    if requested_key is not None:
        candidates = (requested_key,)
    else:
        candidates = (*ROTATION_MATRIX_KEYS, *AXIS_ANGLE_KEYS, *QUATERNION_KEYS)
    errors: list[str] = []
    for key in candidates:
        if key not in arrays:
            continue
        try:
            raw_shape = np.asarray(arrays[key]).shape
            lower_key = key.lower()
            if key in ROTATION_MATRIX_KEYS or raw_shape[-2:] == (3, 3) or lower_key.endswith("rotmat"):
                return _matrix_to_rotations(arrays[key], num_joints, key), key
            if key in QUATERNION_KEYS or "quat" in lower_key or (len(raw_shape) >= 2 and raw_shape[-1:] == (4,)):
                return _quaternion_to_rotations(arrays[key], num_joints, key, quaternion_order), key
            return _axis_angle_to_rotations(arrays[key], num_joints, key), key
        except ValueError as exc:
            errors.append(f"{key}: {exc}")
    detail = "; ".join(errors[:5])
    raise ValueError(f"could not find usable detector pose keys. Tried {list(candidates)}. {detail}")


def _joint_values(values: np.ndarray, num_joints: int, key: str, t_steps: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        return np.full((t_steps, num_joints), float(values), dtype=np.float64)
    if values.ndim == 1:
        if values.shape[0] == t_steps:
            return np.repeat(values[:, None], num_joints, axis=1)
        if values.shape[0] == num_joints:
            return np.repeat(values[None, :], t_steps, axis=0)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim == 2:
        if values.shape == (t_steps, num_joints):
            return values
        if values.shape[0] == t_steps and values.shape[1] > num_joints:
            return _select_body_joint_axis(values, num_joints, key)
    raise ValueError(f"{key} must broadcast to [T, J]={(t_steps, num_joints)}, got {values.shape}")


def _apply_stride(values: np.ndarray, stride: int) -> np.ndarray:
    return np.asarray(values)[::stride]


def _load_confidence(
    arrays: Mapping[str, np.ndarray],
    num_joints: int,
    t_steps: int,
    stride: int,
    key: str | None,
    confidence_scale: float,
) -> np.ndarray:
    confidence_key = _first_key(arrays, CONFIDENCE_KEYS, key)
    if confidence_key is None:
        confidence = np.ones((t_steps, num_joints), dtype=np.float64)
    else:
        confidence = _joint_values(_apply_stride(arrays[confidence_key], stride), num_joints, confidence_key, t_steps)
    if confidence_scale <= 0.0 or not np.isfinite(confidence_scale):
        raise ValueError("confidence_scale must be positive and finite")
    confidence = np.where(np.isfinite(confidence / confidence_scale), confidence / confidence_scale, 0.0)
    return validate_confidence(confidence, (t_steps, num_joints))


def _load_mask(arrays: Mapping[str, np.ndarray], num_joints: int, t_steps: int, stride: int, key: str | None) -> np.ndarray | None:
    mask_key = _first_key(arrays, MASK_KEYS, key)
    if mask_key is None:
        return None
    return _joint_values(_apply_stride(arrays[mask_key], stride), num_joints, mask_key, t_steps).astype(bool)


def _load_joint_noise(
    arrays: Mapping[str, np.ndarray],
    num_joints: int,
    t_steps: int,
    stride: int,
    key: str | None,
) -> np.ndarray | None:
    selected_key = key
    use_degrees = False
    if selected_key is None:
        selected_key = _first_key(arrays, JOINT_NOISE_RAD_KEYS)
        if selected_key is None:
            selected_key = _first_key(arrays, JOINT_NOISE_DEG_KEYS)
            use_degrees = selected_key is not None
    else:
        if selected_key not in arrays:
            raise ValueError(f"requested joint noise key '{selected_key}' is not present; available keys: {sorted(arrays)}")
        use_degrees = selected_key in JOINT_NOISE_DEG_KEYS or selected_key.endswith("_deg")
    if selected_key is None:
        return None
    noise = _joint_values(_apply_stride(arrays[selected_key], stride), num_joints, selected_key, t_steps)
    if use_degrees:
        noise = np.radians(noise)
    if np.any(~np.isfinite(noise)) or np.any(noise <= 0.0):
        raise ValueError("joint measurement noise values must be positive and finite")
    return noise


def _sanitize_rotations(rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.all(np.isfinite(rotations), axis=(-1, -2))
    safe = np.asarray(rotations, dtype=np.float64).copy()
    safe[~finite] = np.eye(3)
    return project_to_so3(safe), finite


def load_detector_measurement(
    path: str | Path,
    frame_rate: int | None = 20,
    num_joints: int = 23,
    noise_deg: float = 10.0,
    pose_key: str | None = None,
    mask_key: str | None = None,
    confidence_key: str | None = None,
    joint_noise_key: str | None = None,
    confidence_scale: float = 1.0,
    quaternion_order: str = "xyzw",
    name: str | None = None,
) -> ImportedMeasurements:
    """Load one real detector/SMPL-fitting output as filter measurements."""
    path = Path(path)
    arrays = _load_arrays(path)
    source_fps = _source_fps(arrays)
    stride = _stride(source_fps, frame_rate)
    rotations, used_pose_key = _load_rotations(arrays, num_joints, pose_key, quaternion_order)
    observations, finite_mask = _sanitize_rotations(_apply_stride(rotations, stride))
    t_steps = observations.shape[0]
    if t_steps < 1:
        raise ValueError(f"{path} has no detector frames after downsampling")

    confidence = _load_confidence(arrays, num_joints, t_steps, stride, confidence_key, confidence_scale)
    explicit_mask = _load_mask(arrays, num_joints, t_steps, stride, mask_key)
    mask = finite_mask if explicit_mask is None else finite_mask & explicit_mask
    mask = mask & (confidence > 0.0)
    confidence = np.where(mask, confidence, 0.0)
    joint_noise_sigma_rad = _load_joint_noise(arrays, num_joints, t_steps, stride, joint_noise_key)

    return ImportedMeasurements(
        name=name or path.stem,
        observations=observations,
        mask=mask,
        confidence=confidence,
        noise_sigma_rad=_scalar_noise_rad(arrays, noise_deg),
        joint_noise_sigma_rad=joint_noise_sigma_rad,
        source_fps=source_fps,
        frame_rate=frame_rate,
        source_path=str(path),
        source=f"detector:{used_pose_key}",
    )


def load_detector_measurement_dataset(
    data_root: str | Path,
    dataset_subset: str,
    frame_rate: int,
    num_joints: int,
    noise_deg: float,
    max_sequences: int | None = None,
    min_frames: int = 1,
    pose_key: str | None = None,
    mask_key: str | None = None,
    confidence_key: str | None = None,
    joint_noise_key: str | None = None,
    confidence_scale: float = 1.0,
    quaternion_order: str = "xyzw",
) -> dict[str, ImportedMeasurements]:
    """Load a directory of detector measurements keyed by sequence/file stem."""
    files = find_detector_measurement_files(data_root, dataset_subset)
    if max_sequences is not None:
        files = files[: int(max_sequences)]
    measurements: dict[str, ImportedMeasurements] = {}
    errors: list[str] = []
    for path in files:
        try:
            measurement = load_detector_measurement(
                path,
                frame_rate=frame_rate,
                num_joints=num_joints,
                noise_deg=noise_deg,
                pose_key=pose_key,
                mask_key=mask_key,
                confidence_key=confidence_key,
                joint_noise_key=joint_noise_key,
                confidence_scale=confidence_scale,
                quaternion_order=quaternion_order,
            )
            if measurement.observations.shape[0] >= min_frames:
                measurements[measurement.name] = measurement
        except Exception as exc:  # Keep scanning mixed detector-output directories.
            errors.append(f"{path}: {exc}")
    if not measurements:
        detail = "\n".join(errors[:5])
        raise ValueError(f"no usable detector measurement files found under {data_root}\n{detail}")
    return measurements


def save_imported_measurements(path: str | Path, measurements: ImportedMeasurements) -> None:
    """Write a standardized `.npz` measurement bundle for later experiment configs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "observations": measurements.observations,
        "mask": measurements.mask.astype(np.bool_),
        "confidence": measurements.confidence,
        "noise_sigma_rad": np.asarray(measurements.noise_sigma_rad, dtype=np.float64),
    }
    if measurements.joint_noise_sigma_rad is not None:
        payload["joint_noise_sigma_rad"] = measurements.joint_noise_sigma_rad
    if measurements.source_fps is not None:
        payload["source_fps"] = np.asarray(measurements.source_fps, dtype=np.float64)
    if measurements.frame_rate is not None:
        payload["frame_rate"] = np.asarray(measurements.frame_rate, dtype=np.float64)
    np.savez_compressed(path, **payload)

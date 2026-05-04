from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_filter.so3 import axis_angle_to_matrix, geodesic_distance  # noqa: E402

MOTION_BINS = (
    ("low_motion", 0.0, 0.5),
    ("medium_motion", 0.5, 1.5),
    ("high_motion", 1.5, float("inf")),
)


@dataclass(frozen=True)
class WindowCandidate:
    source_path: str
    source_fps: float
    source_stride: int
    source_frames: int
    source_start_frame: int
    source_stop_frame: int
    target_start_frame: int
    target_stop_frame: int
    target_frames: int
    motion_deg_per_frame: float
    rank: int | None = None
    output_path: str | None = None


def find_amass_files(
    data_root: Path,
    dataset_subset: str = "",
    max_files: int | None = None,
) -> list[Path]:
    if data_root.is_file():
        files = [data_root]
    else:
        files = sorted(data_root.rglob("*.npz"))

    if dataset_subset:
        needle = dataset_subset.lower()
        files = [path for path in files if needle in str(path).lower()]
    if max_files is not None:
        files = files[: int(max_files)]
    return files


def _source_fps(npz: np.lib.npyio.NpzFile) -> float:
    for key in ("mocap_framerate", "mocap_frame_rate", "fps"):
        if key in npz:
            return float(np.asarray(npz[key]).reshape(-1)[0])
    return 60.0


def _window_starts(n_frames: int, window_frames: int, stride_frames: int) -> list[int]:
    if n_frames <= window_frames:
        return [0]
    last_start = n_frames - window_frames
    starts = list(range(0, last_start + 1, max(1, int(stride_frames))))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _motion_deg_per_frame(rotations: np.ndarray) -> np.ndarray:
    if rotations.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    return np.degrees(np.mean(geodesic_distance(rotations[:-1], rotations[1:]), axis=1))


def _motion_bin(motion_deg_per_frame: float) -> str:
    motion = float(motion_deg_per_frame)
    for name, lower, upper in MOTION_BINS:
        if lower <= motion < upper:
            return name
    return MOTION_BINS[-1][0]


def candidate_windows_for_file(
    path: Path,
    frame_rate: int,
    num_joints: int,
    segment_frames: int,
    stride_frames: int,
) -> list[WindowCandidate]:
    with np.load(path, allow_pickle=False) as data:
        if "poses" not in data:
            raise ValueError(f"{path} has no poses array")
        poses = np.asarray(data["poses"], dtype=np.float64)
        fps = _source_fps(data)

    if poses.ndim != 2:
        raise ValueError(f"{path} poses must be shaped [T, D], got {poses.shape}")
    needed = 3 + int(num_joints) * 3
    if poses.shape[1] < needed:
        raise ValueError(f"{path} poses has {poses.shape[1]} dims, need at least {needed}")

    source_stride = max(1, int(round(fps / float(frame_rate))))
    body_axis_angle = poses[::source_stride, 3:needed].reshape(-1, int(num_joints), 3)
    if body_axis_angle.shape[0] < 2:
        return []

    segment_frames = min(int(segment_frames), body_axis_angle.shape[0])
    rotations = axis_angle_to_matrix(body_axis_angle)
    velocities = _motion_deg_per_frame(rotations)

    candidates = []
    for target_start in _window_starts(
        body_axis_angle.shape[0],
        window_frames=segment_frames,
        stride_frames=int(stride_frames),
    ):
        target_stop = min(target_start + segment_frames, body_axis_angle.shape[0])
        source_start = target_start * source_stride
        source_stop = min(target_stop * source_stride, poses.shape[0])
        velocity_window = velocities[target_start : max(target_start, target_stop - 1)]
        motion = float(np.mean(velocity_window)) if velocity_window.size else 0.0
        candidates.append(
            WindowCandidate(
                source_path=str(path),
                source_fps=float(fps),
                source_stride=int(source_stride),
                source_frames=int(poses.shape[0]),
                source_start_frame=int(source_start),
                source_stop_frame=int(source_stop),
                target_start_frame=int(target_start),
                target_stop_frame=int(target_stop),
                target_frames=int(target_stop - target_start),
                motion_deg_per_frame=motion,
            )
        )
    return candidates


def collect_candidates(
    data_root: Path,
    dataset_subset: str,
    max_files: int | None,
    frame_rate: int,
    num_joints: int,
    segment_frames: int,
    stride_frames: int,
) -> tuple[list[WindowCandidate], list[str]]:
    candidates = []
    errors = []
    for path in find_amass_files(data_root, dataset_subset=dataset_subset, max_files=max_files):
        try:
            candidates.extend(
                candidate_windows_for_file(
                    path,
                    frame_rate=frame_rate,
                    num_joints=num_joints,
                    segment_frames=segment_frames,
                    stride_frames=stride_frames,
                )
            )
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return candidates, errors


def select_windows(
    candidates: list[WindowCandidate],
    max_segments: int,
    selection: str,
    min_motion_deg_per_frame: float,
    max_per_file: int | None,
) -> list[WindowCandidate]:
    filtered = [
        candidate
        for candidate in candidates
        if candidate.motion_deg_per_frame >= float(min_motion_deg_per_frame)
    ]
    if selection == "first":
        ordered = filtered
    elif selection == "uniform":
        if len(filtered) <= max_segments:
            ordered = filtered
        else:
            indices = np.linspace(0, len(filtered) - 1, num=max_segments, dtype=int)
            ordered = [filtered[int(idx)] for idx in indices]
    elif selection == "top-motion":
        ordered = sorted(filtered, key=lambda row: row.motion_deg_per_frame, reverse=True)
    elif selection == "balanced-motion":
        by_bin: dict[str, list[WindowCandidate]] = {
            name: [] for name, _, _ in MOTION_BINS
        }
        for candidate in filtered:
            by_bin[_motion_bin(candidate.motion_deg_per_frame)].append(candidate)
        for name in by_bin:
            by_bin[name].sort(key=lambda row: row.motion_deg_per_frame, reverse=True)
        per_bin = max(1, int(max_segments) // len(MOTION_BINS))
        remainder = int(max_segments) - per_bin * len(MOTION_BINS)
        ordered = []
        for idx, (name, _, _) in enumerate(MOTION_BINS):
            take = per_bin + (1 if idx < remainder else 0)
            ordered.extend(by_bin[name][:take])
        if len(ordered) < int(max_segments):
            selected_ids = {id(candidate) for candidate in ordered}
            fill = [
                candidate
                for candidate in sorted(
                    filtered, key=lambda row: row.motion_deg_per_frame, reverse=True
                )
                if id(candidate) not in selected_ids
            ]
            ordered.extend(fill[: int(max_segments) - len(ordered)])
    else:
        raise ValueError(f"unknown selection: {selection}")

    selected = []
    per_file_counts: dict[str, int] = {}
    for candidate in ordered:
        count = per_file_counts.get(candidate.source_path, 0)
        if max_per_file is not None and count >= int(max_per_file):
            continue
        selected.append(candidate)
        per_file_counts[candidate.source_path] = count + 1
        if len(selected) >= int(max_segments):
            break
    return selected


def _slice_payload(data: np.lib.npyio.NpzFile, start: int, stop: int) -> dict:
    poses = data["poses"]
    payload = {}
    for key in data.files:
        value = data[key]
        if value.shape[:1] == poses.shape[:1]:
            payload[key] = value[start:stop]
        else:
            payload[key] = value
    return payload


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")[:80] or "segment"


def materialize_windows(
    selected: list[WindowCandidate],
    output_dir: Path,
    clean: bool = True,
) -> list[WindowCandidate]:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    materialized = []
    for rank, candidate in enumerate(selected):
        source = Path(candidate.source_path)
        output_path = output_dir / f"segment_{rank:03d}_{_safe_stem(source)}.npz"
        with np.load(source, allow_pickle=False) as data:
            np.savez(
                output_path,
                **_slice_payload(
                    data,
                    candidate.source_start_frame,
                    candidate.source_stop_frame,
                ),
            )
        materialized.append(
            WindowCandidate(
                **{
                    **asdict(candidate),
                    "rank": rank,
                    "output_path": str(output_path),
                }
            )
        )
    return materialized


def write_manifest(path: Path, rows: list[WindowCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(WindowCandidate.__annotations__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def prepare_windows(
    data_root: Path,
    output_dir: Path,
    report_path: Path,
    manifest_path: Path,
    dataset_subset: str = "",
    frame_rate: int = 20,
    num_joints: int = 23,
    segment_frames: int = 80,
    stride_frames: int = 40,
    max_files: int | None = None,
    max_segments: int = 24,
    selection: str = "top-motion",
    min_motion_deg_per_frame: float = 0.0,
    max_per_file: int | None = None,
    clean: bool = True,
) -> dict:
    candidates, errors = collect_candidates(
        data_root,
        dataset_subset=dataset_subset,
        max_files=max_files,
        frame_rate=frame_rate,
        num_joints=num_joints,
        segment_frames=segment_frames,
        stride_frames=stride_frames,
    )
    selected = select_windows(
        candidates,
        max_segments=max_segments,
        selection=selection,
        min_motion_deg_per_frame=min_motion_deg_per_frame,
        max_per_file=max_per_file,
    )
    if not selected:
        detail = "\n".join(errors[:5])
        raise ValueError(f"no AMASS windows selected from {data_root}\n{detail}")

    materialized = materialize_windows(selected, output_dir=output_dir, clean=clean)
    write_manifest(manifest_path, materialized)
    report = {
        "data_root": str(data_root),
        "dataset_subset": dataset_subset,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "frame_rate": int(frame_rate),
        "num_joints": int(num_joints),
        "segment_frames": int(segment_frames),
        "stride_frames": int(stride_frames),
        "max_files": max_files,
        "max_segments": int(max_segments),
        "selection": selection,
        "min_motion_deg_per_frame": float(min_motion_deg_per_frame),
        "max_per_file": max_per_file,
        "candidate_count": len(candidates),
        "selected_count": len(materialized),
        "errors": errors[:20],
        "selected": [asdict(row) for row in materialized],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select high-motion AMASS windows and materialize them as .npz segments."
    )
    parser.add_argument("--data-root", required=True, help="AMASS root directory or .npz file")
    parser.add_argument("--dataset-subset", default="", help="Optional path substring filter")
    parser.add_argument("--output-dir", required=True, help="Output directory for selected windows")
    parser.add_argument("--report", required=True, help="Output JSON report path")
    parser.add_argument("--manifest", required=True, help="Output CSV manifest path")
    parser.add_argument("--frame-rate", type=int, default=20)
    parser.add_argument("--num-joints", type=int, default=23)
    parser.add_argument("--segment-frames", type=int, default=80, help="Window length after downsampling")
    parser.add_argument("--stride-frames", type=int, default=40, help="Window stride after downsampling")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum source files to scan")
    parser.add_argument("--max-segments", type=int, default=24, help="Maximum windows to materialize")
    parser.add_argument(
        "--selection",
        choices=("top-motion", "balanced-motion", "first", "uniform"),
        default="top-motion",
        help="Window selection strategy",
    )
    parser.add_argument("--min-motion-deg-per-frame", type=float, default=0.0)
    parser.add_argument("--max-per-file", type=int, default=None)
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear output-dir first")
    args = parser.parse_args()

    report = prepare_windows(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        report_path=Path(args.report),
        manifest_path=Path(args.manifest),
        dataset_subset=args.dataset_subset,
        frame_rate=args.frame_rate,
        num_joints=args.num_joints,
        segment_frames=args.segment_frames,
        stride_frames=args.stride_frames,
        max_files=args.max_files,
        max_segments=args.max_segments,
        selection=args.selection,
        min_motion_deg_per_frame=args.min_motion_deg_per_frame,
        max_per_file=args.max_per_file,
        clean=not args.keep_existing,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

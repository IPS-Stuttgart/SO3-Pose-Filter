from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _segment_starts(
    n_frames: int,
    segment_frames: int,
    stride_frames: int,
    min_segments: int,
) -> list[int]:
    if n_frames <= segment_frames:
        return [0]

    last_start = n_frames - segment_frames
    starts = list(range(0, last_start + 1, max(1, stride_frames)))
    if starts[-1] != last_start:
        starts.append(last_start)

    if len(starts) >= min_segments:
        return starts

    return sorted(
        {
            int(round(x))
            for x in np.linspace(0, last_start, num=min_segments, endpoint=True)
        }
    )


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


def prepare_segments(
    source: Path,
    output_dir: Path,
    max_frames: int | None,
    segment_frames: int,
    stride_frames: int,
    min_segments: int,
) -> dict:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(source, allow_pickle=False) as data:
        if "poses" not in data:
            raise ValueError(f"{source} has no poses array")
        poses = data["poses"]
        n_source_frames = int(poses.shape[0])
        n_frames = n_source_frames if max_frames is None else min(n_source_frames, max_frames)
        if n_frames < 2:
            raise ValueError(f"{source} has too few frames: {n_source_frames}")
        segment_frames = min(int(segment_frames), n_frames)
        starts = _segment_starts(
            n_frames,
            segment_frames=segment_frames,
            stride_frames=int(stride_frames),
            min_segments=int(min_segments),
        )
        segments = []
        for idx, start in enumerate(starts):
            stop = min(start + segment_frames, n_frames)
            path = output_dir / f"segment_{idx:03d}.npz"
            np.savez(path, **_slice_payload(data, start, stop))
            segments.append(
                {
                    "path": str(path),
                    "start_frame": int(start),
                    "stop_frame": int(stop),
                    "frames": int(stop - start),
                }
            )

    return {
        "source": str(source),
        "output_dir": str(output_dir),
        "source_frames": n_source_frames,
        "used_frames": n_frames,
        "segment_frames": segment_frames,
        "stride_frames": int(stride_frames),
        "segments": segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create AMASS sequence chunks for first-results benchmarks."
    )
    parser.add_argument("--source", required=True, help="Source AMASS .npz")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--report", required=True, help="Output JSON report")
    parser.add_argument("--max-frames", type=int, default=720)
    parser.add_argument("--segment-frames", type=int, default=180)
    parser.add_argument("--stride-frames", type=int, default=120)
    parser.add_argument("--min-segments", type=int, default=3)
    args = parser.parse_args()

    report = prepare_segments(
        source=Path(args.source),
        output_dir=Path(args.output_dir),
        max_frames=args.max_frames,
        segment_frames=args.segment_frames,
        stride_frames=args.stride_frames,
        min_segments=args.min_segments,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

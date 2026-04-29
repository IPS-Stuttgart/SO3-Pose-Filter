from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_filter.evaluation import write_csv, write_json  # noqa: E402
from pose_filter.experiment import load_config  # noqa: E402
from prepare_amass_windows import prepare_windows  # noqa: E402
from run_first_results_benchmark import run_first_results_benchmark  # noqa: E402

DEFAULT_METHODS = (
    "raw",
    "persistence",
    "gaussian_rw",
    "mlp_delta",
    "history_mlp_delta",
    "gru_delta",
)


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def _int_list(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(value)]


def _float_list(value: Any, default: list[float]) -> list[float]:
    if value is None:
        return default
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(value)]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _aggregate_method_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in rows})
    out = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        out.append(
            {
                "method": method,
                "mean_tracking_error_deg": _mean(
                    [float(row["tracking_error_deg"]) for row in method_rows]
                ),
                "mean_improvement_vs_raw_deg": _mean(
                    [float(row["improvement_vs_raw_deg"]) for row in method_rows]
                ),
                "mean_improvement_vs_persistence_deg": _mean(
                    [
                        float(row["improvement_vs_persistence_deg"])
                        for row in method_rows
                    ]
                ),
                "row_count": len(method_rows),
            }
        )
    return out


def _copy_row_with_run(row: dict[str, str], run: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "seed": int(run["seed"]),
        "num_particles": int(run["num_particles"]),
        "run_dir": str(run["output_dir"]),
    }
    out.update(row)
    return out


def _benchmark_config(
    config: dict[str, Any],
    segments_dir: Path,
    output_dir: Path,
    seed: int,
    num_particles: int,
) -> dict[str, Any]:
    run_config = dict(config)
    run_config.update(
        {
            "data_root": str(segments_dir),
            "dataset_subset": "",
            "seed": int(seed),
            "num_particles": int(num_particles),
            "output_dir": str(output_dir / "base"),
            "benchmark_output_dir": str(output_dir),
        }
    )
    checkpoint_dir = output_dir / "checkpoints"
    run_config.setdefault(
        "mlp_transition_checkpoint",
        str(checkpoint_dir / "mlp_delta_checkpoint.npz"),
    )
    run_config.setdefault(
        "history_transition_checkpoint",
        str(checkpoint_dir / "history_mlp_delta_checkpoint.npz"),
    )
    run_config.setdefault(
        "gru_transition_checkpoint",
        str(checkpoint_dir / "gru_delta_checkpoint.npz"),
    )
    return run_config


def _write_markdown_summary(
    path: Path,
    summary: dict[str, Any],
    method_means: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Private ACCAD Evaluation",
        "",
        "Generated outputs are local artifacts and should stay under ignored output folders.",
        "",
        f"- source data root: `{summary['source_data_root']}`",
        f"- selected windows: `{summary['window_report']['selected_count']}`",
        f"- benchmark runs: `{len(summary['runs'])}`",
        f"- best method: `{summary['best_method']}`",
        "",
        "| method | mean tracking error (deg) | improvement vs raw (deg) | improvement vs persistence (deg) | rows |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in method_means:
        lines.append(
            "| {method} | {mean_tracking_error_deg:.4g} | {mean_improvement_vs_raw_deg:.4g} | {mean_improvement_vs_persistence_deg:.4g} | {row_count} |".format(
                **row
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_private_accad_eval(
    config: dict[str, Any],
    *,
    source_data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    prepare: bool = True,
) -> dict[str, Any]:
    """Run a private full-ACCAD evaluation into ignored local output folders."""

    root = _as_path(
        source_data_root
        or config.get("source_data_root")
        or config.get("data_root", "D:/Uni-Data/ACCAD")
    )
    out_dir = _as_path(output_dir or config.get("output_dir", "runs/private_accad_eval"))
    segments_dir = _as_path(config.get("segments_dir", out_dir / "segments"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if prepare:
        window_report = prepare_windows(
            data_root=root,
            output_dir=segments_dir,
            report_path=out_dir / "window_selection_report.json",
            manifest_path=out_dir / "window_manifest.csv",
            dataset_subset=str(config.get("dataset_subset", "")),
            frame_rate=int(config.get("frame_rate", 20)),
            num_joints=int(config.get("num_joints", 23)),
            segment_frames=int(config.get("segment_frames", 80)),
            stride_frames=int(config.get("stride_frames", 40)),
            max_files=(
                None
                if config.get("max_files") is None
                else int(config["max_files"])
            ),
            max_segments=int(config.get("max_segments", 48)),
            selection=str(config.get("selection", "top-motion")),
            min_motion_deg_per_frame=float(
                config.get("min_motion_deg_per_frame", 0.0)
            ),
            max_per_file=(
                None
                if config.get("max_per_file") is None
                else int(config["max_per_file"])
            ),
            clean=not bool(config.get("keep_existing_segments", False)),
        )
    else:
        window_report = {
            "data_root": str(root),
            "output_dir": str(segments_dir),
            "selected_count": None,
            "prepared": False,
        }

    methods = tuple(
        str(method)
        for method in config.get("benchmark_methods", list(DEFAULT_METHODS))
    )
    seeds = _int_list(config.get("benchmark_seeds"), [int(config.get("seed", 19))])
    particle_counts = _int_list(
        config.get("benchmark_num_particles"),
        [int(config.get("num_particles", 128))],
    )
    noise_grid = _float_list(
        config.get("benchmark_noise_deg"),
        [float(config.get("noise_deg", 10.0))],
    )
    occlusion_grid = _float_list(
        config.get("benchmark_occlusion_prob"),
        [float(config.get("occlusion_prob", 0.25))],
    )

    run_summaries = []
    all_metric_rows: list[dict[str, Any]] = []
    all_transition_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for num_particles in particle_counts:
            run_name = f"seed_{seed}_particles_{num_particles}"
            run_dir = out_dir / "benchmarks" / run_name
            run_config = _benchmark_config(
                config,
                segments_dir=segments_dir,
                output_dir=run_dir,
                seed=seed,
                num_particles=num_particles,
            )
            benchmark_summary = run_first_results_benchmark(
                run_config,
                run_dir,
                methods=methods,
                noise_grid=noise_grid,
                occlusion_grid=occlusion_grid,
            )
            run = {
                "seed": seed,
                "num_particles": num_particles,
                "output_dir": str(run_dir),
                "best_method": benchmark_summary["best_method"],
                "best_tracking_error_deg": benchmark_summary[
                    "best_tracking_error_deg"
                ],
            }
            run_summaries.append(run)
            for row in _read_csv(run_dir / "benchmark_metrics.csv"):
                all_metric_rows.append(_copy_row_with_run(row, run))
            for row in _read_csv(run_dir / "transition_metrics.csv"):
                tagged = _copy_row_with_run(row, run)
                all_transition_rows.append(tagged)

    aggregate_metrics_path = out_dir / "aggregate_benchmark_metrics.csv"
    aggregate_transition_path = out_dir / "aggregate_transition_metrics.csv"
    method_means_path = out_dir / "aggregate_method_means.csv"
    method_means = _aggregate_method_means(all_metric_rows)
    write_csv(aggregate_metrics_path, all_metric_rows)
    write_csv(aggregate_transition_path, all_transition_rows)
    write_csv(method_means_path, method_means)

    best_row = min(method_means, key=lambda row: row["mean_tracking_error_deg"])
    means_by_method = {
        row["method"]: float(row["mean_tracking_error_deg"]) for row in method_means
    }
    summary = {
        "source_data_root": str(root),
        "output_dir": str(out_dir),
        "segments_dir": str(segments_dir),
        "methods": list(methods),
        "seeds": seeds,
        "num_particles": particle_counts,
        "noise_deg": noise_grid,
        "occlusion_prob": occlusion_grid,
        "window_report": window_report,
        "runs": run_summaries,
        "method_means": method_means,
        "best_method": best_row["method"],
        "best_tracking_error_deg": best_row["mean_tracking_error_deg"],
        "history_mlp_beats_gaussian_rw": (
            means_by_method.get("history_mlp_delta", float("inf"))
            < means_by_method.get("gaussian_rw", float("-inf"))
        ),
        "gru_beats_gaussian_rw": (
            means_by_method.get("gru_delta", float("inf"))
            < means_by_method.get("gaussian_rw", float("-inf"))
        ),
        "mlp_beats_gaussian_rw": (
            means_by_method.get("mlp_delta", float("inf"))
            < means_by_method.get("gaussian_rw", float("-inf"))
        ),
        "outputs": {
            "aggregate_benchmark_metrics": str(aggregate_metrics_path),
            "aggregate_transition_metrics": str(aggregate_transition_path),
            "aggregate_method_means": str(method_means_path),
            "markdown_summary": str(out_dir / "private_accad_eval_summary.md"),
        },
    }
    write_json(out_dir / "private_accad_eval_summary.json", summary)
    _write_markdown_summary(
        out_dir / "private_accad_eval_summary.md",
        summary,
        method_means,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a private ACCAD evaluation into ignored local output folders."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override source AMASS/ACCAD root from config.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output directory from config.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Reuse an existing segments_dir instead of selecting windows.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    summary = run_private_accad_eval(
        config,
        source_data_root=args.data_root,
        output_dir=args.output,
        prepare=not args.skip_prepare,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

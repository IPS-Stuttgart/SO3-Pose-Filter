#!/usr/bin/env python3
"""Build paper-facing accuracy leaderboards from SO3-Pose-Filter artifacts.

This script consumes either

1. real detector/HMR evaluation outputs from ``scripts/run_detector_measurement_eval.py``
   containing ``detector_filter_metrics.csv`` and/or
   ``detector_measurement_eval_summary.json``; or
2. full-data motion-stratified benchmark artifacts containing
   ``aggregate_method_means_by_motion_bin.csv``.

It writes a ranked leaderboard as CSV, Markdown, LaTeX, and JSON. Lower tracking
error is ranked better. Improvements are positive when a method beats the raw
measurement or persistence baseline.

Example:

    python scripts/build_accuracy_leaderboard.py \
      --detector-run hmr_gaussian_rw=runs/hmr_gaussian_rw \
      --detector-run hmr_history_mlp=runs/hmr_history_mlp \
      --motion-run ACCAD=runs/full_data_accad_artifact \
      --motion-run KIT=runs/full_data_kit_artifact \
      --output-dir runs/accuracy_leaderboard

The script can also run detector/HMR evaluations itself from one base config:

    python scripts/build_accuracy_leaderboard.py \
      --eval-config configs/hmr_measurements.local.json \
      --method gaussian=gaussian_rw:numpy \
      --method hist=history_mlp_delta:numpy \
      --method gru=gru_delta:numpy \
      --output-dir runs/accuracy_leaderboard
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DETECTOR_SUMMARY_KEYS = [
    "observed_error_deg",
    "observed_joint_error_deg",
    "filter_error_deg",
    "persistence_error_deg",
    "filter_observed_joint_error_deg",
    "filter_occluded_joint_error_deg",
    "persistence_observed_joint_error_deg",
    "persistence_occluded_joint_error_deg",
    "mean_confidence",
    "observed_joint_fraction",
    "mean_ess",
    "min_ess",
    "final_ess",
    "resample_count",
    "resample_fraction",
    "mean_particle_spread_deg",
    "final_particle_spread_deg",
]

LEADERBOARD_COLUMNS = [
    "rank",
    "dataset",
    "source",
    "motion_bin",
    "noise_deg",
    "occlusion_prob",
    "method",
    "transition_model",
    "filter_backend",
    "num_particles",
    "tracking_error_deg",
    "visible_error_deg",
    "occluded_error_deg",
    "reappeared_error_deg",
    "raw_measurement_error_deg",
    "persistence_error_deg",
    "improvement_vs_raw_deg",
    "improvement_vs_persistence_deg",
    "mean_ess",
    "collapse_fraction",
    "row_count",
]

PAPER_SUMMARY_COLUMNS = [
    "dataset",
    "source",
    "motion_bin",
    "method",
    "condition_count",
    "win_count",
    "mean_tracking_error_deg",
    "mean_raw_measurement_error_deg",
    "mean_persistence_error_deg",
    "mean_improvement_vs_raw_deg",
    "mean_improvement_vs_persistence_deg",
    "worse_than_raw_count",
    "worse_than_persistence_count",
    "row_count",
]

RAW_METHODS = {"raw", "raw_measurement"}
PERSISTENCE_METHODS = {"persistence", "deterministic_persistence_pf"}


@dataclass(frozen=True)
class DetectorRunSpec:
    method: str
    path: Path


@dataclass(frozen=True)
class MotionRunSpec:
    dataset: str
    path: Path


@dataclass(frozen=True)
class MethodSpec:
    label: str
    transition_model: str
    filter_backend: str
    num_particles: int | None = None


def _is_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _float_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _int_or_blank(value: Any) -> int | str:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return ""
    return number


def _nanmean(values: Iterable[Any]) -> float:
    numbers = [_float_or_nan(value) for value in values]
    valid = [value for value in numbers if math.isfinite(value)]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))


def _format_float(value: Any, digits: int = 4) -> str:
    number = _float_or_nan(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}g}"


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return _format_float(value)
    if value is None:
        return ""
    return str(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADERBOARD_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in LEADERBOARD_COLUMNS})


def _write_table_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _escape_markdown(value: Any) -> str:
    return _format_cell(value).replace("|", "\\|")


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "rank",
        "dataset",
        "source",
        "motion bin",
        "noise",
        "occlusion",
        "method",
        "tracking (deg)",
        "visible",
        "occluded",
        "reappeared",
        "delta raw",
        "delta pers.",
        "ESS",
        "collapse",
        "rows",
    ]
    keys = [
        "rank",
        "dataset",
        "source",
        "motion_bin",
        "noise_deg",
        "occlusion_prob",
        "method",
        "tracking_error_deg",
        "visible_error_deg",
        "occluded_error_deg",
        "reappeared_error_deg",
        "improvement_vs_raw_deg",
        "improvement_vs_persistence_deg",
        "mean_ess",
        "collapse_fraction",
        "row_count",
    ]
    lines = [
        "# Accuracy leaderboard",
        "",
        "Lower tracking error is better. Improvements are positive when a method beats the corresponding baseline.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_markdown(row.get(key, "")) for key in keys) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _escape_latex(value: Any) -> str:
    text = _format_cell(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "rank",
        "dataset",
        "source",
        "motion_bin",
        "noise_deg",
        "occlusion_prob",
        "method",
        "tracking_error_deg",
        "visible_error_deg",
        "occluded_error_deg",
        "reappeared_error_deg",
        "improvement_vs_raw_deg",
        "improvement_vs_persistence_deg",
    ]
    headers = [
        "Rank",
        "Dataset",
        "Source",
        "Motion bin",
        "Noise",
        "Occl. prob.",
        "Method",
        "Track.",
        "Vis.",
        "Occl.",
        "Reapp.",
        r"$\Delta$ raw",
        r"$\Delta$ pers.",
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Accuracy leaderboard. Lower tracking error is better. Positive improvements indicate lower error than the baseline.}",
        r"\label{tab:accuracy-leaderboard}",
        r"\begin{tabular}{rllllllrrrrrr}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_escape_latex(row.get(key, "")) for key in keys) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _standard_outputs(output_dir: Path) -> dict[str, str]:
    return {
        "csv": str(output_dir / "accuracy_leaderboard.csv"),
        "json": str(output_dir / "accuracy_leaderboard.json"),
        "markdown": str(output_dir / "accuracy_leaderboard.md"),
        "latex": str(output_dir / "accuracy_leaderboard.tex"),
        "paper_summary_csv": str(output_dir / "accuracy_leaderboard_paper_summary.csv"),
        "paper_summary_json": str(output_dir / "accuracy_leaderboard_paper_summary.json"),
        "paper_summary_markdown": str(output_dir / "accuracy_leaderboard_paper_summary.md"),
        "paper_summary_latex": str(output_dir / "accuracy_leaderboard_paper_summary.tex"),
        "sanity_report_json": str(output_dir / "accuracy_leaderboard_sanity_report.json"),
        "sanity_report_markdown": str(output_dir / "accuracy_leaderboard_sanity_report.md"),
    }


def _method_baseline_rows(
    *,
    dataset: str,
    source: str,
    raw_error: float,
    persistence_error: float,
    visible_raw_error: float = float("nan"),
    occluded_persistence_error: float = float("nan"),
    row_count: int | str = "",
) -> list[dict[str, Any]]:
    rows = []
    if math.isfinite(raw_error):
        rows.append(
            {
                "dataset": dataset,
                "source": source,
                "motion_bin": "",
                "noise_deg": "",
                "occlusion_prob": "",
                "method": "raw_measurement",
                "transition_model": "none",
                "filter_backend": "none",
                "num_particles": "",
                "tracking_error_deg": raw_error,
                "visible_error_deg": visible_raw_error,
                "occluded_error_deg": float("nan"),
                "reappeared_error_deg": float("nan"),
                "raw_measurement_error_deg": raw_error,
                "persistence_error_deg": persistence_error,
                "improvement_vs_raw_deg": 0.0,
                "improvement_vs_persistence_deg": persistence_error - raw_error if math.isfinite(persistence_error) else float("nan"),
                "mean_ess": float("nan"),
                "collapse_fraction": float("nan"),
                "row_count": row_count,
            }
        )
    if math.isfinite(persistence_error):
        rows.append(
            {
                "dataset": dataset,
                "source": source,
                "motion_bin": "",
                "noise_deg": "",
                "occlusion_prob": "",
                "method": "persistence",
                "transition_model": "deterministic_persistence",
                "filter_backend": "none",
                "num_particles": "",
                "tracking_error_deg": persistence_error,
                "visible_error_deg": float("nan"),
                "occluded_error_deg": occluded_persistence_error,
                "reappeared_error_deg": float("nan"),
                "raw_measurement_error_deg": raw_error,
                "persistence_error_deg": persistence_error,
                "improvement_vs_raw_deg": raw_error - persistence_error if math.isfinite(raw_error) else float("nan"),
                "improvement_vs_persistence_deg": 0.0,
                "mean_ess": float("nan"),
                "collapse_fraction": float("nan"),
                "row_count": row_count,
            }
        )
    return rows


def _summarize_detector_rows(rows: list[dict[str, str]]) -> dict[str, float]:
    return {key: _nanmean(row.get(key, "") for row in rows) for key in DETECTOR_SUMMARY_KEYS}


def _detector_dataset_from_summary(summary: dict[str, Any], default: str) -> str:
    data_root = str(summary.get("measurement_data_root", "") or "")
    if data_root:
        name = Path(data_root).name
        if name:
            return name
    return default


def _detector_row(
    *,
    method: str,
    dataset: str,
    means: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    tracking = _float_or_nan(means.get("filter_error_deg"))
    raw = _float_or_nan(means.get("observed_error_deg"))
    raw_joint = _float_or_nan(means.get("observed_joint_error_deg"))
    persistence = _float_or_nan(means.get("persistence_error_deg"))
    row_count = summary.get("row_count", "")
    if not row_count and "sequence" in means:
        row_count = ""
    return {
        "dataset": dataset,
        "source": "detector_hmr",
        "motion_bin": "",
        "noise_deg": "",
        "occlusion_prob": "",
        "method": method,
        "transition_model": summary.get("transition_model", ""),
        "filter_backend": summary.get("filter_backend", ""),
        "num_particles": _int_or_blank(summary.get("num_particles", "")),
        "tracking_error_deg": tracking,
        "visible_error_deg": _float_or_nan(means.get("filter_observed_joint_error_deg")),
        "occluded_error_deg": _float_or_nan(means.get("filter_occluded_joint_error_deg")),
        "reappeared_error_deg": float("nan"),
        "raw_measurement_error_deg": raw,
        "persistence_error_deg": persistence,
        "improvement_vs_raw_deg": raw - tracking if math.isfinite(raw) else raw_joint - tracking,
        "improvement_vs_persistence_deg": persistence - tracking,
        "mean_ess": _float_or_nan(means.get("mean_ess")),
        "collapse_fraction": float("nan"),
        "row_count": row_count,
    }


def load_detector_run(spec: DetectorRunSpec, default_dataset: str) -> list[dict[str, Any]]:
    summary = _read_json(spec.path / "detector_measurement_eval_summary.json")
    metrics_rows = _read_csv(spec.path / "detector_filter_metrics.csv")
    if not summary and not metrics_rows:
        raise FileNotFoundError(f"{spec.path} does not contain detector_measurement_eval_summary.json or detector_filter_metrics.csv")
    means = dict(summary.get("means", {}))
    if metrics_rows:
        row_means = _summarize_detector_rows(metrics_rows)
        means = {**row_means, **{key: value for key, value in means.items() if _is_finite(value)}}
        summary = {**summary, "row_count": summary.get("row_count", len(metrics_rows))}
        if "transition_model" not in summary and metrics_rows:
            summary["transition_model"] = metrics_rows[0].get("transition_model", "")
        if "filter_backend" not in summary and metrics_rows:
            summary["filter_backend"] = metrics_rows[0].get("filter_backend", "")
        if "num_particles" not in summary and metrics_rows:
            summary["num_particles"] = metrics_rows[0].get("num_particles", "")
    dataset = _detector_dataset_from_summary(summary, default_dataset)
    main_row = _detector_row(
        method=spec.method,
        dataset=dataset,
        means=means,
        summary=summary,
    )
    return [
        *_method_baseline_rows(
            dataset=dataset,
            source="detector_hmr",
            raw_error=_float_or_nan(means.get("observed_error_deg")),
            persistence_error=_float_or_nan(means.get("persistence_error_deg")),
            visible_raw_error=_float_or_nan(means.get("observed_joint_error_deg")),
            occluded_persistence_error=_float_or_nan(means.get("persistence_occluded_joint_error_deg")),
            row_count=summary.get("row_count", ""),
        ),
        main_row,
    ]


def _coerce_motion_row(row: dict[str, str], dataset: str) -> dict[str, Any]:
    method = row.get("method", "")
    tracking = _float_or_nan(row.get("mean_tracking_error_deg"))
    raw_improvement = _float_or_nan(row.get("mean_improvement_vs_raw_deg"))
    persistence_improvement = _float_or_nan(row.get("mean_improvement_vs_persistence_deg"))
    raw = tracking + raw_improvement if math.isfinite(raw_improvement) else float("nan")
    persistence = tracking + persistence_improvement if math.isfinite(persistence_improvement) else float("nan")
    source = "motion_stratified"
    motion_bin = row.get("motion_bin", "")
    if motion_bin:
        source = f"motion_stratified:{motion_bin}"
    return {
        "dataset": dataset,
        "source": source,
        "motion_bin": motion_bin,
        "noise_deg": row.get("noise_deg", ""),
        "occlusion_prob": row.get("occlusion_prob", ""),
        "method": method,
        "transition_model": row.get("transition_model", method),
        "filter_backend": row.get("filter_backend", ""),
        "num_particles": _int_or_blank(row.get("num_particles", "")),
        "tracking_error_deg": tracking,
        "visible_error_deg": float("nan"),
        "occluded_error_deg": float("nan"),
        "reappeared_error_deg": float("nan"),
        "raw_measurement_error_deg": raw,
        "persistence_error_deg": persistence,
        "improvement_vs_raw_deg": raw_improvement,
        "improvement_vs_persistence_deg": persistence_improvement,
        "mean_ess": float("nan"),
        "collapse_fraction": float("nan"),
        "row_count": _int_or_blank(row.get("row_count", "")),
    }


def load_motion_run(spec: MotionRunSpec) -> list[dict[str, Any]]:
    preferred = spec.path / "aggregate_method_means_by_noise_occlusion_motion.csv"
    fallback = spec.path / "aggregate_method_means_by_motion_bin.csv"
    rows = _read_csv(preferred) or _read_csv(fallback)
    if not rows:
        raise FileNotFoundError(f"{spec.path} does not contain aggregate_method_means_by_noise_occlusion_motion.csv or aggregate_method_means_by_motion_bin.csv")
    return [_coerce_motion_row(row, spec.dataset) for row in rows]


def parse_detector_run(value: str) -> DetectorRunSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("detector run must be METHOD=PATH")
    method, path = value.split("=", 1)
    if not method.strip():
        raise argparse.ArgumentTypeError("detector run method must not be empty")
    return DetectorRunSpec(method=method.strip(), path=Path(path).expanduser())


def parse_motion_run(value: str) -> MotionRunSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("motion run must be DATASET=PATH")
    dataset, path = value.split("=", 1)
    if not dataset.strip():
        raise argparse.ArgumentTypeError("motion run dataset must not be empty")
    return MotionRunSpec(dataset=dataset.strip(), path=Path(path).expanduser())


def parse_method(value: str) -> MethodSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("method must be LABEL=TRANSITION[:BACKEND[:NUM_PARTICLES]]")
    label, payload = value.split("=", 1)
    parts = payload.split(":")
    if not label.strip() or not parts[0].strip():
        raise argparse.ArgumentTypeError("method label and transition model are required")
    backend = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "numpy"
    num_particles = int(parts[2]) if len(parts) > 2 and parts[2].strip() else None
    return MethodSpec(
        label=label.strip(),
        transition_model=parts[0].strip(),
        filter_backend=backend,
        num_particles=num_particles,
    )


def _load_detector_evaluation_tools() -> tuple[Any, Any]:
    from pose_filter.detector_evaluation import (  # noqa: PLC0415
        load_detector_eval_config,
        run_detector_measurement_eval,
    )

    return load_detector_eval_config, run_detector_measurement_eval


def _run_detector_methods(
    *,
    eval_config_path: Path,
    method_specs: list[MethodSpec],
    output_dir: Path,
) -> list[DetectorRunSpec]:
    if not method_specs:
        raise ValueError("--eval-config requires at least one --method")
    load_detector_eval_config, run_detector_measurement_eval = _load_detector_evaluation_tools()
    base_config = load_detector_eval_config(eval_config_path)
    run_specs = []
    runs_root = output_dir / "detector_method_runs"
    for spec in method_specs:
        config = copy.deepcopy(base_config)
        config["transition_model"] = spec.transition_model
        config["filter_backend"] = spec.filter_backend
        if spec.num_particles is not None:
            config["num_particles"] = int(spec.num_particles)
        run_dir = runs_root / spec.label
        config["output_dir"] = str(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "leaderboard_eval_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        run_detector_measurement_eval(config)
        run_specs.append(DetectorRunSpec(method=spec.label, path=run_dir))
    return run_specs


def _deduplicate_baselines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen_baselines = set()
    for row in rows:
        key = (
            row.get("dataset"),
            row.get("source"),
            row.get("motion_bin"),
            row.get("noise_deg"),
            row.get("occlusion_prob"),
            row.get("method"),
        )
        if row.get("method") in {"raw_measurement", "persistence"}:
            if key in seen_baselines:
                continue
            seen_baselines.add(key)
        out.append(row)
    return out


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _deduplicate_baselines(rows)
    rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("dataset", "")),
            str(row.get("source", "")),
            str(row.get("motion_bin", "")),
            _float_or_nan(row.get("noise_deg")) if math.isfinite(_float_or_nan(row.get("noise_deg"))) else -1.0,
            _float_or_nan(row.get("occlusion_prob")) if math.isfinite(_float_or_nan(row.get("occlusion_prob"))) else -1.0,
            _float_or_nan(row.get("tracking_error_deg")) if math.isfinite(_float_or_nan(row.get("tracking_error_deg"))) else float("inf"),
            str(row.get("method", "")),
        ),
    )
    ranked = []
    previous_group: tuple[str, str, str, str, str] | None = None
    rank = 0
    for row in rows:
        group = (
            str(row.get("dataset", "")),
            str(row.get("source", "")),
            str(row.get("motion_bin", "")),
            str(row.get("noise_deg", "")),
            str(row.get("occlusion_prob", "")),
        )
        if group != previous_group:
            rank = 1
            previous_group = group
        else:
            rank += 1
        ranked.append({"rank": rank, **row})
    return ranked


def _condition_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("source", "")),
        str(row.get("motion_bin", "")),
        str(row.get("noise_deg", "")),
        str(row.get("occlusion_prob", "")),
    )


def _paper_group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("source", "")),
        str(row.get("motion_bin", "")),
        str(row.get("method", "")),
    )


def _sum_row_counts(rows: list[dict[str, Any]]) -> int | str:
    values = [_int_or_blank(row.get("row_count", "")) for row in rows]
    numeric = [value for value in values if isinstance(value, int)]
    if not numeric:
        return ""
    return int(sum(numeric))


def build_paper_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    condition_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    paper_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        condition_groups[_condition_key(row)].append(row)
        paper_groups[_paper_group_key(row)].append(row)

    win_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for condition_rows in condition_groups.values():
        finite_rows = [row for row in condition_rows if math.isfinite(_float_or_nan(row.get("tracking_error_deg")))]
        if not finite_rows:
            continue
        best = min(_float_or_nan(row.get("tracking_error_deg")) for row in finite_rows)
        for row in finite_rows:
            if math.isclose(_float_or_nan(row.get("tracking_error_deg")), best, rel_tol=1e-12, abs_tol=1e-12):
                win_counts[_paper_group_key(row)] += 1

    summary_rows = []
    for key, group_rows in paper_groups.items():
        dataset, source, motion_bin, method = key
        condition_count = len({_condition_key(row) for row in group_rows})
        summary_rows.append(
            {
                "dataset": dataset,
                "source": source,
                "motion_bin": motion_bin,
                "method": method,
                "condition_count": condition_count,
                "win_count": win_counts.get(key, 0),
                "mean_tracking_error_deg": _nanmean(row.get("tracking_error_deg") for row in group_rows),
                "mean_raw_measurement_error_deg": _nanmean(row.get("raw_measurement_error_deg") for row in group_rows),
                "mean_persistence_error_deg": _nanmean(row.get("persistence_error_deg") for row in group_rows),
                "mean_improvement_vs_raw_deg": _nanmean(row.get("improvement_vs_raw_deg") for row in group_rows),
                "mean_improvement_vs_persistence_deg": _nanmean(row.get("improvement_vs_persistence_deg") for row in group_rows),
                "worse_than_raw_count": sum(1 for row in group_rows if _float_or_nan(row.get("improvement_vs_raw_deg")) < 0.0),
                "worse_than_persistence_count": sum(1 for row in group_rows if _float_or_nan(row.get("improvement_vs_persistence_deg")) < 0.0),
                "row_count": _sum_row_counts(group_rows),
            }
        )
    return sorted(
        summary_rows,
        key=lambda row: (
            str(row.get("dataset", "")),
            str(row.get("source", "")),
            str(row.get("motion_bin", "")),
            _float_or_nan(row.get("mean_tracking_error_deg")) if math.isfinite(_float_or_nan(row.get("mean_tracking_error_deg"))) else float("inf"),
            str(row.get("method", "")),
        ),
    )


def build_sanity_report(
    rows: list[dict[str, Any]],
    paper_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    condition_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    context_method_counts: dict[tuple[str, str, str, str, str, str], int] = defaultdict(int)
    for row in rows:
        condition_key = _condition_key(row)
        condition_groups[condition_key].append(row)
        context_method_counts[(*condition_key, str(row.get("method", "")))] += 1

    duplicate_context_rows = [
        {
            "dataset": key[0],
            "source": key[1],
            "motion_bin": key[2],
            "noise_deg": key[3],
            "occlusion_prob": key[4],
            "method": key[5],
            "count": count,
        }
        for key, count in sorted(context_method_counts.items())
        if count > 1
    ]

    missing_baseline_conditions = []
    for condition_key, condition_rows in sorted(condition_groups.items()):
        methods = {str(row.get("method", "")) for row in condition_rows}
        missing = []
        if not methods.intersection(RAW_METHODS):
            missing.append("raw")
        if not methods.intersection(PERSISTENCE_METHODS):
            missing.append("persistence")
        if missing:
            missing_baseline_conditions.append(
                {
                    "dataset": condition_key[0],
                    "source": condition_key[1],
                    "motion_bin": condition_key[2],
                    "noise_deg": condition_key[3],
                    "occlusion_prob": condition_key[4],
                    "missing": missing,
                }
            )

    worse_than_raw = [row for row in paper_summary if int(row.get("worse_than_raw_count", 0)) > 0 and str(row.get("method", "")) not in RAW_METHODS]
    worse_than_persistence = [row for row in paper_summary if int(row.get("worse_than_persistence_count", 0)) > 0 and str(row.get("method", "")) not in PERSISTENCE_METHODS]

    best_methods = []
    summary_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paper_summary:
        summary_groups[
            (
                str(row.get("dataset", "")),
                str(row.get("source", "")),
                str(row.get("motion_bin", "")),
            )
        ].append(row)
    for summary_key, group_rows in sorted(summary_groups.items()):
        finite_rows = [row for row in group_rows if math.isfinite(_float_or_nan(row.get("mean_tracking_error_deg")))]
        if not finite_rows:
            continue
        best = min(finite_rows, key=lambda row: _float_or_nan(row.get("mean_tracking_error_deg")))
        best_methods.append(
            {
                "dataset": summary_key[0],
                "source": summary_key[1],
                "motion_bin": summary_key[2],
                "method": best.get("method", ""),
                "mean_tracking_error_deg": best.get("mean_tracking_error_deg", ""),
                "mean_improvement_vs_raw_deg": best.get("mean_improvement_vs_raw_deg", ""),
                "mean_improvement_vs_persistence_deg": best.get(
                    "mean_improvement_vs_persistence_deg",
                    "",
                ),
                "condition_count": best.get("condition_count", ""),
                "win_count": best.get("win_count", ""),
            }
        )

    return {
        "row_count": len(rows),
        "condition_count": len(condition_groups),
        "paper_summary_row_count": len(paper_summary),
        "duplicate_context_rows": duplicate_context_rows,
        "missing_baseline_conditions": missing_baseline_conditions,
        "methods_worse_than_raw": [
            {
                "dataset": row.get("dataset", ""),
                "source": row.get("source", ""),
                "motion_bin": row.get("motion_bin", ""),
                "method": row.get("method", ""),
                "worse_than_raw_count": row.get("worse_than_raw_count", 0),
                "condition_count": row.get("condition_count", 0),
            }
            for row in worse_than_raw
        ],
        "methods_worse_than_persistence": [
            {
                "dataset": row.get("dataset", ""),
                "source": row.get("source", ""),
                "motion_bin": row.get("motion_bin", ""),
                "method": row.get("method", ""),
                "worse_than_persistence_count": row.get("worse_than_persistence_count", 0),
                "condition_count": row.get("condition_count", 0),
            }
            for row in worse_than_persistence
        ],
        "best_methods_by_dataset_motion_bin": best_methods,
    }


def _write_paper_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "dataset",
        "motion bin",
        "method",
        "conditions",
        "wins",
        "tracking (deg)",
        "delta raw",
        "delta pers.",
        "worse raw",
        "worse pers.",
    ]
    keys = [
        "dataset",
        "motion_bin",
        "method",
        "condition_count",
        "win_count",
        "mean_tracking_error_deg",
        "mean_improvement_vs_raw_deg",
        "mean_improvement_vs_persistence_deg",
        "worse_than_raw_count",
        "worse_than_persistence_count",
    ]
    lines = [
        "# Accuracy leaderboard paper summary",
        "",
        "Rows are aggregated by dataset, motion bin, and method across noise and occlusion conditions.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_markdown(row.get(key, "")) for key in keys) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_paper_summary_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "dataset",
        "motion_bin",
        "method",
        "condition_count",
        "win_count",
        "mean_tracking_error_deg",
        "mean_improvement_vs_raw_deg",
        "mean_improvement_vs_persistence_deg",
    ]
    headers = [
        "Dataset",
        "Motion bin",
        "Method",
        "Cond.",
        "Wins",
        "Track.",
        r"$\Delta$ raw",
        r"$\Delta$ pers.",
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paper summary of the accuracy leaderboard, aggregated across noise and occlusion conditions.}",
        r"\label{tab:accuracy-leaderboard-paper-summary}",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_escape_latex(row.get(key, "")) for key in keys) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_sanity_report_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Accuracy leaderboard sanity report",
        "",
        f"- rows: {report['row_count']}",
        f"- conditions: {report['condition_count']}",
        f"- paper summary rows: {report['paper_summary_row_count']}",
        f"- duplicate context rows: {len(report['duplicate_context_rows'])}",
        f"- missing baseline conditions: {len(report['missing_baseline_conditions'])}",
        "",
        "## Best Methods",
        "",
        "| dataset | motion bin | method | tracking (deg) | delta raw | delta pers. | wins | conditions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["best_methods_by_dataset_motion_bin"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(row.get("dataset", "")),
                    _escape_markdown(row.get("motion_bin", "")),
                    _escape_markdown(row.get("method", "")),
                    _escape_markdown(row.get("mean_tracking_error_deg", "")),
                    _escape_markdown(row.get("mean_improvement_vs_raw_deg", "")),
                    _escape_markdown(row.get("mean_improvement_vs_persistence_deg", "")),
                    _escape_markdown(row.get("win_count", "")),
                    _escape_markdown(row.get("condition_count", "")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Warnings", ""])
    if report["duplicate_context_rows"]:
        lines.append("- Duplicate rows exist for at least one dataset/noise/occlusion/method context.")
    if report["missing_baseline_conditions"]:
        lines.append("- At least one condition is missing a raw or persistence baseline.")
    if report["methods_worse_than_raw"]:
        lines.append("- At least one non-raw method is worse than raw in one or more conditions.")
    if report["methods_worse_than_persistence"]:
        lines.append("- At least one non-persistence method is worse than persistence in one or more conditions.")
    if not any(
        [
            report["duplicate_context_rows"],
            report["missing_baseline_conditions"],
            report["methods_worse_than_raw"],
            report["methods_worse_than_persistence"],
        ]
    ):
        lines.append("- No sanity warnings.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_leaderboard(
    *,
    detector_runs: list[DetectorRunSpec],
    motion_runs: list[MotionRunSpec],
    detector_dataset: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for detector_spec in detector_runs:
        rows.extend(load_detector_run(detector_spec, default_dataset=detector_dataset))
    for motion_spec in motion_runs:
        rows.extend(load_motion_run(motion_spec))
    if not rows:
        raise ValueError("no rows were loaded; provide --detector-run, --motion-run, or --eval-config/--method")
    return _rank_rows(rows)


def write_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = _standard_outputs(output_dir)
    paper_summary = build_paper_summary(rows)
    sanity_report = build_sanity_report(rows, paper_summary)
    _write_csv(Path(outputs["csv"]), rows)
    Path(outputs["json"]).write_text(
        json.dumps(
            {
                "rows": rows,
                "row_count": len(rows),
                "paper_summary": paper_summary,
                "sanity_report": sanity_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_markdown(Path(outputs["markdown"]), rows)
    _write_latex(Path(outputs["latex"]), rows)
    _write_table_csv(Path(outputs["paper_summary_csv"]), paper_summary, PAPER_SUMMARY_COLUMNS)
    Path(outputs["paper_summary_json"]).write_text(
        json.dumps({"rows": paper_summary, "row_count": len(paper_summary)}, indent=2),
        encoding="utf-8",
    )
    _write_paper_summary_markdown(Path(outputs["paper_summary_markdown"]), paper_summary)
    _write_paper_summary_latex(Path(outputs["paper_summary_latex"]), paper_summary)
    Path(outputs["sanity_report_json"]).write_text(
        json.dumps(sanity_report, indent=2),
        encoding="utf-8",
    )
    _write_sanity_report_markdown(Path(outputs["sanity_report_markdown"]), sanity_report)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-facing SO(3)^K pose-filter accuracy leaderboards.")
    parser.add_argument(
        "--detector-run",
        action="append",
        default=[],
        type=parse_detector_run,
        help="Existing detector/HMR eval run as METHOD=PATH.",
    )
    parser.add_argument(
        "--motion-run",
        action="append",
        default=[],
        type=parse_motion_run,
        help="Existing motion-stratified artifact as DATASET=PATH.",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        help="Detector-measurement config. If set, each --method is evaluated before building the leaderboard.",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        type=parse_method,
        help="Detector method to evaluate as LABEL=TRANSITION[:BACKEND[:NUM_PARTICLES]].",
    )
    parser.add_argument(
        "--detector-dataset",
        default="detector_hmr",
        help="Dataset label for detector runs when the summary does not provide a better name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/accuracy_leaderboard"),
        help="Directory for CSV/JSON/Markdown/LaTeX outputs.",
    )
    args = parser.parse_args()

    detector_runs = list(args.detector_run)
    if args.eval_config is not None:
        detector_runs.extend(
            _run_detector_methods(
                eval_config_path=args.eval_config,
                method_specs=list(args.method),
                output_dir=args.output_dir,
            )
        )
    elif args.method:
        raise SystemExit("--method can only be used together with --eval-config")

    rows = build_leaderboard(
        detector_runs=detector_runs,
        motion_runs=list(args.motion_run),
        detector_dataset=str(args.detector_dataset),
    )
    outputs = write_outputs(args.output_dir, rows)
    print(json.dumps({"row_count": len(rows), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()

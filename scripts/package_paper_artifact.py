from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

SENSITIVE_KEYS = {
    "source_data_root",
    "data_root",
    "segments_dir",
    "output_dir",
    "run_dir",
    "output_path",
    "input_path",
    "path",
    "sample_path",
    "source_path",
    "target_path",
    "source",
    "python_executable",
}

PUBLIC_FILENAMES = {
    "run_manifest.json",
    "motion_stratified_validation.json",
    "motion_stratified_private_accad_eval_summary.json",
    "motion_stratified_private_accad_eval_summary.md",
    "aggregate_benchmark_metrics_by_motion_bin.csv",
    "aggregate_transition_metrics_by_motion_bin.csv",
    "aggregate_transition_means_by_motion_bin.csv",
    "aggregate_method_means_by_motion_bin.csv",
    "aggregate_method_means_by_noise_occlusion_motion.csv",
    "robustness_summary_by_motion_bin.csv",
    "transition_tracking_diagnostics_by_motion_bin.csv",
}

PUBLIC_BENCHMARK_FILENAMES = {
    "first_results_summary.json",
    "benchmark_metrics.csv",
    "transition_metrics.csv",
}


def _redacted(value: Any) -> str:
    return "<redacted>" if value not in (None, "") else value


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            key_str = str(key)
            if key_str in SENSITIVE_KEYS or key_str.endswith("_path") or key_str.endswith("_root") or key_str.endswith("_dir"):
                out[key_str] = _redacted(child)
            else:
                out[key_str] = _sanitize_json(child)
        return out
    if isinstance(value, list):
        return [_sanitize_json(child) for child in value]
    return value


def _copy_sanitized_json(source: Path, target: Path) -> None:
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        target.write_text(json.dumps(_sanitize_json(payload), indent=2, sort_keys=True), encoding="utf-8")
    except json.JSONDecodeError:
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _copy_sanitized_csv(source: Path, target: Path) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []
    kept = [name for name in fieldnames if name not in SENSITIVE_KEYS and not name.endswith("_path") and not name.endswith("_root") and not name.endswith("_dir")]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=kept)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in kept})


def _copy_text(source: Path, target: Path) -> None:
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"`(?:[A-Za-z]:[/\\]|/home/|<redacted>/)[^`]*`", "`<redacted>`", text)
    text = re.sub(r"(?:[A-Za-z]:[/\\]|/home/)\S+", "<redacted>", text)
    target.write_text(text, encoding="utf-8")


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".json":
        _copy_sanitized_json(source, target)
    elif source.suffix == ".csv":
        _copy_sanitized_csv(source, target)
    elif source.suffix in {".md", ".txt"}:
        _copy_text(source, target)
    else:
        shutil.copy2(source, target)


def package_artifact(result_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    for name in PUBLIC_FILENAMES:
        source = result_root / name
        if source.exists():
            target = output_dir / name
            _copy_file(source, target)
            copied.append(target.relative_to(output_dir).as_posix())

    for source in sorted((result_root / "benchmarks").glob("**/*")) if (result_root / "benchmarks").exists() else []:
        if not source.is_file():
            continue
        if source.name in PUBLIC_BENCHMARK_FILENAMES or source.suffix == ".svg":
            target = output_dir / source.relative_to(result_root)
            _copy_file(source, target)
            copied.append(target.relative_to(output_dir).as_posix())

    manifest = {
        "schema_version": 1,
        "source_result_root": "<redacted>",
        "file_count": len(copied),
        "files": copied,
        "redaction": {
            "path_like_fields_removed_or_redacted": True,
            "raw_motion_data_included": False,
            "motion_bin_segment_files_included": False,
        },
    }
    (output_dir / "paper_artifact_package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a public-safe paper artifact package from full-data benchmark outputs.")
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = package_artifact(args.result_root, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

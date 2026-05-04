from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess  # nosec B404
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KEY_PACKAGES = (
    "numpy",
    "pyrecest",
    "torch",
    "scipy",
    "matplotlib",
    "so3-pose-filter",
)

GITHUB_ENV_KEYS = (
    "GITHUB_ACTION",
    "GITHUB_ACTOR",
    "GITHUB_EVENT_NAME",
    "GITHUB_JOB",
    "GITHUB_REF",
    "GITHUB_REF_NAME",
    "GITHUB_REPOSITORY",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_NUMBER",
    "GITHUB_SHA",
    "GITHUB_WORKFLOW",
    "RUNNER_ARCH",
    "RUNNER_NAME",
    "RUNNER_OS",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(args: list[str]) -> str | None:
    try:
        # Query local git metadata with fixed commands for the run manifest.
        return subprocess.check_output(  # nosec B603 B607
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in KEY_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _file_record(path: Path, root: Path, max_hash_bytes: int) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if stat.st_size <= max_hash_bytes:
        record["sha256"] = _sha256_file(path)
    else:
        record["sha256"] = None
        record["sha256_skipped_reason"] = f"file exceeds max_hash_bytes={max_hash_bytes}"
    return record


def _result_files(root: Path, max_hash_bytes: int) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    files = [path for path in root.rglob("*") if path.is_file()]
    return [_file_record(path, root, max_hash_bytes) for path in sorted(files)]


def build_manifest(
    *,
    experiment_name: str,
    config_path: Path,
    result_root: Path,
    output_path: Path,
    source_data_root: str | None,
    max_hash_bytes: int,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    result_root = result_root.resolve()
    config_content = _json_load(config_path) if config_path.exists() else None
    github_env = {key: os.environ[key] for key in GITHUB_ENV_KEYS if key in os.environ}
    files = _result_files(result_root, max_hash_bytes=max_hash_bytes)
    return {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "source_data_root": source_data_root,
        "config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path) if config_path.exists() else None,
            "content": config_content,
        },
        "result_root": str(result_root),
        "manifest_path": str(output_path.resolve()),
        "git": {
            "sha": _git(["rev-parse", "HEAD"]),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "ref": _git(["symbolic-ref", "--short", "HEAD"]),
            "status_short": _git(["status", "--short"]),
        },
        "github": github_env,
        "runtime": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "package_versions": _package_versions(),
        },
        "outputs": {
            "file_count": len(files),
            "files": files,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a reproducibility manifest for an experiment output directory.")
    parser.add_argument("--experiment-name", required=True, help="Stable experiment identifier, e.g. accad-motion-stratified.")
    parser.add_argument("--config", required=True, type=Path, help="Config JSON used for the run.")
    parser.add_argument("--result-root", required=True, type=Path, help="Directory containing result artifacts.")
    parser.add_argument("--output", required=True, type=Path, help="Manifest JSON output path.")
    parser.add_argument("--source-data-root", default=None, help="Source dataset root or dataset identifier used by the run.")
    parser.add_argument("--max-hash-mb", default=100.0, type=float, help="Only compute SHA-256 for files up to this size.")
    args = parser.parse_args()

    manifest = build_manifest(
        experiment_name=args.experiment_name,
        config_path=args.config,
        result_root=args.result_root,
        output_path=args.output,
        source_data_root=args.source_data_root,
        max_hash_bytes=int(args.max_hash_mb * 1024 * 1024),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(args.output), "file_count": manifest["outputs"]["file_count"]}, indent=2))


if __name__ == "__main__":
    main()

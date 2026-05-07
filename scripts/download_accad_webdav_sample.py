from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import subprocess  # nosec B404
from pathlib import Path

from download_amass_sample import is_amass_npz


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def _rclone_webdav_flags(webdav_url: str, key: str, obscured_password: str) -> list[str]:
    return [
        "--webdav-url",
        webdav_url,
        "--webdav-vendor",
        "owncloud",
        "--webdav-user",
        key,
        "--webdav-pass",
        obscured_password,
    ]


def _obscure_password(password: str) -> str:
    return subprocess.check_output(["rclone", "obscure", password], text=True).strip()  # nosec B603 B607


def list_webdav_files(webdav_url: str, key: str, obscured_password: str) -> list[str]:
    output = subprocess.check_output(  # nosec B603 B607
        [
            "rclone",
            "lsf",
            ":webdav:",
            "--recursive",
            "--files-only",
            *_rclone_webdav_flags(webdav_url, key, obscured_password),
        ],
        text=True,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def select_pose_npz_files(paths: list[str], max_files: int, candidate_limit: int) -> list[str]:
    if max_files < 1:
        raise ValueError("max_files must be at least 1")
    if candidate_limit < max_files:
        raise ValueError("candidate_limit must be at least max_files")

    pose_paths = [
        path
        for path in sorted(paths, key=str.casefold)
        if path.lower().endswith(".npz") and posixpath.basename(path).lower().endswith("_poses.npz")
    ]
    fallback_paths = [
        path
        for path in sorted(paths, key=str.casefold)
        if path.lower().endswith(".npz") and path not in pose_paths
    ]
    return (pose_paths + fallback_paths)[:candidate_limit]


def _output_name(valid_index: int, max_files: int) -> str:
    if max_files == 1:
        return "sample.npz"
    return f"sample_{valid_index:03d}.npz"


def _copy_remote_file(remote_path: str, output_path: Path, webdav_url: str, key: str, obscured_password: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # nosec B603 B607
        [
            "rclone",
            "copyto",
            f":webdav:{remote_path}",
            str(output_path),
            *_rclone_webdav_flags(webdav_url, key, obscured_password),
        ],
        check=True,
    )


def _path_digest(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def download_accad_webdav_sample(output_dir: Path, max_files: int, candidate_limit: int) -> dict:
    webdav_url = _require_env("ACCAD_DATA_WEBDAV_URL")
    key = _require_env("ACCAD_DATA_KEY")
    password = _require_env("ACCAD_DATA_PASSWORD")
    obscured_password = _obscure_password(password)

    output_dir.mkdir(parents=True, exist_ok=True)
    remote_files = list_webdav_files(webdav_url, key, obscured_password)
    candidates = select_pose_npz_files(remote_files, max_files=max_files, candidate_limit=candidate_limit)
    if not candidates:
        raise RuntimeError("ACCAD WebDAV share did not contain any .npz candidates")

    downloaded: list[dict] = []
    errors: list[dict] = []
    for remote_path in candidates:
        if len(downloaded) >= max_files:
            break
        output_path = output_dir / _output_name(len(downloaded), max_files)
        try:
            _copy_remote_file(remote_path, output_path, webdav_url, key, obscured_password)
            ok, metadata = is_amass_npz(output_path)
            if not ok:
                errors.append(
                    {
                        "remote_basename": posixpath.basename(remote_path),
                        "remote_path_sha256": _path_digest(remote_path),
                        "metadata": metadata,
                    }
                )
                output_path.unlink(missing_ok=True)
                continue
            downloaded.append(
                {
                    "sample_path": str(output_path),
                    "remote_basename": posixpath.basename(remote_path),
                    "remote_path_sha256": _path_digest(remote_path),
                    "metadata": metadata,
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "remote_basename": posixpath.basename(remote_path),
                    "remote_path_sha256": _path_digest(remote_path),
                    "error": str(exc),
                }
            )

    if len(downloaded) < max_files:
        raise RuntimeError(f"downloaded {len(downloaded)} valid AMASS files, expected {max_files}; errors: {errors}")

    return {
        "source": "ACCAD_DATA_WEBDAV_URL",
        "selected_count": len(downloaded),
        "remote_file_count": len(remote_files),
        "candidate_count": len(candidates),
        "candidate_limit": candidate_limit,
        "files": downloaded,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a bounded ACCAD AMASS sample from an ownCloud/WebDAV share with rclone.")
    parser.add_argument("--output-dir", required=True, help="Directory where selected .npz files are written.")
    parser.add_argument("--report", required=True, help="Output JSON report path.")
    parser.add_argument("--max-files", type=int, default=1, help="Number of valid AMASS .npz files to download.")
    parser.add_argument("--candidate-limit", type=int, default=50, help="Maximum number of listed .npz candidates to try.")
    args = parser.parse_args()

    report = download_accad_webdav_sample(Path(args.output_dir), max_files=args.max_files, candidate_limit=args.candidate_limit)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

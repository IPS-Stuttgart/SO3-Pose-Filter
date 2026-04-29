from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


def _require_http_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"expected an HTTP(S) URL, got scheme {parsed.scheme!r}")
    return url


def is_amass_npz(path: Path) -> tuple[bool, dict]:
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            if "poses" not in keys:
                return False, {"keys": sorted(keys)}
            poses_shape = tuple(data["poses"].shape)
            ok = (
                len(poses_shape) == 2
                and poses_shape[0] >= 2
                and poses_shape[1] >= 72
            )
            framerate_key = (
                "mocap_framerate"
                if "mocap_framerate" in keys
                else "mocap_frame_rate"
            )
            return ok, {
                "keys": sorted(keys),
                "poses_shape": poses_shape,
                "mocap_framerate": float(
                    np.asarray(data[framerate_key]).reshape(-1)[0]
                )
                if framerate_key in keys
                else None,
            }
    except Exception as exc:
        return False, {"error": str(exc)}


def public_share_filename(url: str) -> str | None:
    try:
        url = _require_http_url(url)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "SO3-Pose-Filter-CI/1.0"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310
            page = response.read(2_000_000).decode("utf-8", "replace")
    except Exception:
        return None

    match = re.search(
        r'id="initial-state-files_sharing-filename"\s+value="([^"]+)"',
        page,
    )
    if not match:
        return None
    try:
        encoded = html.unescape(match.group(1))
        decoded = base64.b64decode(encoded).decode("utf-8")
    except Exception:
        return None
    return decoded or None


def candidate_urls(url: str) -> list[str]:
    out = [url]
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme:
        return out
    clean = url.rstrip("/")
    out.extend([f"{clean}/download", f"{clean}?download=1"])
    match = re.search(r"/s/([^/?#]+)", parsed.path)
    if match:
        token = match.group(1)
        base = f"{parsed.scheme}://{parsed.netloc}"
        filename = public_share_filename(url)
        if filename:
            quoted_filename = urllib.parse.quote(filename)
            out.extend(
                [
                    f"{clean}/download/{quoted_filename}",
                    f"{base}/s/{token}/download/{quoted_filename}",
                    f"{base}/index.php/s/{token}/download/{quoted_filename}",
                ]
            )
        out.extend(
            [
                f"{base}/s/{token}/download",
                f"{base}/index.php/s/{token}/download",
                f"{base}/public.php/dav/files/{token}",
            ]
        )
    return list(dict.fromkeys(out))


def download(url: str, destination: Path) -> None:
    url = _require_http_url(url)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "SO3-Pose-Filter-CI/1.0",
            "Accept": "application/octet-stream,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def download_amass_sample(sample_url: str, output: Path) -> dict:
    if not sample_url:
        raise ValueError("sample URL is empty")

    output.parent.mkdir(parents=True, exist_ok=True)
    target_dir = output.parent
    errors: list[str] = []
    selected: Path | None = None
    selected_meta: dict | None = None

    for idx, url in enumerate(candidate_urls(sample_url)):
        raw_path = target_dir / f"candidate_{idx}"
        try:
            download(url, raw_path)
            ok, meta = is_amass_npz(raw_path)
            if ok:
                selected = raw_path
                selected_meta = meta
            elif zipfile.is_zipfile(raw_path):
                extract_dir = target_dir / f"candidate_{idx}_unzipped"
                with zipfile.ZipFile(raw_path) as archive:
                    archive.extractall(extract_dir)
                for npz_path in extract_dir.rglob("*.npz"):
                    ok, meta = is_amass_npz(npz_path)
                    if ok:
                        selected, selected_meta = npz_path, meta
                        break
            if selected is not None:
                break
            errors.append(f"{url}: downloaded but not an AMASS .npz")
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    if selected is None or selected_meta is None:
        raise RuntimeError(
            "Could not download a valid AMASS .npz. Use a direct .npz/archive "
            "download URL if the public share page does not expose raw file bytes.\n"
            + "\n".join(errors)
        )

    if selected != output:
        shutil.copy2(selected, output)
    return {
        "sample_path": str(output),
        "metadata": selected_meta,
        "source": "AMASS_ACCAD_SAMPLE",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate an AMASS .npz")
    parser.add_argument(
        "--url",
        default=os.environ.get("SAMPLE_URL", ""),
        help="AMASS sample URL. Defaults to SAMPLE_URL.",
    )
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--report", required=True, help="Output JSON report path")
    args = parser.parse_args()

    report = download_amass_sample(args.url.strip(), Path(args.output))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

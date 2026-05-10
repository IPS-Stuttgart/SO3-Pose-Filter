from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_validator(repo_root: Path):
    sys.path.insert(0, str(repo_root / "scripts"))
    from download_amass_sample import is_amass_npz

    return is_amass_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a cached AMASS WebDAV sample.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--min-files", required=True, type=int)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache-source", required=True)
    args = parser.parse_args()

    data_root = args.data_root
    files = sorted(data_root.glob("*.npz"))
    if len(files) < args.min_files:
        raise SystemExit(f"Expected at least {args.min_files} .npz files in {data_root}, found {len(files)}")

    is_amass_npz = _load_validator(args.repo_root)
    records = []
    for path in files:
        ok, metadata = is_amass_npz(path)
        if not ok:
            raise SystemExit(f"Cached sample is not a valid AMASS .npz: {path}; metadata={metadata}")
        records.append({"sample_path": str(path), "metadata": metadata})

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "source": args.source_name,
                "cache_hit": args.cache_source != "download",
                "cache_source": args.cache_source,
                "selected_count": len(records),
                "files": records,
                "errors": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        "\n".join(
            [
                f"source={args.source_name}",
                f"cache_source={args.cache_source}",
                f"data_root={data_root}",
                "",
                *[f"{path.stat().st_size} {path}" for path in files],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Validated {len(records)} AMASS sample file(s) in {data_root}")


if __name__ == "__main__":
    main()

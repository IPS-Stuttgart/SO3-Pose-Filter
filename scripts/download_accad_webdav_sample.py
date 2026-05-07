from __future__ import annotations

import json
from pathlib import Path

from download_webdav_amass_sample import build_parser, download_webdav_amass_sample


def main() -> None:
    parser = build_parser()
    parser.description = "Download a bounded ACCAD AMASS sample from an ownCloud/WebDAV share with rclone."
    args = parser.parse_args()
    source_name = args.source_name or "ACCAD_DATA_WEBDAV_URL"

    report = download_webdav_amass_sample(
        Path(args.output_dir),
        max_files=args.max_files,
        candidate_limit=args.candidate_limit,
        webdav_url_env=args.webdav_url_env,
        key_env=args.key_env,
        password_env=args.password_env,
        source_name=source_name,
        preferred_suffix=args.preferred_suffix,
        webdav_vendor=args.webdav_vendor,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

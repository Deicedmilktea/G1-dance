"""Inspect the model manifest locally; never download or deserialize weights."""

import argparse
import gzip
import hashlib
import json
import tarfile
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def inspect_asset(asset, verify):
    path = ROOT / asset["path"]
    result = {"id": asset["id"], "path": asset["path"], "status": "missing"}
    if not path.is_file():
        return result
    result["bytes"] = path.stat().st_size
    expected_size = asset["expected_bytes"]
    if expected_size is not None and result["bytes"] != expected_size:
        result["status"] = "size_mismatch"
        result["expected_bytes"] = expected_size
        return result
    result["status"] = "present_unverified"
    if not verify:
        return result
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
        expected_sha256 = asset["expected_sha256"]
        if expected_sha256 and result["sha256"] != expected_sha256:
            result["status"] = "sha256_mismatch"
            return result
        if path.name.endswith(".tar.gz") or asset.get("archive_format") == "tar.gz":
            # Reading through every member checks truncation and gzip integrity.
            with gzip.open(path, "rb") as compressed:
                with tarfile.open(fileobj=compressed, mode="r|") as archive:
                    count = 0
                    for member in archive:
                        if member.isfile():
                            stream = archive.extractfile(member)
                            for _ in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                                pass
                        count += 1
                # Consume the gzip trailer even when tar reached its end markers.
                for _ in iter(lambda: compressed.read(8 * 1024 * 1024), b""):
                    pass
            result["archive_members"] = count
        else:
            with zipfile.ZipFile(path) as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise ValueError(f"Archive CRC failed: {bad_member}")
                result["archive_members"] = len(archive.infolist())
        result["status"] = "file_integrity_checked"
        result["expected_sha256_matched"] = bool(expected_sha256)
    except (OSError, EOFError, ValueError, zlib.error, tarfile.TarError, zipfile.BadZipFile) as error:
        result["status"] = "invalid_archive"
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Hash files and read archive members.")
    parser.add_argument("--report", type=Path, help="Save a JSON report to this project-relative path.")
    args = parser.parse_args()
    if args.report is not None:
        report_path = (ROOT / args.report).resolve()
        if not report_path.is_relative_to(ROOT):
            parser.error("The report must stay inside the project directory.")
    manifest = json.loads((ROOT / "configs/models.json").read_text())
    results = [inspect_asset(asset, args.verify) for asset in manifest["assets"]]
    for result in results:
        print(f"{result['id']}: {result['status']} ({result['path']})")
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Local file inventory and optional archive integrity; no model loading or dance inference.",
        "all_files_present": all(item["status"] in ("present_unverified", "file_integrity_checked") for item in results),
        "assets": results,
    }
    if args.report is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report["all_files_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

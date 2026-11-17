"""Download explicitly selected public assets from configs/models.json."""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from gdown.download import get_url_from_gdrive_confirmation

from check_models import ROOT, inspect_asset


def open_download(session, asset, offset):
    source = asset["source"]
    if source["kind"] == "google_drive":
        url = f"https://drive.google.com/uc?export=download&id={source['file_id']}"
    else:
        url = source["url"]
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    for _ in range(4):
        response = session.get(url, headers=headers, stream=True, timeout=(20, 90))
        response.raise_for_status()
        if response.headers.get("Content-Type", "").lower().startswith("text/html"):
            if source["kind"] != "google_drive":
                response.close()
                raise ValueError("The download endpoint returned HTML instead of a model.")
            try:
                # Confirmation pages are small. Never save one as a model file.
                page = response.raw.read(256 * 1024, decode_content=True).decode("utf-8")
            finally:
                response.close()
            url = get_url_from_gdrive_confirmation(page)
            continue
        if response.status_code == 206:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if not match or int(match[1]) != offset or int(match[3]) != asset["expected_bytes"]:
                response.close()
                raise ValueError("The server returned an unexpected byte range.")
            return response, offset
        if response.status_code != 200:
            response.close()
            raise ValueError("Unexpected download response status.")
        # A server that ignores Range must replace, rather than append to, a partial.
        return response, 0
    raise ValueError("Too many Google Drive confirmation redirects.")


def download_asset(asset, report_dir):
    target = ROOT / asset["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    status_path = report_dir / (asset["id"] + ".json")
    status = {"id": asset["id"], "path": asset["path"], "source_url": asset["source"]["url"]}

    def save_status():
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)

    if target.exists():
        info = inspect_asset(asset, verify=True)
        if info["status"] != "file_integrity_checked":
            raise ValueError(f"Existing file needs review: {asset['path']} ({info['status']})")
        status.update(info, download="reused_existing")
        save_status()
        print(f"{asset['id']}: existing file checked", flush=True)
        return True

    for attempt in range(1, 4):
        status.update(download="in_progress", attempt=attempt)
        save_status()
        session = requests.Session()
        session.headers.update({"User-Agent": "G1-dance-model-download/1.0", "Accept-Encoding": "identity"})
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > asset["expected_bytes"]:
                raise ValueError("Partial file is larger than the expected model; retain it for inspection.")
            if offset < asset["expected_bytes"]:
                response, offset = open_download(session, asset, offset)
                print(f"{asset['id']}: attempt {attempt}, starting at {offset:,} bytes", flush=True)
                last_update = time.monotonic()
                with response, partial.open("ab" if offset else "wb") as stream:
                    downloaded = offset
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        stream.write(chunk)
                        downloaded += len(chunk)
                        if downloaded > asset["expected_bytes"]:
                            raise ValueError("Received more bytes than expected.")
                        if time.monotonic() - last_update >= 15:
                            status["downloaded_bytes"] = downloaded
                            save_status()
                            print(f"{asset['id']}: {downloaded:,}/{asset['expected_bytes']:,} bytes", flush=True)
                            last_update = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
            if not partial.is_file() or partial.stat().st_size != asset["expected_bytes"]:
                raise ValueError("The download ended before the expected file size was reached.")
            # Reuse the read-only checker without exposing a final filename early.
            candidate = {**asset, "path": str(partial.relative_to(ROOT))}
            # The checker uses the archive suffix for the NJU tarball.
            if target.name.endswith(".tar.gz"):
                candidate["path"] = str(partial.relative_to(ROOT))
                candidate["archive_format"] = "tar.gz"
            info = inspect_asset(candidate, verify=True)
            if info["status"] != "file_integrity_checked":
                raise ValueError(f"Downloaded file integrity check failed: {info['status']}")
            partial.replace(target)
            info["path"] = asset["path"]
            status.pop("error", None)
            status.update(info, download="complete")
            save_status()
            print(f"{asset['id']}: complete, SHA256 {info['sha256']}", flush=True)
            return True
        except Exception as error:
            status.update(download="failed", error=f"{type(error).__name__}: {str(error)[:400]}")
            status["downloaded_bytes"] = partial.stat().st_size if partial.exists() else 0
            save_status()
            print(f"{asset['id']}: attempt {attempt} failed ({type(error).__name__})", flush=True)
            retryable = isinstance(error, (requests.Timeout, requests.ConnectionError)) or (
                isinstance(error, requests.HTTPError)
                and error.response is not None
                and error.response.status_code >= 500
            )
            if not retryable:
                break
            if attempt < 3:
                time.sleep(3)
        finally:
            session.close()
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", required=True, help="Explicit public asset IDs to download.")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/models.json").read_text())
    assets = {asset["id"]: asset for asset in manifest["assets"]}
    for name in args.assets:
        if name not in assets or assets[name]["source"]["kind"] == "registered_download":
            parser.error(f"Not a downloadable public asset: {name}")
    report_dir = ROOT / "outputs/model_preparation/downloads"
    report_dir.mkdir(parents=True, exist_ok=True)
    success = True
    for name in dict.fromkeys(args.assets):
        success = download_asset(assets[name], report_dir) and success
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

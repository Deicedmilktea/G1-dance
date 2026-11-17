"""Fetch pinned upstream repositories into third_party/ and apply local patches."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/upstream_sources.json"


def run(cmd, cwd=ROOT):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def fetch_source(entry, dest, force=False):
    if dest.exists():
        if not force:
            print(f"Skip existing {dest.relative_to(ROOT)} (pass --force to re-fetch)")
            return False
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if "commit" in entry:
        run(["git", "clone", entry["repository"], str(dest)])
        run(["git", "checkout", entry["commit"]], cwd=dest)
        return True
    url = entry["source_archive"]
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "source.tar.gz"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive, "r:gz") as tf:
            members = tf.getmembers()
            top = members[0].name.split("/", 1)[0]
            tf.extractall(tmp)
        shutil.move(str(Path(tmp) / top), str(dest))
    return True


def apply_patch(repo_dir, patch_rel):
    patch = ROOT / patch_rel
    if not patch.is_file():
        raise FileNotFoundError(f"Patch not found: {patch}")
    if (repo_dir / ".git").exists():
        check = subprocess.run(["git", "apply", "--check", str(patch)], cwd=repo_dir, stderr=subprocess.PIPE, text=True)
        if check.returncode == 0:
            run(["git", "apply", str(patch)], cwd=repo_dir)
            return
        reverse = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=repo_dir, stderr=subprocess.DEVNULL)
        if reverse.returncode == 0:
            print(f"Patch already applied: {patch_rel}")
            return
        raise RuntimeError(f"Failed to apply {patch_rel} to {repo_dir}: {check.stderr}")
    run(["patch", "-p1", "-i", str(patch)], cwd=repo_dir)


def ensure_symlink(link_path, target_path):
    if not target_path.exists():
        return
    if link_path.is_symlink() or not link_path.exists():
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(target_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-clone/re-download existing third_party directories")
    parser.add_argument("--skip-patches", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest["sources"]:
        dest = ROOT / entry["local_path"]
        fetched = fetch_source(entry, dest, force=args.force)
        if args.skip_patches:
            continue
        for key in ("local_patch", "local_heading_patch"):
            if key in entry and (fetched or args.force):
                apply_patch(dest, entry[key])

    ensure_symlink(ROOT / "third_party/GVHMR/inputs/checkpoints", ROOT / "models/motion")
    ensure_symlink(ROOT / "third_party/GMR/assets/body_models", ROOT / "models/motion/body_models")
    print("Third-party setup complete.")


if __name__ == "__main__":
    main()

"""Check publishable text and optional Git objects without printing private values."""

import argparse
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "personal_home_path": re.compile(r"(?<![A-Za-z0-9:/])/(?:home|Users|root)/[^\s\"'<>`]+"),
    "windows_user_path": re.compile(r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\s\"'<>`]+"),
    "email_address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "github_token": re.compile(r"(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def local_identities():
    values = set()
    for key in ("user.name", "user.email"):
        result = subprocess.run(["git", "config", "--global", "--get-all", key], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        values.update(line.strip() for line in result.stdout.splitlines() if len(line.strip()) >= 4)
    username = Path.home().name
    if len(username) >= 4 and username.lower() not in {"root", "user", "runner", "admin"}:
        values.add(username)
    return [re.compile(r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", re.IGNORECASE)
            for value in values]


def inspect(text, identities):
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        for rule, pattern in PATTERNS.items():
            matches = list(pattern.finditer(line))
            if rule == "email_address":
                matches = [match for match in matches
                           if match.group().split("@", 1)[1].lower() not in {"example.invalid", "example.com", "example.org"}]
            if matches:
                findings.append((number, rule))
        if any(pattern.search(line) for pattern in identities):
            findings.append((number, "local_identity"))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="Also inspect all stored commit and text blob objects")
    args = parser.parse_args()
    identities = local_identities()
    findings = []
    files = sorted(set(git("ls-files", "-co", "--exclude-standard", "-z").split(b"\0")) - {b""})
    checked = 0
    for entry in files:
        relative = os.fsdecode(entry)
        findings.extend((relative, 0, rule) for _, rule in inspect(relative, identities))
        path = ROOT / relative
        if path.is_symlink():
            value = os.readlink(path)
        elif path.is_file():
            try:
                value = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
        else:
            continue
        checked += 1
        findings.extend((relative, number, rule) for number, rule in inspect(value, identities))
    objects = 0
    if args.history:
        inventory = git("cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype)")
        for line in inventory.decode().splitlines():
            oid, kind = line.split()
            if kind not in {"commit", "blob", "tag"}:
                continue
            try:
                value = git("cat-file", kind, oid).decode("utf-8")
            except UnicodeDecodeError:
                continue
            objects += 1
            findings.extend((f"git-object:{oid[:12]}", number, rule) for number, rule in inspect(value, identities))
    for source, number, rule in findings:
        print(f"{source}:{number}: {rule}")
    if findings:
        print(f"Privacy check failed: {len(findings)} findings. Matched private values are not printed.")
        return 1
    print(f"Privacy check passed: {checked} publishable text files, {objects} Git text objects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

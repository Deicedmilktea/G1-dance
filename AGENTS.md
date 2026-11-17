# Public repository privacy

- Treat every tracked file and commit as public. Do not add the repository owner's personal identity, email, hostname, account details, credentials or workstation paths.
- Use repository-relative paths in documentation and shared configuration. Discover runtime paths dynamically; keep machine-specific settings in ignored local files.
- Keep raw training outputs, videos, weights, caches and recovery backups out of Git. Sanitize any report selected for publication before adding it.
- Preserve upstream copyright and license attribution.
- Run `python3 tools/check_public_privacy.py --history` before publishing. This repository uses a generic local Git author identity; preserve it unless the owner explicitly requests another public identity.

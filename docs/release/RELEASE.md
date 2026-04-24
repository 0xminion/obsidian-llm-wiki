# Release process

This project treats installability as part of correctness. A release is not ready
because tests pass from a source checkout; it is ready when the built artifact can
initialize a fresh vault.

## Pre-release checklist

1. Update [`CHANGELOG.md`](CHANGELOG.md) under `Unreleased` or the target version.
2. Confirm `pyproject.toml` version matches the release tag.
3. Run local verification:

```bash
ruff check .
pyflakes pipeline tests
pytest -q
python -m build --wheel --outdir dist
python -m venv /tmp/obsidian-wheel-smoke
/tmp/obsidian-wheel-smoke/bin/pip install dist/*.whl
/tmp/obsidian-wheel-smoke/bin/pipeline init /tmp/obsidian-wheel-vault --force --quiet
test -f /tmp/obsidian-wheel-vault/Meta/prompts/batch-create.prompt
test -f /tmp/obsidian-wheel-vault/Meta/Templates/Entry.md
```

4. Push a branch and wait for GitHub Actions CI to pass.
5. Tag with `vX.Y.Z` only after CI is green.
6. Build and publish from the tag, not from a dirty working tree.

## Versioning rule

- Patch: bug fixes, packaging fixes, test/CI improvements.
- Minor: new CLI features or schema-compatible pipeline capabilities.
- Major: vault schema migrations that require manual user action.

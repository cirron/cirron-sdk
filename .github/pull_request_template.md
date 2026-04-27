## Description

Briefly describe what this PR changes and why.

## Related Issues

Closes #<issue_number>

## Checklist

- [ ] Tests added / updated and `uv run pytest tests/unit -v` passes locally.
- [ ] README and/or `docs/` updated if public surface or user-facing behavior changed.
- [ ] `docs/spool-format.md` bumped if the local spool format changed (it's public API).
- [ ] New dependencies (if any) listed below with name, version, and reason. See `CONTRIBUTING.md`.

## New dependencies

<!-- Delete this section if no new deps. Otherwise list them:
- `pyarrow >= 17.0.0` (runtime, optional extra `[arrow]`): needed for Parquet column pushdown in `ci.load()`.
- `httpx == 0.27.2` (dev): replaces `requests` in test fixtures for HTTP/2 support.
-->

## Trace / output sample (if applicable)

<!-- Paste a `ci.trace()` snippet, spool diff, or CLI output that shows the new behavior. -->

## Additional Notes

<!-- Perf notes, follow-up tickets, deferred work. -->

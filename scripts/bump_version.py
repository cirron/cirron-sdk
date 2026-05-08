#!/usr/bin/env python3
"""Write a new version into pyproject.toml.

Invoked by auto's @auto-it/exec plugin "version" hook with the
already-computed target version (e.g. "0.1.0", "1.0.0-rc.0") as the
sole argument. Auto computes the bump from PR labels itself; this
script just persists the result.

Surgical text replace (not a TOML round-trip) so comments and
formatting in pyproject.toml are preserved.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
VERSION_RE = re.compile(r'^(version\s*=\s*)"[^"]*"', re.MULTILINE)


def main(version: str) -> None:
    text = PYPROJECT.read_text()
    new_text, n = VERSION_RE.subn(rf'\1"{version}"', text, count=1)
    if n != 1:
        sys.exit("error: could not locate `version = \"...\"` in pyproject.toml")
    PYPROJECT.write_text(new_text)
    print(f"pyproject.toml -> {version}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: bump_version.py <version>")
    main(sys.argv[1])

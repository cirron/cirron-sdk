#!/usr/bin/env python3
"""Write a new version into pyproject.toml.

Invoked by auto's @auto-it/exec plugin "version" hook with the
already-computed target version (e.g. "0.1.0", "1.0.0-rc.0") as the
sole argument. Auto computes the bump from PR labels itself; this
script just persists the result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tomli_w
import tomllib

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def main(version: str) -> None:
    data = tomllib.loads(PYPROJECT.read_text())
    data["project"]["version"] = version
    PYPROJECT.write_text(tomli_w.dumps(data))
    print(f"pyproject.toml -> {version}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: bump_version.py <version>")
    main(sys.argv[1])

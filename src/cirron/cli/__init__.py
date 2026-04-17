"""``cirron`` CLI — stub for SDK-18.

Per spec §4.11: ``cirron login``, ``cirron status``, ``cirron spool inspect |
flush | clear``. The CLI is a thin wrapper over the ``Cirron`` class.
"""

from __future__ import annotations

import sys


def main() -> int:
    sys.stderr.write(
        "cirron CLI is not implemented yet (SDK-18).\n"
        "Planned commands: login, status, spool {inspect,flush,clear}.\n"
    )
    return 1

"""``ci.mark(name, value, **attrs)`` — scaffold no-op."""

from __future__ import annotations

import warnings
from typing import Any, Union

_warned = False


def mark(name: str, value: Union[float, int, str, bool], **attrs: Any) -> None:
    global _warned
    if not _warned:
        warnings.warn(
            "cirron.mark() runtime is not implemented yet (SDK-13); "
            "calls are no-ops.",
            stacklevel=2,
        )
        _warned = True

"""``ci.wrap()`` — sklearn opt-in estimator wrapper.

Per spec §4.8: no auto-hook for sklearn; users call ``ci.wrap(estimator)`` to
open a scope around ``fit``. SDK-13 provides the real wrapping; today this is
a passthrough that warns once.
"""

from __future__ import annotations

import warnings
from typing import Any

_warned = False


def wrap(estimator: Any) -> Any:
    global _warned
    if not _warned:
        warnings.warn(
            "cirron.wrap() runtime is not implemented yet (SDK-13); "
            "estimator is returned unchanged.",
            stacklevel=2,
        )
        _warned = True
    return estimator

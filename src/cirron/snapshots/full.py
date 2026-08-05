"""Full weight/gradient snapshot mode.

The :mod:`cirron.snapshots.sampled` path with ``should_sample``
short-circuited, so every epoch boundary serializes every tensor.
"""

from __future__ import annotations

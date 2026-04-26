"""Full weight/gradient snapshot mode.

``snapshots="full"`` is ``sampled`` with the sample-rate roll always
succeeding: every epoch boundary serializes every weight + gradient
tensor. Debug-only by design — the SDK explicitly calls out that this
is not recommended for 100M+ parameter models.

The implementation is just the sampled path with ``should_sample``
short-circuited, so this module is intentionally tiny; both modes share
the same ``serialize_and_enqueue`` plumbing.
"""

from __future__ import annotations

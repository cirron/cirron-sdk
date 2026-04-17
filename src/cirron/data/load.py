"""``ci.load()`` — unified data-access dispatcher (spec §4.7).

Today this is a scaffold that preserves the public signature so downstream
code can type-check and import against it. The real implementation — source
scheme dispatch (S3/GCS/Azure/Postgres/Databricks/Snowflake), match /
where / columns / map wiring, lazy + batched returns, registered-dataset
resolution via the platform — lands in SDK-28 (dispatcher + registered
datasets), SDK-29 (filesystem pattern matching), SDK-30 (SQL sources),
SDK-31 (row/batch ``map=`` transform).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

Source = str | list[str | dict[str, Any]]
As = Literal["pandas", "polars", "iter", "tensor", "hf"]


def load(
    source: Source,
    match: dict[str, Any] | None = None,
    where: str | None = None,
    columns: list[str] | None = None,
    map: Callable[..., Any] | None = None,
    as_: As = "pandas",
    lazy: bool = False,
    batch_size: int = 10_000,
) -> Any:
    raise NotImplementedError(
        "cirron.load() runtime is not implemented yet (SDK-28). "
        "Signature is fixed per spec §4.7; source dispatch, match/where/columns, "
        "and as_/lazy/batch_size wiring land with the real implementation."
    )

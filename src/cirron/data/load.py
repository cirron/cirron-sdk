"""``ci.load()`` — unified data-access dispatcher (spec §4.7).

Today this is a scaffold that preserves the public signature so downstream
code can type-check and import against it. The real implementation — source
scheme dispatch (S3/GCS/Azure/Postgres/Databricks/Snowflake), match /
where / columns / map wiring, lazy + batched returns, registered-dataset
resolution via the platform — lands in SDK-13+.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Union

Source = Union[str, List[Union[str, Dict[str, Any]]]]
As = Literal["pandas", "polars", "iter", "tensor", "hf"]


def load(
    source: Source,
    match: Optional[Dict[str, Any]] = None,
    where: Optional[str] = None,
    columns: Optional[List[str]] = None,
    map: Optional[Callable[..., Any]] = None,
    as_: As = "pandas",
    lazy: bool = False,
    batch_size: int = 10_000,
) -> Any:
    raise NotImplementedError(
        "cirron.load() runtime is not implemented yet (SDK-13). "
        "Signature is fixed per spec §4.7; source dispatch, match/where/columns, "
        "and as_/lazy/batch_size wiring land with the real implementation."
    )

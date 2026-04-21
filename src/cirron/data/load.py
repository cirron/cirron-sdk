"""``ci.load()`` — unified data-access dispatcher (spec §4.7, SDK-28).

The dispatcher parses a flat keyword signature into a :class:`LoadRequest`,
resolves each input to a concrete ``DataSource``, enforces the
:mod:`cirron.data.size` tier policy, runs the load(s) in parallel when
there are multiple, concatenates, and converts to the requested return
type via :mod:`cirron.data.returns`.

What SDK-28 executes vs accepts-and-raises:

================  ============  ==========================================
parameter         executed?     notes
================  ============  ==========================================
``source``        yes           "local" | "platform" | scheme-in-string
``columns``       yes           pushed to parquet reader; slice otherwise
``as_``           yes           pandas | polars | iter | tensor | hf
``lazy``          yes           returns :class:`LazyHandle`
``batch_size``    yes           only applied when ``as_='iter'``
``confirm_large`` yes           size-tier override
``match`` /``ext``     raises   SDK-29 filesystem glob
``where``              raises   SDK-30 SQL pushdown
``map``                raises   SDK-31 row/batch transform
``search``/``top_k``   raises   platform embeddings feature
================  ============  ==========================================

The "accepts-and-raises" pattern keeps the signature stable now so
downstream code can be written against the final shape; calls that use
those params today get a clear ``NotImplementedError`` naming the story
that will deliver them rather than a cryptic ``TypeError``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from cirron.core.errors import CirronDependencyError
from cirron.data.lazy import LazyHandle
from cirron.data.returns import create_adapter
from cirron.data.size import enforce_tiers
from cirron.data.sources import DataSource, SourceConfig

if TYPE_CHECKING:
    from cirron.core.config import Cirron

log = logging.getLogger("cirron.load")

Source = Literal["local", "platform"]
As = Literal["pandas", "polars", "iter", "tensor", "hf"]
_VALID_SOURCES = ("local", "platform")
_VALID_AS = ("pandas", "polars", "iter", "tensor", "hf")

_SCHEMES = {
    "s3": "s3",
    "gs": "gs",
    "gcs": "gs",
    "azure": "azure",
    "file": "local",
    "postgres": "postgres",
    "databricks": "databricks",
    "snowflake": "snowflake",
}


@dataclass
class LoadRequest:
    """Normalized form of one positional ``name`` + the kwargs.

    The dispatcher builds one request per input (single string → one,
    list → one per element), then hands each to a ``DataSource``. Sources
    read from ``request`` to decide how to filter / project.
    """

    name: str
    source: Source
    match: str | None = None
    ext: list[str] | None = None
    columns: list[str] | None = None
    map: Callable[..., Any] | None = None
    where: str | None = None
    search: str | None = None
    top_k: int | None = None
    as_: As = "pandas"
    lazy: bool = False
    batch_size: int = 10_000
    confirm_large: bool = False
    scheme: str | None = field(default=None)  # populated if name has "://"


def load(
    name: str | list[str],
    *,
    source: Source = "local",
    match: str | None = None,
    ext: list[str] | None = None,
    columns: list[str] | None = None,
    map: Callable[..., Any] | None = None,  # noqa: A002 — public API
    where: str | None = None,
    search: str | None = None,
    top_k: int | None = None,
    as_: As = "pandas",
    lazy: bool = False,
    batch_size: int = 10_000,
    confirm_large: bool = False,
    cirron: Cirron | None = None,
) -> Any:
    """Load data from local disk, the Cirron platform, or an external URI.

    See module docstring for the parameter matrix and deferred-story map.
    """
    if source not in _VALID_SOURCES:
        raise ValueError(f"source must be one of {_VALID_SOURCES}, got {source!r}")
    if as_ not in _VALID_AS:
        raise ValueError(f"as_ must be one of {_VALID_AS}, got {as_!r}")

    names = [name] if isinstance(name, str) else list(name)
    if not names:
        raise ValueError("ci.load() requires at least one source name")

    cirron = cirron or _default_cirron()

    requests = [
        _build_request(
            n,
            source=source,
            match=match,
            ext=ext,
            columns=columns,
            map_=map,
            where=where,
            search=search,
            top_k=top_k,
            as_=as_,
            lazy=lazy,
            batch_size=batch_size,
            confirm_large=confirm_large,
        )
        for n in names
    ]

    for req in requests:
        _reject_unsupported(req)

    sources = [_resolve_source(req, cirron) for req in requests]

    _enforce_size(sources, requests, cirron)

    if lazy:
        return LazyHandle(lambda: _run_and_convert(sources, requests))
    return _run_and_convert(sources, requests)


def _default_cirron() -> Cirron:
    from cirron.core.config import get_default

    return get_default()


def _build_request(
    name: str,
    *,
    source: Source,
    match: str | None,
    ext: list[str] | None,
    columns: list[str] | None,
    map_: Callable[..., Any] | None,
    where: str | None,
    search: str | None,
    top_k: int | None,
    as_: As,
    lazy: bool,
    batch_size: int,
    confirm_large: bool,
) -> LoadRequest:
    scheme = _scheme_of(name)
    return LoadRequest(
        name=name,
        source=source,
        match=match,
        ext=ext,
        columns=columns,
        map=map_,
        where=where,
        search=search,
        top_k=top_k,
        as_=as_,
        lazy=lazy,
        batch_size=batch_size,
        confirm_large=confirm_large,
        scheme=scheme,
    )


def _scheme_of(name: str) -> str | None:
    if "://" not in name:
        return None
    raw = name.split("://", 1)[0].lower()
    mapped = _SCHEMES.get(raw)
    if mapped is None:
        raise ValueError(f"unknown URI scheme '{raw}' in source '{name}'")
    return mapped


def _reject_unsupported(req: LoadRequest) -> None:
    """Fail loudly on parameters whose execution isn't shipped yet.

    Kept centralised so every source inherits the same deferred-field
    contract; no source needs its own ``if request.match: raise`` block.
    """
    if req.match is not None or req.ext is not None:
        raise NotImplementedError(
            "match= / ext= filesystem glob execution lands in SDK-29; the "
            "parameters are accepted today so call sites remain stable."
        )
    if req.map is not None:
        raise NotImplementedError(
            "map= row/batch transform lands in SDK-31; the parameter is "
            "accepted today so call sites remain stable."
        )
    if req.where is not None:
        raise NotImplementedError(
            "where= SQL pushdown lands in SDK-30 (Postgres / Databricks / "
            "Snowflake sources); accepted today so call sites remain stable."
        )
    if req.search is not None or req.top_k is not None:
        raise NotImplementedError(
            "search= / top_k= semantic search requires the platform vector "
            "index, which is not yet available."
        )


def _resolve_source(req: LoadRequest, cirron: Cirron) -> DataSource:
    if req.scheme is not None:
        return _scheme_source(req)
    if req.source == "platform":
        from cirron.data.sources.registered import RegisteredDataset

        return RegisteredDataset(req.name, cirron, req).resolve()
    # default local
    from cirron.data.sources.local import LocalDataSource

    return LocalDataSource(
        SourceConfig(source_type="local", path=req.name),
        req,
    )


def _split_object_path(path: str) -> tuple[str | None, str | None]:
    """Classify an object-store path suffix into (folder_path, key).

    - ``""`` (bare bucket) → folder listing rooted at the bucket.
    - ``"prefix/"`` → folder listing with that prefix.
    - ``"prefix/file.parquet"`` → single-object key.

    Without this, ``s3://bucket`` would set ``path=""`` and the S3
    source would call ``get_object(Key="")`` which is not a valid key.
    """
    if not path:
        return ("", None)
    if path.endswith("/"):
        return (path, None)
    return (None, path)


def _scheme_source(req: LoadRequest) -> DataSource:
    """Build a scheme-specific source. Scheme URIs are resource pointers;
    credentials come from the user's environment (boto3 / google-cloud
    default credential chains) — the SDK doesn't hold them."""
    assert req.scheme is not None
    if req.scheme == "s3":
        from cirron.data.sources.s3 import S3DataSource

        bucket, _, path = req.name[len("s3://") :].partition("/")
        folder, key = _split_object_path(path)
        return S3DataSource(
            SourceConfig(
                source_type="s3",
                bucket_name=bucket,
                folder_path=folder,
                path=key,
            ),
            req,
        )
    if req.scheme == "gs":
        from cirron.data.sources.gcs import GCSDataSource

        prefix = "gs://" if req.name.startswith("gs://") else "gcs://"
        bucket, _, path = req.name[len(prefix) :].partition("/")
        folder, key = _split_object_path(path)
        return GCSDataSource(
            SourceConfig(
                source_type="gs",
                bucket_name=bucket,
                folder_path=folder,
                path=key,
            ),
            req,
        )
    if req.scheme == "azure":
        from cirron.data.sources.azure import AzureDataSource

        # azure://<account>/<container>/<path>
        body = req.name[len("azure://") :]
        account, _, rest = body.partition("/")
        container, _, path = rest.partition("/")
        folder, key = _split_object_path(path)
        # TODO(SDK-29): AzureDataSource still builds account_url from
        # container_name (see sources/azure.py:~26), so account_name
        # populated here is ignored and azure:// loads hit the wrong
        # endpoint in prod. The dispatcher correctly captures account
        # and container separately — the fix is on the consumer side
        # and is owned by SDK-29.
        return AzureDataSource(
            SourceConfig(
                source_type="azure",
                account_name=account,
                container_name=container,
                folder_path=folder,
                path=key,
            ),
            req,
        )
    if req.scheme == "local":
        from cirron.data.sources.local import LocalDataSource

        return LocalDataSource(
            SourceConfig(source_type="local", path=req.name[len("file://") :]),
            req,
        )
    if req.scheme in ("postgres", "databricks", "snowflake"):
        raise NotImplementedError(
            f"{req.scheme}:// sources land in SDK-30; the scheme is parsed "
            "today so call sites remain stable."
        )
    raise ValueError(f"unreachable scheme: {req.scheme}")


def _enforce_size(
    sources: list[DataSource],
    requests: list[LoadRequest],
    cirron: Cirron,
) -> None:
    total = 0
    count = 0
    any_sized = False
    for src in sources:
        bytes_, objs = src.estimate_size()
        if bytes_ is not None:
            total += bytes_
            any_sized = True
        if objs is not None:
            count += objs
    if not any_sized:
        return
    # Requests share the same ``confirm_large`` by construction (all built
    # from the same call).
    enforce_tiers(
        total,
        count or None,
        warn_bytes=cirron.load_warn_bytes,
        max_bytes=cirron.load_max_bytes,
        confirm_large=requests[0].confirm_large,
    )


def _run_and_convert(sources: list[DataSource], requests: list[LoadRequest]) -> Any:
    if len(sources) == 1:
        raw = sources[0].load()
    else:
        with ThreadPoolExecutor(max_workers=min(8, len(sources))) as ex:
            loaded = list(ex.map(lambda s: s.load(), sources))
        raw = _concat(loaded)

    req = requests[0]
    return _convert(raw, req)


def _concat(parts: list[Any]) -> Any:
    """Concatenate a homogeneous list of source results."""
    if not parts:
        raise ValueError("no data produced by any source")
    first = parts[0]
    try:
        import pandas as pd

        if isinstance(first, pd.DataFrame):
            return pd.concat(parts, ignore_index=True)
    except ImportError:
        pass
    try:
        import polars as pl

        if isinstance(first, pl.DataFrame):
            return pl.concat(parts)
    except ImportError:
        pass
    if isinstance(first, list):
        out: list[Any] = []
        for p in parts:
            out.extend(p)
        return out
    raise ValueError(
        f"cannot concatenate ci.load() results of type {type(first).__name__}; "
        "load sources individually and combine them yourself."
    )


def _convert(raw: Any, req: LoadRequest) -> Any:
    # ``iter`` goes straight to the generator without building an adapter
    # for every row; everything else routes through the adapter layer.
    if req.as_ == "iter":
        return _to_iter(raw, req.batch_size)

    try:
        adapter = create_adapter(raw)
    except ValueError:
        # Raw isn't a tabular type the adapter layer understands (e.g.,
        # a JSON document → dict/list, an image → PIL.Image, plain text
        # → str). pandas is the permissive default: return the raw
        # payload so ``ci.load('cfg.json')`` or ``ci.load('img.png')``
        # "just works" on a laptop. The non-pandas targets (polars,
        # tensor, hf) are explicit opt-ins to a tabular conversion —
        # if we can't build an adapter, the caller asked for something
        # we genuinely can't produce, so raise a clear error rather
        # than silently returning a mis-typed object.
        if req.as_ == "pandas":
            return raw
        raise CirronDependencyError(
            f"ci.load(as_={req.as_!r}) requires a tabular source; "
            f"got {type(raw).__name__}. Use as_='pandas' (default) or "
            "point at a CSV/Parquet/JSONL source."
        ) from None

    if req.as_ == "pandas":
        _require_pandas()
        return adapter.to_pandas()
    if req.as_ == "polars":
        return adapter.to_polars()
    if req.as_ == "tensor":
        return adapter.to_tensor()
    if req.as_ == "hf":
        return adapter.to_hf()
    raise ValueError(f"unreachable as_: {req.as_}")


def _to_iter(raw: Any, batch_size: int) -> Any:
    try:
        adapter = create_adapter(raw)
    except ValueError:
        if isinstance(raw, list):
            return iter(raw)
        raise
    return adapter.to_iter(batch_size=batch_size)


def _require_pandas() -> None:
    try:
        import pandas  # noqa: F401
    except ImportError as e:
        raise CirronDependencyError(
            "ci.load(as_='pandas') requires 'pandas'. "
            "Install with: pip install 'cirron-sdk[pandas]'"
        ) from e

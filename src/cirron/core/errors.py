class CirronError(Exception):
    """Base class for SDK-raised errors."""


class CirronSecretNotFound(CirronError):
    """Raised by ``ci.secret`` when the requested secret is not mounted."""


class CirronDependencyError(CirronError):
    """Raised when an optional dependency (pandas, polars, torch, ...) is required but not installed."""

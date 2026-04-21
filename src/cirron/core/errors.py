class CirronError(Exception):
    """Base class for SDK-raised errors."""


class CirronSecretNotFound(CirronError):
    """Raised by ``ci.secret`` when the requested secret is not mounted."""


class CirronDependencyError(CirronError):
    """Raised when an optional dependency (pandas, polars, torch, ...) is required but not installed."""


class CirronDatasetNotFound(CirronError):
    """Raised when ``ci.load(name, source='platform')`` cannot resolve ``name`` on the platform."""


class CirronPlatformRequired(CirronError):
    """Raised when a platform-only operation is attempted without platform credentials or connectivity."""


class CirronDataSizeError(CirronError):
    """Raised when a ``ci.load()`` query would pull more than ``load_max_bytes`` without ``confirm_large=True``."""

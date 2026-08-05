"""Exception hierarchy raised by the SDK.

Every error below inherits :class:`CirronError`, so callers can catch the
whole surface with one ``except``. Caller bugs raise plain ``ValueError`` /
``TypeError`` instead. See the error rule in ``CONTRIBUTING.md``.
"""


class CirronError(Exception):
    """Base class for SDK-raised errors."""


class CirronSecretNotFound(CirronError):
    """Raised by ``ci.secret`` when the requested secret is not mounted."""


class CirronDependencyError(CirronError):
    """Raised when a required optional dependency is not installed."""


class CirronDatasetNotFound(CirronError):
    """Raised when a platform ``ci.load()`` cannot resolve the given name."""


class CirronPlatformRequired(CirronError):
    """Raised when a platform-only operation lacks credentials or connectivity."""


class CirronDataSizeError(CirronError):
    """Raised when a ``ci.load()`` query exceeds ``load_max_bytes``.

    Pass ``confirm_large=True`` to proceed anyway.
    """

"""Guards on the public ``ci.*`` surface as users encounter it.

These assert the canonical-docstring rule in ``docs/style-guide.md``: the
module-level re-export owns the full docstring, because that is what ``help()``
and IDE hover resolve to. Thinning one of those docstrings in favour of a
``Mirrors ...`` pointer is the exact regression this file exists to catch.
"""

import inspect

import pytest

import cirron as ci
from cirron.data.load import load as _load_impl

PUBLIC_DELEGATORS = (
    "profile",
    "scope",
    "mark",
    "epochs",
    "batches",
    "env",
    "secret",
    "load",
    "inference",
    "wrap",
)


@pytest.mark.parametrize("name", PUBLIC_DELEGATORS)
def test_public_delegator_carries_the_canonical_docstring(name):
    doc = inspect.getdoc(getattr(ci, name))
    assert doc, f"ci.{name} has no docstring"
    assert "Args:" in doc, f"ci.{name} lost its Args block"
    assert len(doc.splitlines()) >= 5, f"ci.{name} docstring is a stub"


@pytest.mark.parametrize("name", PUBLIC_DELEGATORS)
def test_public_delegator_docstring_is_not_a_mirror_pointer(name):
    """The pointer belongs on the inner layers, never on the public surface.

    A ``Mirrors``/``Implements`` sentence here would mean the canonical
    docstring moved inward, leaving ``help(ci.<name>)`` pointing elsewhere.
    """
    doc = inspect.getdoc(getattr(ci, name)) or ""
    assert "Mirrors :func:" not in doc
    assert "Implements :func:" not in doc


def test_ci_load_signature_matches_the_implementation_minus_cirron():
    """``ci.load`` must expose the real parameters, not ``*args, **kwargs``.

    It previously forwarded opaquely, so ``help()`` and IDE hover showed
    nothing for the flagship data API and the docstring compensated with an
    ad-hoc parameter list that could drift from the implementation.
    """
    public = list(inspect.signature(ci.load).parameters)
    impl = [p for p in inspect.signature(_load_impl).parameters if p != "cirron"]
    assert public == impl


def test_ci_load_accepts_only_name_positionally():
    params = inspect.signature(ci.load).parameters
    positional = [
        p.name for p in params.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert positional == ["name"]


def test_cirron_instance_methods_mirror_the_module_level_functions():
    for name in PUBLIC_DELEGATORS:
        assert hasattr(ci.Cirron, name), f"Cirron.{name} is missing"

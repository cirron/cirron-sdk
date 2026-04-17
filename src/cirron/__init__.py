"""Cirron SDK — Python-side profiler and data loader for the Cirron platform.

Surface area is defined in ``docs/spec.md`` §4. Runtime behavior lands
story-by-story — scope in SDK-9, mark in SDK-10, flush in SDK-11, transport
in SDK-12, ``profile()`` orchestration in SDK-13, wrappers in SDK-14,
``Cirron`` config class in SDK-16, secrets in SDK-17, CLI in SDK-18, hooks
in SDK-19–23, snapshots in SDK-24/25, inference in SDK-26/27, data loading
in SDK-28–31. Until those land, the names below resolve to stubs that warn
when invoked (``scope``/``mark``/``epochs``/``batches``/``inference``/``wrap``)
or raise ``NotImplementedError`` (``load``). ``profile`` is wired as a
YAML-config scaffold per the existing contract in ``tests/unit/test_profile.py``.
"""

from cirron.core.config import (
    Cirron,
    CirronYamlError,
    find_cirron_yaml,
    load_cirron_yaml,
)
from cirron.core.env import env
from cirron.core.errors import (
    CirronDependencyError,
    CirronError,
    CirronSecretNotFound,
)
from cirron.core.mark import mark
from cirron.core.profiler import Profiler, profile
from cirron.core.scope import scope
from cirron.core.wrappers import batches, epochs
from cirron.core.yaml_types import CirronYaml, ProfilingConfig, ServingConfig
from cirron.data.load import load
from cirron.hooks.sklearn import wrap
from cirron.inference.decorator import inference
from cirron.secrets.client import get_secret

__version__ = "0.1.0"

__all__ = [
    "Cirron",
    "CirronDependencyError",
    "CirronError",
    "CirronSecretNotFound",
    "CirronYaml",
    "CirronYamlError",
    "ProfilingConfig",
    "Profiler",
    "ServingConfig",
    "batches",
    "env",
    "epochs",
    "find_cirron_yaml",
    "get_secret",
    "inference",
    "load",
    "load_cirron_yaml",
    "mark",
    "profile",
    "scope",
    "wrap",
]

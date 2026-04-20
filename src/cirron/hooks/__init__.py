"""Framework hook package.

Importing this package triggers each framework module's
``register_installer(...)`` side effect, populating the registry that
``ci.profile()`` consults via ``install_hooks``. sklearn is opt-in via
``ci.wrap()`` and intentionally stays out of the auto-install path.
"""

from cirron.hooks import tensorflow, torch, transformers  # noqa: F401

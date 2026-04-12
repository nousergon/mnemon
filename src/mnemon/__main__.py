"""Allow ``python -m mnemon`` as an entry point.

Delegates to :func:`mnemon.cli.main` so ``python -m mnemon <cmd>``
behaves identically to the console-script installed by the package. This
matters for subprocess invocations (integration tests, CI) where the
console script may not be on PATH but the package is importable.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    main()

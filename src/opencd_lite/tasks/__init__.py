"""Training tasks (Lightning modules) — the ``train`` extra.

Importing this package requires ``lightning``; the core model package
(:mod:`opencd_lite.models`) never depends on it.
"""

from .change_detection import ChangeDetectionTask

__all__ = ["ChangeDetectionTask"]

"""The Provider seam: the pure scheduler's view of an execution target, and the
one adapter (``BackendProvider``) that bridges it to today's ``Backend``s."""

from __future__ import annotations

from omnirun.providers.adapter import BackendProvider
from omnirun.providers.base import CancelMode, Provider

__all__ = [
    "BackendProvider",
    "CancelMode",
    "Provider",
]

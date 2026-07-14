"""Scoped progress narration for long, otherwise-silent operations.

A `submit` spends most of its wall-clock inside a backend provisioning a remote
(a Colab VM cold-starts in ~30-90s, a Kaggle kernel queues, an ssh worker builds
its venv). Those steps live deep under `Control` → `Provider` → `Backend.submit`,
far from the CLI. Threading a callback through every layer (and every test
double) would be noisy, so progress flows through a context-scoped sink instead:

- The CLI opens `reporting(sink)` around the slow call; `sink` renders each
  message (e.g. onto a rich status line).
- Any code running inside that scope calls `report("provisioning VM…")`.
- Outside a scope (the daemon, tests that don't opt in) `report` is a no-op, so
  library code can narrate unconditionally with zero coupling to the UI.

The sink is a `ContextVar`, so it is confined to the calling task/thread and
never leaks between concurrent submits.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator
from contextlib import contextmanager

ProgressSink = Callable[[str], None]

_sink: contextvars.ContextVar[ProgressSink | None] = contextvars.ContextVar(
    "omnirun_progress_sink", default=None
)


def report(message: str) -> None:
    """Narrate one step of the current operation. No-op outside a `reporting`
    scope, so library code can call it unconditionally."""
    sink = _sink.get()
    if sink is not None:
        try:
            sink(message)
        except Exception:
            # Progress rendering must never break the operation it describes.
            pass


@contextmanager
def reporting(sink: ProgressSink) -> Iterator[None]:
    """Install `sink` as the progress reporter for the duration of the block."""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)

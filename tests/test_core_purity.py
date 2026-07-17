"""The core scheduler language must never name a concrete backend.

The reconciler/scheduler/budget/provider-seam speak only in generic resource
vocabulary ("held resource", "capacity-occupying resource", "placement"). A
backend concept leaking into these modules — even in a comment or a log string —
is how the duck-typed reap flags smeared backend knowledge across the core in
the first place. This guard reads each core module's source text and fails if any
backend-specific word appears, naming the file, word, and line number.
"""

from __future__ import annotations

from pathlib import Path

from omnirun import budget, control, daemon, scheduler
from omnirun.engine import engine as engine_engine
from omnirun.engine import jobstream as engine_jobstream
from omnirun.engine import observer as engine_observer
from omnirun.engine import outcomes as engine_outcomes
from omnirun.engine import providertypes as engine_providertypes
from omnirun.engine import supervisor as engine_supervisor
from omnirun.engine import workitems as engine_workitems
from omnirun.providers import adapter, asyncadapter
from omnirun.providers import base as providers_base

# The core modules that must stay free of backend-specific vocabulary.
_CORE_MODULES = (
    control,
    scheduler,
    budget,
    providers_base,
    adapter,
    asyncadapter,
    daemon,
    engine_outcomes,
    engine_providertypes,
    engine_workitems,
    engine_supervisor,
    engine_observer,
    engine_jobstream,
    engine_engine,
)

# Substrings that name a concrete backend / backend-specific concept.
_BANNED = (
    "colab",
    "kaggle",
    "slurm",
    "vast",
    "runpod",
    "thunder",
    "notebook",
    "kernel",
)


def test_core_modules_name_no_backend() -> None:
    violations: list[str] = []
    for module in _CORE_MODULES:
        assert module.__file__ is not None
        path = Path(module.__file__)
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            lowered = line.lower()
            for word in _BANNED:
                if word in lowered:
                    violations.append(
                        f"{path}:{lineno}: backend word {word!r} in core module: "
                        f"{line.strip()!r}"
                    )
    assert not violations, (
        "core scheduler language leaks backend concepts:\n" + "\n".join(violations)
    )

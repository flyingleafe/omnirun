"""Client-side group/matrix expansion (FUT-1).

``submit --matrix "lr=0.1,0.3×seed=0,1"`` (or ``--group NAME --count N``)
expands CLIENT-SIDE into one ``group``-stamped set of specs sharing ONE
resolved code plan: the base spec's plan/``.env`` blob are resolved once and
copied onto every cell, so a 12-cell sweep costs one ``gh``/git round, not 12.
Per-cell parameters ride ``env_vars`` (the job command reads them from its
environment). Deliberately minimal — no DAG engine; dependencies are the
separate ``depends_on`` edge (FUT-2).
"""

from __future__ import annotations

import re
import secrets

from omnirun.models import JobSpec

# Dimension separators accepted in a --matrix expression: the spec's "×" plus
# "*" for keyboards without it.
_DIM_SPLIT = re.compile(r"[×*]")


def parse_matrix(expr: str) -> list[dict[str, str]]:
    """``"lr=0.1,0.3×seed=0,1"`` → the cross product, one dict per cell.

    Each dimension is ``key=v1,v2,…``; dimensions are separated by ``×`` (or
    ``*``). Cells are produced in row-major order (later dimensions vary
    fastest). Raises ``ValueError`` on a malformed expression."""
    dims: list[tuple[str, list[str]]] = []
    for raw in _DIM_SPLIT.split(expr):
        part = raw.strip()
        if not part:
            continue
        key, sep, values = part.partition("=")
        key = key.strip()
        vals = [v.strip() for v in values.split(",") if v.strip()]
        if not sep or not key or not vals:
            raise ValueError(
                f"bad matrix dimension {part!r}: expected key=v1,v2 "
                "(dimensions separated by ×)"
            )
        if any(key == k for k, _ in dims):
            raise ValueError(f"duplicate matrix dimension {key!r}")
        dims.append((key, vals))
    if not dims:
        raise ValueError(f"empty matrix expression {expr!r}")
    cells: list[dict[str, str]] = [{}]
    for key, vals in dims:
        cells = [{**cell, key: v} for cell in cells for v in vals]
    return cells


def make_group_name(base: str) -> str:
    """A fresh, human-recognizable group name derived from the job name."""
    safe = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-") or "group"
    return f"{safe[:24]}-{secrets.token_hex(2)}"


def _cell_suffix(cell: dict[str, str]) -> str:
    parts = [re.sub(r"[^A-Za-z0-9.]+", "", f"{k}{v}") for k, v in cell.items()]
    return "-".join(p for p in parts if p)


def expand_cells(
    spec: JobSpec, cells: list[dict[str, str]], group: str
) -> list[JobSpec]:
    """Expand a base *spec* (code plan already resolved) into per-cell specs.

    Every cell keeps the shared plan/env blob verbatim; its parameters are
    merged into ``env_vars`` (cell wins), its name gains the cell suffix, and
    each spec gets a fresh unique ``job_id`` plus the shared ``group``."""
    out: list[JobSpec] = []
    for cell in cells:
        suffix = _cell_suffix(cell)
        name = f"{spec.name}-{suffix}" if suffix else spec.name
        out.append(
            spec.model_copy(
                update={
                    "job_id": JobSpec.make_job_id(name),
                    "name": name,
                    "env_vars": {**spec.env_vars, **cell},
                    "group": group,
                }
            )
        )
    return out

"""Tolerant tar extraction for the notebook backends (kaggle, colab), which
return a job's outputs as a tarball the client must unpack.

Python >=3.12 defaults tar extraction to the PEP-706 ``data`` filter. That
filter is right (it refuses path traversal, device nodes and links pointing
outside the destination), but ``extractall`` aborts the *whole* archive the
first time the filter rejects a member. The common offender is a symlink whose
target is an absolute path — every Weights & Biases run leaves one at
``wandb/*/logs/debug-core.log`` — so a single such link would strand every
output of an otherwise-successful job.

``extract_all`` applies the same ``data`` filter per member but skips (with a
warning) any member the filter rejects, so the rest of the tree still lands.
"""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path


def extract_all(tf: tarfile.TarFile, dest: Path) -> list[str]:
    """Extract every member of ``tf`` under ``dest`` with the safe ``data``
    filter, skipping (not aborting on) any member the filter rejects. Returns
    the names of the skipped members (also warned to stderr)."""
    skipped: list[str] = []
    for member in tf.getmembers():
        try:
            tf.extract(member, dest, filter="data")
        except tarfile.FilterError as e:
            skipped.append(member.name)
            print(
                f"omnirun: warning: skipped unsafe archive entry "
                f"{member.name!r} ({type(e).__name__})",
                file=sys.stderr,
            )
    return skipped

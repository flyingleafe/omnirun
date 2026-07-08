"""Tolerant tar extraction (shared by the kaggle + colab pull paths)."""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

from omnirun.backends import tarsafe


def _tar_with_abs_symlink(dest_target: str) -> io.BytesIO:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"real output\n"
        good = tarfile.TarInfo("outputs/results.txt")
        good.size = len(data)
        tf.addfile(good, io.BytesIO(data))
        link = tarfile.TarInfo("outputs/wandb/run-x/logs/debug-core.log")
        link.type = tarfile.SYMTYPE
        link.linkname = dest_target
        tf.addfile(link)
    buf.seek(0)
    return buf


def test_extract_all_skips_absolute_symlink_and_keeps_rest(tmp_path: Path) -> None:
    # every W&B run leaves such a link; the default data filter would abort here.
    buf = _tar_with_abs_symlink("/etc/hostname")
    with tarfile.open(fileobj=buf) as tf:
        skipped = tarsafe.extract_all(tf, tmp_path)
    assert skipped == ["outputs/wandb/run-x/logs/debug-core.log"]
    assert (tmp_path / "outputs" / "results.txt").read_bytes() == b"real output\n"
    assert not (tmp_path / "outputs" / "wandb" / "run-x" / "logs").exists()


def test_extract_all_extracts_normal_tree_completely(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("a")
    (src / "sub" / "b.txt").write_text("b")
    os.symlink("a.txt", src / "rel")  # a *relative* symlink is allowed by data filter
    tar = tmp_path / "t.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(src, arcname="outputs")
    out = tmp_path / "out"
    with tarfile.open(tar) as tf:
        skipped = tarsafe.extract_all(tf, out)
    assert skipped == []
    assert (out / "outputs" / "a.txt").read_text() == "a"
    assert (out / "outputs" / "sub" / "b.txt").read_text() == "b"
    assert (out / "outputs" / "rel").is_symlink()

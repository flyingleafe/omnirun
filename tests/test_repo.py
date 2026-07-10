from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from omnirun import repo as repo_mod
from omnirun.repo import (
    RepoError,
    capture_repo_state,
    create_bundle,
    find_repo_root,
    remote_clone_plan,
    repo_slug,
    worker_clone_url,
)
from tests.conftest import git

# --- find_repo_root ---------------------------------------------------------


def test_find_repo_root_from_subdir(sample_repo: Path) -> None:
    sub = sample_repo / "a" / "b"
    sub.mkdir(parents=True)
    assert find_repo_root(sub) == sample_repo


def test_find_repo_root_outside_repo(tmp_path: Path) -> None:
    lonely = tmp_path / "no-repo"
    lonely.mkdir()
    with pytest.raises(RepoError, match="not inside a git repo"):
        find_repo_root(lonely)


# --- slug --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("git@github.com:me/omnirun.git", "omnirun"),
        ("https://github.com/me/my-repo", "my-repo"),
        ("https://github.com/me/my-repo.git/", "my-repo"),
        ("ssh://git@host:2222/team/w e i r d!.git", "w-e-i-r-d"),
    ],
)
def test_slug_from_remote(url: str, expected: str, tmp_path: Path) -> None:
    assert repo_slug(url, tmp_path) == expected


def test_slug_from_dir_name(tmp_path: Path) -> None:
    root = tmp_path / "My Project"
    root.mkdir()
    assert repo_slug(None, root) == "My-Project"
    assert repo_slug("", root) == "My-Project"


# --- clean / dirty ------------------------------------------------------------


def test_capture_clean(sample_repo: Path) -> None:
    ref = capture_repo_state(sample_repo)
    assert ref.sha == git(sample_repo, "rev-parse", "HEAD")
    assert ref.branch == "main"
    assert ref.slug == "sample"
    assert ref.remote_url == ""
    assert ref.dirty is False


def test_capture_dirty_tracked(sample_repo: Path) -> None:
    (sample_repo / "job.py").write_text("changed\n")
    with pytest.raises(RepoError, match=r"modified/staged.*commit"):
        capture_repo_state(sample_repo)


def test_capture_dirty_untracked(sample_repo: Path) -> None:
    (sample_repo / "scratch.txt").write_text("wip\n")
    with pytest.raises(RepoError, match="untracked"):
        capture_repo_state(sample_repo)


def test_capture_allow_dirty_snapshots_working_tree(sample_repo: Path) -> None:
    """--dirty ships a wip commit whose tree is the working tree, not plain HEAD,
    and leaves the user's HEAD/index/working tree untouched (issue #6)."""
    head = git(sample_repo, "rev-parse", "HEAD")
    (sample_repo / "job.py").write_text("changed\n")
    (sample_repo / "scratch.txt").write_text("wip\n")  # untracked, non-ignored

    ref = capture_repo_state(sample_repo, allow_dirty=True)

    assert ref.dirty is True
    # A real, distinct commit parented on HEAD — not HEAD itself.
    assert ref.sha != head
    assert git(sample_repo, "rev-parse", f"{ref.sha}^") == head
    # The wip tree carries the uncommitted edits and the untracked file.
    assert git(sample_repo, "show", f"{ref.sha}:job.py") == "changed"
    assert git(sample_repo, "show", f"{ref.sha}:scratch.txt") == "wip"
    # The user's repo is not mutated: HEAD unchanged, tree still dirty on disk.
    assert git(sample_repo, "rev-parse", "HEAD") == head
    assert (sample_repo / "job.py").read_text() == "changed\n"
    assert git(sample_repo, "status", "--porcelain") != ""


def test_capture_allow_dirty_excludes_gitignored(sample_repo: Path) -> None:
    """A gitignored file (e.g. .env) rides out-of-band, never in the wip tree."""
    (sample_repo / ".gitignore").write_text(".env\n")
    git(sample_repo, "add", ".gitignore")
    git(sample_repo, "commit", "-q", "-m", "add gitignore")
    (sample_repo / ".env").write_text("SECRET=1\n")
    (sample_repo / "job.py").write_text("changed\n")

    ref = capture_repo_state(sample_repo, allow_dirty=True)

    assert git(sample_repo, "show", f"{ref.sha}:job.py") == "changed"
    missing = subprocess.run(
        ["git", "-C", str(sample_repo), "cat-file", "-e", f"{ref.sha}:.env"],
        capture_output=True,
    )
    assert missing.returncode != 0  # .env is absent from the wip tree


def test_capture_allow_dirty_captures_deletion(sample_repo: Path) -> None:
    """A deleted tracked file is absent from the wip tree."""
    (sample_repo / "job.py").unlink()
    ref = capture_repo_state(sample_repo, allow_dirty=True)
    gone = subprocess.run(
        ["git", "-C", str(sample_repo), "cat-file", "-e", f"{ref.sha}:job.py"],
        capture_output=True,
    )
    assert gone.returncode != 0


def test_capture_allow_dirty_clean_tree_is_head(sample_repo: Path) -> None:
    """--dirty on a clean tree is a no-op: sha stays HEAD, no wip commit."""
    ref = capture_repo_state(sample_repo, allow_dirty=True)
    assert ref.dirty is False
    assert ref.sha == git(sample_repo, "rev-parse", "HEAD")


# --- pushed check ---------------------------------------------------------------


@pytest.fixture
def origin(sample_repo: Path, tmp_path: Path) -> Path:
    """A bare 'origin' remote with the current main branch pushed."""
    bare = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", str(bare))
    git(sample_repo, "remote", "add", "origin", str(bare))
    git(sample_repo, "push", "-q", "origin", "main")
    return bare


def _commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", name)
    return git(repo, "rev-parse", "HEAD")


def test_pushed_head_passes(sample_repo: Path, origin: Path) -> None:
    ref = capture_repo_state(sample_repo)
    assert ref.remote_url == str(origin)
    assert ref.slug == "origin"  # slug follows the remote url basename


def test_unpushed_head_raises(sample_repo: Path, origin: Path) -> None:
    _commit(sample_repo, "new.txt")
    with pytest.raises(RepoError, match="push"):
        capture_repo_state(sample_repo)


def test_auto_push(sample_repo: Path, origin: Path) -> None:
    sha = _commit(sample_repo, "new.txt")
    ref = capture_repo_state(sample_repo, auto_push=True)
    assert ref.sha == sha
    # the sha actually arrived on the remote
    subprocess.run(
        ["git", "--git-dir", str(origin), "cat-file", "-e", f"{sha}^{{commit}}"],
        check=True,
    )


def test_auto_push_failure_surfaces_stderr(sample_repo: Path, origin: Path) -> None:
    _commit(sample_repo, "new.txt")
    git(sample_repo, "remote", "set-url", "origin", str(origin) + "-missing")
    with pytest.raises(RepoError, match="git push origin main"):
        capture_repo_state(sample_repo, auto_push=True)


def test_no_origin_skips_pushed_check(sample_repo: Path) -> None:
    _commit(sample_repo, "new.txt")
    ref = capture_repo_state(sample_repo)  # must not raise
    assert ref.remote_url == ""


def test_detached_head_skips_pushed_check(sample_repo: Path, origin: Path) -> None:
    git(sample_repo, "checkout", "-q", "--detach")
    _commit(sample_repo, "detached.txt")  # unpushed, but detached -> no check
    ref = capture_repo_state(sample_repo)
    assert ref.branch == "detached"


# --- bundles ----------------------------------------------------------------------


def test_create_bundle_clonable(sample_repo: Path, tmp_path: Path) -> None:
    sha = git(sample_repo, "rev-parse", "HEAD")
    dest = tmp_path / "deep" / "job.bundle"
    assert create_bundle(sample_repo, sha, dest) == dest
    assert dest.is_file()
    # the temp ref did not leak into the source repo
    assert "refs/omnirun" not in git(sample_repo, "for-each-ref")

    # bare-clone from the bundle (what bootstrap does) and find the sha
    clone = tmp_path / "clone.git"
    git(tmp_path, "clone", "-q", "--bare", str(dest), str(clone))
    git(clone, "cat-file", "-e", f"{sha}^{{commit}}")

    # fetch path (pre-existing bare repo) works too
    bare2 = tmp_path / "existing.git"
    git(tmp_path, "init", "-q", "--bare", str(bare2))
    git(bare2, "fetch", "-q", str(dest), "+refs/*:refs/*")
    git(bare2, "cat-file", "-e", f"{sha}^{{commit}}")


def test_create_bundle_bad_sha(sample_repo: Path, tmp_path: Path) -> None:
    with pytest.raises(RepoError):
        create_bundle(sample_repo, "0" * 40, tmp_path / "x.bundle")


# --- public-repo direct clone -----------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("git@github.com:me/proj.git", "https://github.com/me/proj.git"),
        ("ssh://git@github.com/me/proj.git", "https://github.com/me/proj.git"),
        ("https://github.com/me/proj.git", "https://github.com/me/proj.git"),
        ("git://example.com/x/y.git", "https://example.com/x/y.git"),
        ("/local/path/repo", None),  # local remote -> ship a bundle
        ("", None),
    ],
)
def test_worker_clone_url(url: str, expected: str | None) -> None:
    assert worker_clone_url(url) == expected


def test_remote_clone_plan_public_reachable(
    sample_repo: Path, origin: Path, monkeypatch
) -> None:
    # stub the network side (public detection + url mapping); the reachability /
    # ancestry logic runs for real against the local bare `origin`.
    monkeypatch.setattr(repo_mod, "remote_is_public", lambda _u: True)
    monkeypatch.setattr(repo_mod, "worker_clone_url", lambda _u: str(origin))
    ref = capture_repo_state(sample_repo)  # HEAD is pushed to origin/main
    assert remote_clone_plan(ref, sample_repo) == str(origin)


def test_remote_clone_plan_private_ships_bundle(
    sample_repo: Path, origin: Path, monkeypatch
) -> None:
    monkeypatch.setattr(repo_mod, "remote_is_public", lambda _u: False)
    monkeypatch.setattr(repo_mod, "worker_clone_url", lambda _u: str(origin))
    ref = capture_repo_state(sample_repo)
    assert remote_clone_plan(ref, sample_repo) is None


def test_remote_clone_plan_dirty_ships_bundle(
    sample_repo: Path, origin: Path, monkeypatch
) -> None:
    monkeypatch.setattr(repo_mod, "remote_is_public", lambda _u: True)
    monkeypatch.setattr(repo_mod, "worker_clone_url", lambda _u: str(origin))
    ref = capture_repo_state(sample_repo).model_copy(update={"dirty": True})
    assert remote_clone_plan(ref, sample_repo) is None


def test_remote_clone_plan_unreachable_sha_ships_bundle(
    sample_repo: Path, origin: Path, monkeypatch
) -> None:
    monkeypatch.setattr(repo_mod, "remote_is_public", lambda _u: True)
    monkeypatch.setattr(repo_mod, "worker_clone_url", lambda _u: str(origin))
    ref = capture_repo_state(sample_repo)
    # a local commit that never reached origin: descendant of the tip, so not an
    # ancestor of it -> a direct clone could not find it -> bundle.
    unpushed = _commit(sample_repo, "unpushed.txt")
    ref = ref.model_copy(update={"sha": unpushed})
    assert remote_clone_plan(ref, sample_repo) is None

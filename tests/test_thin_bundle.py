"""Thin bundle (CODE-2c): committed-but-unpushed shas ride the spec as a
delta bundle over the best origin-reachable base — daemonless AND daemon mode
(the placer needs no local git objects)."""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

import pytest

from omnirun import repo as repo_mod
from omnirun.backends import jobdir
from omnirun.bootstrap import BootstrapParams, CodeSource, generate_bootstrap
from omnirun.deploykey import resolve_code_plan
from omnirun.execlayer.local import LocalExec
from omnirun.models import CodePlan, DeployKey
from omnirun.repo import (
    RepoError,
    capture_repo_state,
    create_thin_bundle,
    sha_on_origin,
)
from tests.conftest import git


@pytest.fixture
def origin(sample_repo: Path, tmp_path: Path) -> Path:
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


def test_sha_on_origin(sample_repo: Path, origin: Path) -> None:
    pushed = git(sample_repo, "rev-parse", "HEAD")
    assert sha_on_origin(sample_repo, pushed)
    unpushed = _commit(sample_repo, "new.txt")
    assert not sha_on_origin(sample_repo, unpushed)


def test_thin_bundle_is_a_delta_and_worker_flow_lands_the_sha(
    sample_repo: Path, origin: Path, tmp_path: Path
) -> None:
    # A big file in the PUSHED history must not be re-shipped by the delta.
    (sample_repo / "big.bin").write_bytes(b"x" * 300_000)
    git(sample_repo, "add", "big.bin")
    git(sample_repo, "commit", "-q", "-m", "big")
    git(sample_repo, "push", "-q", "origin", "main")
    git(sample_repo, "fetch", "-q", "origin")  # refresh remote-tracking refs
    sha = _commit(sample_repo, "delta.txt")

    bundle = create_thin_bundle(sample_repo, sha, tmp_path / "thin.bundle")
    assert bundle.stat().st_size < 100_000  # delta only — no big.bin

    # The worker flow: clone origin (the base), then fetch the bundle on top.
    worker = tmp_path / "worker.git"
    subprocess.run(
        ["git", "clone", "-q", "--bare", str(origin), str(worker)], check=True
    )
    subprocess.run(
        [
            "git",
            "--git-dir",
            str(worker),
            "fetch",
            "-q",
            str(bundle),
            "+refs/*:refs/*",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "--git-dir", str(worker), "cat-file", "-e", f"{sha}^{{commit}}"],
        check=True,
    )


def test_thin_bundle_size_guard(
    sample_repo: Path, origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repo_mod, "THIN_BUNDLE_MAX_BYTES", 64)
    sha = _commit(sample_repo, "delta.txt")
    with pytest.raises(RepoError, match="push your branch"):
        create_thin_bundle(sample_repo, sha, tmp_path / "thin.bundle")
    assert not (tmp_path / "thin.bundle").exists()


def test_resolve_code_plan_attaches_bundle_for_public_unpushed(
    sample_repo: Path, origin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sha = _commit(sample_repo, "delta.txt")
    ref = capture_repo_state(sample_repo)  # no longer refuses unpushed
    assert ref.sha == sha
    monkeypatch.setattr(repo_mod, "remote_is_public", lambda url: True)
    monkeypatch.setattr(
        repo_mod, "worker_clone_url", lambda url: "https://forge/x/y.git"
    )
    plan = resolve_code_plan(
        ref, get_key=lambda origin: None, register_key=lambda dk: None
    )
    assert plan.kind == "remote"
    assert plan.clone_url == "https://forge/x/y.git"
    assert plan.bundle_b64 is not None
    raw = base64.b64decode(plan.bundle_b64)
    assert raw.startswith(b"# v2 git bundle")


def test_resolve_code_plan_attaches_bundle_for_known_private_unpushed(
    sample_repo: Path, origin: Path
) -> None:
    _commit(sample_repo, "delta.txt")
    ref = capture_repo_state(sample_repo).model_copy(
        update={"remote_url": "git@github.com:me/p.git"}
    )
    dk = DeployKey(origin="git@github.com:me/p.git", private_key="K", public_key="P")
    plan = resolve_code_plan(
        ref, get_key=lambda origin: dk, register_key=lambda dk: None
    )
    assert plan.kind == "private"
    assert plan.bundle_b64 is not None


def test_resolve_code_plan_pushed_sha_ships_no_bundle(
    sample_repo: Path, origin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = capture_repo_state(sample_repo)
    dk = DeployKey(origin=ref.remote_url, private_key="K", public_key="P")
    monkeypatch.setattr(repo_mod, "ssh_clone_url", lambda url: "git@host:o/r.git")
    plan = resolve_code_plan(
        ref, get_key=lambda origin: dk, register_key=lambda dk: None
    )
    assert plan.kind == "private"
    assert plan.bundle_b64 is None


# --------------------------------------------------------------------------- bootstrap


def test_bootstrap_fetches_bundle_after_origin_clone(job_spec) -> None:
    params = BootstrapParams(
        code=CodeSource(
            kind="remote", clone_url="https://forge/x.git", fetch_bundle=True
        )
    )
    script = generate_bootstrap(job_spec, params)
    clone_at = script.index('git clone --bare "$CLONE_URL"')
    fetch_at = script.index('fetch "$BUNDLE"')
    assert clone_at < fetch_at  # base first, delta second
    assert 'fail "bundle fetch failed"' in script


def test_bootstrap_no_bundle_block_without_flag(job_spec) -> None:
    params = BootstrapParams(
        code=CodeSource(kind="remote", clone_url="https://forge/x.git")
    )
    script = generate_bootstrap(job_spec, params)
    assert 'fetch "$BUNDLE"' not in script


# --------------------------------------------------------------------------- staging


def test_stage_job_delivers_and_decodes_bundle(
    job_spec, tmp_path: Path, sample_repo: Path
) -> None:
    payload = b"BUNDLEBYTES\x00\x01"
    spec = job_spec.model_copy(
        update={
            "code": CodePlan(
                kind="remote",
                clone_url="https://forge/x.git",
                origin="https://forge/x.git",
                bundle_b64=base64.b64encode(payload).decode(),
            )
        }
    )
    exec_ = LocalExec()
    root = str(tmp_path / "root")
    params = BootstrapParams(project_root=str(tmp_path / "proj"))
    job_dir = jobdir.stage_job(exec_, spec, sample_repo, params, root)
    assert (Path(job_dir) / "bundle.git").read_bytes() == payload
    assert not (Path(job_dir) / "bundle.b64").exists()  # decoded + removed
    assert params.code.fetch_bundle is True
    script = (Path(job_dir) / "bootstrap.sh").read_text()
    assert 'fetch "$BUNDLE"' in script

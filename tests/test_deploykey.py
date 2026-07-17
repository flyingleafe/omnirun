"""Client-side code-delivery resolution (``deploykey.resolve_code_plan``) and the
deploy-key store round-trip. These cover the branch matrix without touching the
network: git/`gh`/public-ness helpers are stubbed per case."""

from __future__ import annotations

import pytest

from omnirun import deploykey, repo
from omnirun.models import CodePlan, DeployKey, RepoRef
from omnirun.repo import RepoError


def _ref(**over: object) -> RepoRef:
    base: dict[str, object] = {
        "remote_url": "git@github.com:me/proj.git",
        "sha": "a" * 40,
        "branch": "main",
        "slug": "proj",
        "local_root": None,
    }
    base.update(over)
    return RepoRef.model_validate(base)


class _Keys:
    """A tiny in-memory get/register pair standing in for the Client verbs."""

    def __init__(self, seed: dict[str, DeployKey] | None = None) -> None:
        self.store: dict[str, DeployKey] = dict(seed or {})
        self.registered: list[DeployKey] = []

    def get(self, origin: str) -> DeployKey | None:
        return self.store.get(origin)

    def register(self, dk: DeployKey) -> None:
        self.store[dk.origin] = dk
        self.registered.append(dk)


def test_known_private_origin_uses_ssh_clone(monkeypatch):
    """An origin we already hold a key for → private ssh clone, no gh/public probe."""
    monkeypatch.setattr(repo, "remote_is_public", lambda url: pytest.fail("probed"))
    keys = _Keys(
        {
            "git@github.com:me/proj.git": DeployKey(
                origin="git@github.com:me/proj.git", private_key="k", public_key="p"
            )
        }
    )
    plan = deploykey.resolve_code_plan(
        _ref(), get_key=keys.get, register_key=keys.register
    )
    assert plan.kind == "private"
    assert plan.clone_url == "git@github.com:me/proj.git"
    assert plan.deploy_key_material is None  # material is injected later, never here


def test_public_origin_clones_anonymously(monkeypatch):
    monkeypatch.setattr(
        repo, "worker_clone_url", lambda url: "https://github.com/me/proj.git"
    )
    monkeypatch.setattr(repo, "remote_is_public", lambda url: True)
    keys = _Keys()
    plan = deploykey.resolve_code_plan(
        _ref(), get_key=keys.get, register_key=keys.register
    )
    assert plan.kind == "remote"
    assert plan.clone_url == "https://github.com/me/proj.git"
    assert not keys.registered


def test_private_github_provisions_key_via_gh(monkeypatch):
    monkeypatch.setattr(repo, "remote_is_public", lambda url: False)
    monkeypatch.setattr(repo, "gh_can_admin", lambda slug: True)
    monkeypatch.setattr(
        repo, "generate_deploy_keypair", lambda comment="": ("PRIV", "PUB")
    )
    monkeypatch.setattr(repo, "gh_create_deploy_key", lambda slug, pub, title: "42")
    keys = _Keys()
    plan = deploykey.resolve_code_plan(
        _ref(), get_key=keys.get, register_key=keys.register
    )
    assert plan.kind == "private"
    assert plan.clone_url == "git@github.com:me/proj.git"
    assert len(keys.registered) == 1
    dk = keys.registered[0]
    assert dk.private_key == "PRIV" and dk.public_key == "PUB" and dk.key_id == "42"


def test_private_no_key_no_gh_falls_back_to_local(monkeypatch):
    monkeypatch.setattr(repo, "remote_is_public", lambda url: False)
    monkeypatch.setattr(repo, "gh_can_admin", lambda slug: False)
    keys = _Keys()
    plan = deploykey.resolve_code_plan(
        _ref(local_root="/repo"), get_key=keys.get, register_key=keys.register
    )
    assert plan.kind == "local"
    assert not keys.registered


def test_private_no_key_no_gh_remote_daemon_refuses_local_fallback(monkeypatch):
    # A REMOTE daemon can't use the client's local checkout, so the local
    # fallback is refused HERE (at submit) with the actionable message — not
    # deferred to a cryptic [Errno 2] at placement (issue #23).
    monkeypatch.setattr(repo, "remote_is_public", lambda url: False)
    monkeypatch.setattr(repo, "gh_can_admin", lambda slug: False)
    keys = _Keys()
    with pytest.raises(RepoError, match="remote.*deploy-key add|deploy-key add"):
        deploykey.resolve_code_plan(
            _ref(local_root="/repo"),
            get_key=keys.get,
            register_key=keys.register,
            allow_local_fallback=False,
        )


def test_private_no_key_no_gh_no_local_raises(monkeypatch):
    monkeypatch.setattr(repo, "remote_is_public", lambda url: False)
    monkeypatch.setattr(repo, "gh_can_admin", lambda slug: False)
    keys = _Keys()
    with pytest.raises(RepoError, match="deploy-key add"):
        deploykey.resolve_code_plan(
            _ref(), get_key=keys.get, register_key=keys.register
        )


def test_non_github_private_only_manual(monkeypatch):
    """A non-github private origin cannot auto-provision (no gh); with a local root
    it falls back to local objects."""
    monkeypatch.setattr(repo, "remote_is_public", lambda url: False)
    keys = _Keys()
    plan = deploykey.resolve_code_plan(
        _ref(remote_url="git@gitlab.com:me/proj.git", local_root="/repo"),
        get_key=keys.get,
        register_key=keys.register,
    )
    assert plan.kind == "local"


def test_code_plan_default_is_remote():
    assert CodePlan().kind == "remote"


def test_resolve_spec_code_stamps_env_dotenv_from_local_dotenv(tmp_path):
    """The gitignored <local_root>/.env is read CLIENT-SIDE into spec.env_dotenv,
    so the placer (possibly a remote daemon with no access to this filesystem)
    can deliver it. `code` is pre-set so only the .env read runs."""
    from omnirun.client import resolve_spec_code
    from omnirun.models import JobSpec

    (tmp_path / ".env").write_text("SECRET=hunter2\n")
    keys = _Keys()
    spec = JobSpec(
        job_id="j-000001",
        name="j",
        command="echo hi",
        repo=_ref(local_root=str(tmp_path)),
        code=CodePlan(kind="remote", clone_url="https://x"),
    )
    out = resolve_spec_code(spec, keys.get, keys.register)
    assert out.env_dotenv == "SECRET=hunter2\n"


def test_resolve_spec_code_no_dotenv_leaves_env_none(tmp_path):
    from omnirun.client import resolve_spec_code
    from omnirun.models import JobSpec

    keys = _Keys()  # tmp_path has no .env
    spec = JobSpec(
        job_id="j-000002",
        name="j",
        command="echo hi",
        repo=_ref(local_root=str(tmp_path)),
        code=CodePlan(kind="remote", clone_url="https://x"),
    )
    out = resolve_spec_code(spec, keys.get, keys.register)
    assert out.env_dotenv is None

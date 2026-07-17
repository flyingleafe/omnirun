"""Client-side code-delivery resolution: turn a captured ``RepoRef`` into a
``CodePlan`` the placer (daemon or in-process controller) can honor WITHOUT the
client's local git objects.

Workers always clone from origin (DESIGN, relaxed invariant #3): public repos
anonymously, private repos with a per-origin read-only deploy key auto-provisioned
through ``gh``. Only when there is no cloneable origin do we fall back to
delivering the repo from the placer's own local objects (``local`` — daemonless
or co-located only).

This module holds no state: the caller passes ``get_key``/``register_key``
callables (the Client's store-backed deploy-key verbs), so the same resolution
works for a LocalClient (hits the store) and a RemoteClient (asks the daemon).
"""

from __future__ import annotations

import socket
from collections.abc import Callable
from pathlib import Path

from omnirun import repo
from omnirun.models import CodePlan, DeployKey, RepoRef
from omnirun.progress import report
from omnirun.repo import RepoError

GetKey = Callable[[str], DeployKey | None]
RegisterKey = Callable[[DeployKey], None]


def _key_title() -> str:
    return f"omnirun-{socket.gethostname()}"


def _public_clone_url(ref: RepoRef, root: Path | None) -> str | None:
    """The anonymous https url the worker can clone, or None when not public.

    With a local checkout we use ``remote_clone_plan`` (proves the sha is
    reachable on the remote branch); without one we can only check public-ness."""
    if not ref.remote_url or ref.branch == "detached":
        return None
    if root is not None:
        return repo.remote_clone_plan(ref, root)
    url = repo.worker_clone_url(ref.remote_url)
    return url if (url and repo.remote_is_public(ref.remote_url)) else None


def resolve_code_plan(
    ref: RepoRef,
    *,
    get_key: GetKey,
    register_key: RegisterKey,
    allow_local_fallback: bool = True,
) -> CodePlan:
    """Decide how the worker gets the code for *ref*.

    Order: a known-private origin (we already hold a deploy key) → ssh clone; a
    public + reachable origin → anonymous https clone; a private origin we can
    provision for (github + `gh` admin) → auto-create a read-only deploy key →
    ssh clone; otherwise fall back to the placer's local objects (``local``), or
    raise with actionable guidance when there is nothing to fall back to.

    ``allow_local_fallback`` is False when the placer is a REMOTE daemon (it has
    no access to this client's filesystem): the ``local`` fallback is then
    refused HERE, at submit, with the actionable message — rather than the daemon
    failing placement later with a cryptic ``[Errno 2]`` on the client's path."""
    origin = ref.remote_url
    root = Path(ref.local_root) if ref.local_root else None

    # Known-private: we already hold a key for this origin — clone via ssh.
    if origin and get_key(origin) is not None:
        ssh_url = repo.ssh_clone_url(origin)
        if ssh_url:
            return CodePlan(kind="private", clone_url=ssh_url, origin=origin)

    # Public + reachable: anonymous clone, no key.
    public = _public_clone_url(ref, root)
    if public is not None:
        return CodePlan(kind="remote", clone_url=public, origin=origin)

    # Private origin: provision a read-only deploy key if `gh` lets us.
    if origin:
        ssh_url = repo.ssh_clone_url(origin)
        slug = repo.github_slug(origin)
        if ssh_url and slug and repo.gh_can_admin(slug):
            report(f"provisioning a read-only deploy key for {slug}…")
            priv, pub = repo.generate_deploy_keypair(comment=f"omnirun-{ref.slug}")
            key_id = repo.gh_create_deploy_key(slug, pub, title=_key_title())
            register_key(
                DeployKey(
                    origin=origin, private_key=priv, public_key=pub, key_id=key_id
                )
            )
            return CodePlan(kind="private", clone_url=ssh_url, origin=origin)

    # Fallback: the placer delivers from its OWN local objects. Only a co-located
    # placer (daemonless, or a loopback daemon) can — a remote daemon has no
    # access to this client's filesystem.
    if root is not None and allow_local_fallback:
        return CodePlan(kind="local", origin=origin)

    # Nothing worked — raise the most actionable message for the situation.
    if origin:
        remote_note = (
            " and the configured daemon is remote (it cannot use this machine's "
            "local checkout)"
            if root
            is not None  # we HAVE a checkout, but the remote placer can't use it
            else ", and this process has no local checkout to fall back to"
        )
        raise RepoError(
            f"{origin} is private, no deploy key is registered{remote_note}. "
            "Authenticate `gh` as a repo admin and retry, or register a key "
            f"manually: omnirun deploy-key add {origin} <keyfile>"
        )
    if root is not None:  # local-only repo (no origin) but the placer is remote
        raise RepoError(
            "this repo has no origin remote to clone from, so only a co-located "
            "placer can deliver it — but the configured daemon is remote. Add a "
            "remote the worker can clone, or run daemonless (omnirun --local)."
        )
    raise RepoError(
        "cannot determine how to deliver the repo to the worker (no origin remote "
        "and no local checkout)"
    )

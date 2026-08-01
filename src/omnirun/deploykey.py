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


def _anon_clone_url(ref: RepoRef) -> str | None:
    """The anonymous https url a credential-less worker can clone, or None when
    the origin is not public (or is not a cloneable url at all)."""
    if not ref.remote_url or ref.branch == "detached":
        return None
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

    Order: a public + reachable origin → anonymous https clone; a known-private
    origin (we already hold a deploy key) → ssh clone; a private origin we can
    provision for (github + `gh` admin) → auto-create a read-only deploy key →
    ssh clone; otherwise fall back to the placer's local objects (``local``), or
    raise with actionable guidance when there is nothing to fall back to.

    Public-ness is asked BEFORE the key cache, and a public origin always wins:
    a repo can become public after we provisioned a key for it, and the cheapest
    correct clone must not depend on when the key was minted.

    A committed-but-UNPUSHED sha with a cloneable origin gets the origin plan
    PLUS a thin delta bundle (CODE-2c): the worker clones origin for the base
    and fetches the bundle on top, so unpushed work runs in daemonless AND
    daemon mode without the local-push restriction.

    ``allow_local_fallback`` is False when the placer is a REMOTE daemon (it has
    no access to this client's filesystem): the ``local`` fallback is then
    refused HERE, at submit, with the actionable message — rather than the daemon
    failing placement later with a cryptic ``[Errno 2]`` on the client's path."""
    origin = ref.remote_url
    root = Path(ref.local_root) if ref.local_root else None
    unpushed = (
        bool(origin) and root is not None and not repo.sha_on_origin(root, ref.sha)
    )

    def _bundle(plan: CodePlan) -> CodePlan:
        if unpushed and root is not None:
            report("bundling the unpushed delta…")
            return plan.model_copy(
                update={"bundle_b64": repo.thin_bundle_b64(root, ref.sha)}
            )
        return plan

    # Public origin: anonymous clone, no key, always. This is asked FIRST, and
    # it is the ONLY question that decides between public and private delivery.
    # A stored key proves the origin WAS private when we minted it, never that
    # it still is; and an unpushed or unprovable SHA is a property of the commit,
    # not of the origin — a deploy key cannot conjure an object the forge does
    # not have. Either mistake sends a worker with no outbound ssh to the forge
    # into a private fetch it never needed.
    origin_public = bool(origin) and repo.remote_is_public(origin)
    anon = repo.worker_clone_url(origin) if origin and origin_public else None
    if anon is not None:
        return _bundle(CodePlan(kind="remote", clone_url=anon, origin=origin))

    # Known-private: we already hold a key for this origin — clone via ssh.
    if origin and get_key(origin) is not None:
        ssh_url = repo.ssh_clone_url(origin)
        if ssh_url:
            return _bundle(CodePlan(kind="private", clone_url=ssh_url, origin=origin))

    # Private origin: provision a read-only deploy key if `gh` lets us. Guarded
    # on origin_public: we mint a key ONLY for an origin we positively know is
    # private. Every other reason to be here (an undecipherable url, a sha we
    # cannot place on the remote, a probe that failed) must not put a key on a
    # public repo — that key then poisons every later submit for that origin.
    if origin and not origin_public:
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
            return _bundle(CodePlan(kind="private", clone_url=ssh_url, origin=origin))

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
        if origin_public:
            raise RepoError(
                f"{origin} is public, but omnirun cannot derive an anonymous "
                f"clone url from it{remote_note}. Give the repo an http(s) or "
                "scp-style origin a credential-less worker can clone."
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

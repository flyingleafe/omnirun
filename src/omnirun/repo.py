"""Client-side git state: repo discovery, clean/pushed checks, RepoRef capture,
and bundle creation for notebook backends.

Every git call is a `git -C <root>` subprocess — we never chdir and never use
gitpython.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from omnirun.models import RepoRef

# scp-style ssh remote, e.g. git@github.com:owner/repo.git (no scheme, host:path)
_SCP_URL = re.compile(r"^(?:[^@/]+@)?(?P<host>[^:/]+):(?P<path>.+)$")


class RepoError(RuntimeError):
    """Repo-state problem; the message tells the user what to do about it."""


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=600,
    )


def find_repo_root(start: Path | None = None) -> Path:
    """Toplevel of the git repo containing `start` (default: cwd)."""
    start = start or Path.cwd()
    r = _git(start, "rev-parse", "--show-toplevel")
    if r.returncode != 0:
        raise RepoError(
            f"{start} is not inside a git repository — omnirun submits jobs from a "
            "repo; cd into one (or `git init` and commit your code) and retry"
        )
    return Path(r.stdout.strip())


def repo_slug(remote_url: str | None, root: Path) -> str:
    """Filesystem-safe repo name: remote url basename (sans .git) or dir name."""
    if remote_url:
        base = remote_url.rstrip("/").split("/")[-1]
        base = base.split(":")[-1]  # scp-style ssh urls: git@host:repo.git
        base = base.removesuffix(".git")
    else:
        base = root.name
    return re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-.") or "repo"


def capture_repo_state(
    root: Path, *, allow_dirty: bool = False, auto_push: bool = False
) -> RepoRef:
    """Snapshot the repo state a job will run against, enforcing the submit-time
    invariants: clean working tree (unless allow_dirty) and, when an origin
    remote exists, HEAD pushed (unless auto_push does it for us)."""
    st = _git(root, "status", "--porcelain")
    if st.returncode != 0:
        raise RepoError(f"not a git repository: {root} ({st.stderr.strip()})")
    lines = [ln for ln in st.stdout.splitlines() if ln.strip()]
    untracked = [ln for ln in lines if ln.startswith("??")]
    tracked = [ln for ln in lines if not ln.startswith("??")]
    dirty = bool(lines)
    if dirty and not allow_dirty:
        parts = []
        if tracked:
            parts.append(f"{len(tracked)} modified/staged file(s)")
        if untracked:
            parts.append(f"{len(untracked)} untracked file(s)")
        raise RepoError(
            f"working tree at {root} is dirty ({', '.join(parts)}) — commit your "
            "changes so the job runs a real revision, or pass --dirty to run "
            "HEAD as-is"
        )

    head = _git(root, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise RepoError(
            f"repo at {root} has no commits yet — make an initial commit first"
        )
    sha = head.stdout.strip()

    br = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    branch = br.stdout.strip() if br.returncode == 0 else "HEAD"
    if branch == "HEAD":
        branch = "detached"

    origin = _git(root, "remote", "get-url", "origin")
    remote_url = origin.stdout.strip() if origin.returncode == 0 else ""

    if remote_url and branch != "detached":
        contains = _git(root, "branch", "-r", "--contains", "HEAD")
        pushed = contains.returncode == 0 and bool(contains.stdout.strip())
        if not pushed:
            if auto_push:
                push = _git(root, "push", "origin", branch)
                if push.returncode != 0:
                    raise RepoError(
                        f"`git push origin {branch}` failed:\n{push.stderr.strip()}"
                    )
            else:
                raise RepoError(
                    f"HEAD ({sha[:12]}) is not on any remote branch of origin — "
                    f"run `git push origin {branch}` first or pass --push"
                )

    return RepoRef(
        remote_url=remote_url,
        sha=sha,
        branch=branch,
        slug=repo_slug(remote_url or None, root),
        dirty=dirty,
        local_root=str(root),
    )


def local_root_of(ref: RepoRef) -> Path:
    """Local repo root a submit should stage from: the path captured at
    capture_repo_state time, falling back to cwd discovery for refs built
    without one (old records, hand-rolled specs)."""
    if ref.local_root:
        return Path(ref.local_root)
    return find_repo_root()


def env_file(root: Path) -> Path | None:
    """<root>/.env if it exists and is NOT tracked by git — the uncommitted
    secrets file we ship out-of-band. A committed .env is already in the sha, so
    we leave it alone (returning None) rather than double-shipping it."""
    p = root / ".env"
    if not p.is_file():
        return None
    tracked = _git(root, "ls-files", "--error-unmatch", ".env")
    return None if tracked.returncode == 0 else p


def worker_clone_url(remote_url: str) -> str | None:
    """Anonymous https URL a credential-less worker can clone `remote_url` from.

    The configured origin may be scp-style ssh (`git@host:o/r.git`), `ssh://`,
    `git://`, or already http(s) — normalize all to `https://<host>/<path>` so a
    worker with no ssh key or stored credentials can still fetch a *public* repo.
    Returns None for local-only / undecipherable remotes (caller ships a bundle).
    """
    u = remote_url.strip()
    if not u:
        return None
    if u.startswith(("https://", "http://")):
        return u
    for scheme in ("ssh://", "git://"):
        if u.startswith(scheme):
            rest = u[len(scheme) :].split("@", 1)[-1]  # drop any user@
            host, _, path = rest.partition("/")
            return f"https://{host}/{path}" if host and path else None
    if m := _SCP_URL.match(u):
        return f"https://{m['host']}/{m['path']}"
    return None


def remote_is_public(remote_url: str) -> bool:
    """Whether `remote_url` is anonymously cloneable (i.e. public).

    Tries `gh` for GitHub (fast, definitive), else an unauthenticated smart-http
    advertisement probe with `curl` (host-agnostic: a public repo answers 200,
    a private one 401/404). Any failure or uncertainty → False, so the caller
    falls back to shipping a bundle."""
    https = worker_clone_url(remote_url)
    if https is None:
        return False
    parts = https.split("/", 3)
    if len(parts) < 4:
        return False
    host, path = parts[2], parts[3].removesuffix(".git")
    if host.endswith("github.com") and shutil.which("gh"):
        r = subprocess.run(
            ["gh", "repo", "view", path, "--json", "visibility", "-q", ".visibility"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if r.returncode == 0:  # a gh failure (unauthed/not-found) falls through
            return r.stdout.strip().upper() == "PUBLIC"
    if shutil.which("curl"):
        probe = f"{https}/info/refs?service=git-upload-pack"
        r = subprocess.run(
            ["curl", "-fsSL", "--no-netrc", "-o", os.devnull, probe],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return r.returncode == 0
    return False


def remote_clone_plan(ref: RepoRef, root: Path) -> str | None:
    """The anonymous https url a worker should clone to materialize `ref.sha`,
    or None when the code must instead ride along as a bundle.

    Returns a url only when all three hold, so a credential-less worker clone
    cannot then fail to find the commit: (1) there's a real origin and the sha
    is a normal pushed branch commit (not a `--dirty` wip or detached HEAD);
    (2) that origin is anonymously public; (3) `ref.sha` is provably reachable
    from the current remote branch tip (asked over the wire, ancestry checked
    against our local objects). Otherwise None → bundle."""
    if not ref.remote_url or ref.dirty or ref.branch == "detached":
        return None
    url = worker_clone_url(ref.remote_url)
    if url is None or not remote_is_public(ref.remote_url):
        return None
    ls = subprocess.run(
        ["git", "-C", str(root), "ls-remote", url, ref.branch],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if ls.returncode != 0 or not ls.stdout.strip():
        return None
    tip = ls.stdout.split()[0]
    anc = _git(root, "merge-base", "--is-ancestor", ref.sha, tip)
    return url if anc.returncode == 0 else None


def create_bundle(root: Path, sha: str, dest: Path) -> Path:
    """`git bundle` carrying `sha` (and its history), for backends where the
    client cannot push directly (Kaggle datasets, Colab uploads).

    Bundles can only record refs, so the sha is pinned under a temporary
    branch ref — it must live in refs/heads/ because the bootstrap's
    `git clone --bare <bundle>` copies only branch heads (a plain
    refs/omnirun/... ref would produce an empty clone)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    ref = f"refs/heads/omnirun/bundle-{sha[:12]}"
    r = _git(root, "update-ref", ref, sha)
    if r.returncode != 0:
        raise RepoError(
            f"cannot bundle {sha[:12]}: not a commit in {root} ({r.stderr.strip()})"
        )
    try:
        b = _git(root, "bundle", "create", str(dest), ref)
        if b.returncode != 0:
            raise RepoError(f"git bundle create failed:\n{b.stderr.strip()}")
    finally:
        _git(root, "update-ref", "-d", ref)
    return dest

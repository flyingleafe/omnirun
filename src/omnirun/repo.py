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
from omnirun.progress import report

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


def current_project_slug(start: Path | None = None) -> str | None:
    """The project slug of the repo enclosing *start* (default: cwd), or ``None``.

    Derives the slug EXACTLY as ``capture_repo_state`` does — ``repo_slug`` over
    the origin remote url (or the repo dir name when there is no origin) — so the
    two never disagree. Cheap (one ``rev-parse`` + one ``remote get-url``) and
    never raises: outside a git repo, or on any git error, it returns ``None``.
    """
    start = start or Path.cwd()
    try:
        r = _git(start, "rev-parse", "--show-toplevel")
        if r.returncode != 0:
            return None
        root = Path(r.stdout.strip())
        origin = _git(root, "remote", "get-url", "origin")
        remote_url = origin.stdout.strip() if origin.returncode == 0 else ""
        return repo_slug(remote_url or None, root)
    except (OSError, subprocess.SubprocessError):
        # git binary missing / cwd gone / subprocess timeout — scoping degrades
        # to "all projects", never a crash.
        return None


def capture_repo_state(root: Path, *, auto_push: bool = False) -> RepoRef:
    """Snapshot the repo state a job will run against, enforcing the submit-time
    invariant: a clean working tree (CODE-3). A dirty tree is always refused —
    a job must run a real, reproducible revision, not an on-disk snapshot.

    A committed-but-UNPUSHED HEAD is allowed: code-plan resolution delivers it
    as a thin delta bundle over the best origin-reachable base (CODE-2c), so
    fast iteration never routes around code capture. ``auto_push=True``
    (``--push``) still pushes the branch first, keeping origin authoritative."""
    st = _git(root, "status", "--porcelain")
    if st.returncode != 0:
        raise RepoError(f"not a git repository: {root} ({st.stderr.strip()})")
    lines = [ln for ln in st.stdout.splitlines() if ln.strip()]
    if lines:
        untracked = [ln for ln in lines if ln.startswith("??")]
        tracked = [ln for ln in lines if not ln.startswith("??")]
        parts = []
        if tracked:
            parts.append(f"{len(tracked)} modified/staged file(s)")
        if untracked:
            parts.append(f"{len(untracked)} untracked file(s)")
        raise RepoError(
            f"working tree at {root} is dirty ({', '.join(parts)}) — commit (or "
            "stash) your changes so the job runs a real, reproducible revision"
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

    if remote_url and branch != "detached" and auto_push:
        contains = _git(root, "branch", "-r", "--contains", "HEAD")
        pushed = contains.returncode == 0 and bool(contains.stdout.strip())
        if not pushed:
            report(f"pushing {branch} to origin…")
            push = _git(root, "push", "origin", branch)
            if push.returncode != 0:
                raise RepoError(
                    f"`git push origin {branch}` failed:\n{push.stderr.strip()}"
                )

    return RepoRef(
        remote_url=remote_url,
        sha=sha,
        branch=branch,
        slug=repo_slug(remote_url or None, root),
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
    is a normal pushed branch commit (not a detached HEAD); (2) that origin is
    anonymously public; (3) `ref.sha` is provably reachable from the current
    remote branch tip (asked over the wire, ancestry checked against our local
    objects). Otherwise None → bundle."""
    if not ref.remote_url or ref.branch == "detached":
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


# --------------------------------------------------------------------------- deploy keys
#
# Workers always clone from origin (DESIGN, relaxed invariant #3): public repos
# anonymously, private repos with a per-origin read-only DEPLOY KEY. These are the
# client-side primitives — key generation and the GitHub `gh` provisioning path.
# The client orchestrates them (it holds the store, via the Client) in
# ``omnirun.deploykey``; here we only shell out to `ssh-keygen` and `gh`.


def ssh_clone_url(remote_url: str) -> str | None:
    """The ssh clone url a worker uses with a deploy key: ``git@host:owner/repo``.

    Already-ssh scp-style (``git@host:owner/repo.git``) and ``ssh://`` urls pass
    through (normalized to scp-style); an http(s) url is rewritten to scp-style so
    a deploy key (an ssh key) can authenticate. Returns None for undecipherable
    or local-only remotes."""
    u = remote_url.strip()
    if not u:
        return None
    if u.startswith(("https://", "http://")):
        rest = u.split("://", 1)[1]
        host, _, path = rest.partition("/")
        return f"git@{host}:{path}" if host and path else None
    if u.startswith("ssh://"):
        rest = u[len("ssh://") :].split("@", 1)[-1]
        host, _, path = rest.partition("/")
        return f"git@{host}:{path}" if host and path else None
    if _SCP_URL.match(u):
        return u  # already git@host:owner/repo(.git)
    return None


def github_slug(remote_url: str) -> str | None:
    """``owner/repo`` for a github.com remote (any url form), else None."""
    https = worker_clone_url(remote_url)
    if https is None:
        return None
    parts = https.split("/", 3)
    if len(parts) < 4 or not parts[2].endswith("github.com"):
        return None
    return parts[3].removesuffix(".git")


def generate_deploy_keypair(comment: str = "omnirun-deploy") -> tuple[str, str]:
    """Generate an ed25519 keypair with ``ssh-keygen``; return (private, public).

    Both are text (OpenSSH private PEM + the single-line public key). No passphrase
    — the key is read-only and scoped to one repo."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        key = Path(d) / "id_ed25519"
        r = subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-C", comment, "-f", str(key)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            raise RepoError(f"ssh-keygen failed:\n{r.stderr.strip()}")
        return key.read_text(), (key.with_suffix(".pub")).read_text().strip()


def _gh(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gh", *args], capture_output=True, text=True, timeout=timeout
    )


def gh_available() -> bool:
    return shutil.which("gh") is not None


def gh_can_admin(owner_repo: str) -> bool:
    """Whether the authenticated `gh` user can add a deploy key to *owner_repo*
    (i.e. has admin permission). False on any gh/auth/lookup failure."""
    if not gh_available():
        return False
    r = _gh("api", f"repos/{owner_repo}", "--jq", ".permissions.admin")
    return r.returncode == 0 and r.stdout.strip() == "true"


def gh_create_deploy_key(owner_repo: str, public_key: str, title: str) -> str:
    """Register *public_key* as a READ-ONLY deploy key on *owner_repo* via `gh`;
    return the created key's id (as a string). Raises RepoError on failure."""
    r = _gh(
        "api",
        "-X",
        "POST",
        f"repos/{owner_repo}/keys",
        "-f",
        f"title={title}",
        "-f",
        f"key={public_key}",
        "-F",
        "read_only=true",
        "--jq",
        ".id",
    )
    if r.returncode != 0:
        raise RepoError(
            f"creating a deploy key on {owner_repo} via gh failed:\n{r.stderr.strip()}"
        )
    return r.stdout.strip()


def gh_delete_deploy_key(owner_repo: str, key_id: str) -> None:
    """Best-effort delete of a deploy key by id (for `omnirun deploy-key rm`)."""
    _gh("api", "-X", "DELETE", f"repos/{owner_repo}/keys/{key_id}")


# A thin bundle rides the spec as base64 JSON through the store and the wire;
# the guard keeps unpushed deltas honest (push your branch for bigger changes).
THIN_BUNDLE_MAX_BYTES = 16 * 1024 * 1024


def sha_on_origin(root: Path, sha: str) -> bool:
    """Whether *sha* is reachable from any remote-tracking branch — the same
    local knowledge the old pushed-check used (no network round-trip)."""
    r = _git(root, "branch", "-r", "--contains", sha)
    return r.returncode == 0 and bool(r.stdout.strip())


def create_thin_bundle(root: Path, sha: str, dest: Path) -> Path:
    """A DELTA ``git bundle`` for a committed-but-unpushed *sha* (CODE-2c).

    The bundle carries only the commits between the best origin-reachable
    bases — every remote-tracking ref is excluded (`^refs/remotes/...`), so
    the boundary is the merge-base with the remote tips — and pins *sha*
    under a temporary branch ref (branch heads survive `git clone --bare`).
    The worker first clones/fetches origin (providing the prerequisites),
    then fetches this bundle on top. Size-guarded: a delta over
    ``THIN_BUNDLE_MAX_BYTES`` is refused with a push-your-branch hint."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    ref = f"refs/heads/omnirun/bundle-{sha[:12]}"
    r = _git(root, "update-ref", ref, sha)
    if r.returncode != 0:
        raise RepoError(
            f"cannot bundle {sha[:12]}: not a commit in {root} ({r.stderr.strip()})"
        )
    remotes = _git(root, "for-each-ref", "--format=%(refname)", "refs/remotes")
    negatives = [
        line.strip()
        for line in remotes.stdout.splitlines()
        if line.strip() and not line.strip().endswith("/HEAD")
    ]
    try:
        b = _git(
            root, "bundle", "create", str(dest), ref, *[f"^{n}" for n in negatives]
        )
        if b.returncode != 0:
            raise RepoError(f"git bundle create failed:\n{b.stderr.strip()}")
    finally:
        _git(root, "update-ref", "-d", ref)
    size = dest.stat().st_size
    if size > THIN_BUNDLE_MAX_BYTES:
        dest.unlink(missing_ok=True)
        raise RepoError(
            f"the unpushed delta for {sha[:12]} bundles to {size / 1e6:.1f} MB "
            f"(cap {THIN_BUNDLE_MAX_BYTES / 1e6:.0f} MB) — push your branch "
            "(git push, or submit with --push) instead"
        )
    return dest


def thin_bundle_b64(root: Path, sha: str) -> str:
    """The base64 payload of :func:`create_thin_bundle` (rides ``CodePlan``)."""
    import base64
    import tempfile

    with tempfile.TemporaryDirectory(prefix="omnirun-bundle-") as d:
        bundle = create_thin_bundle(root, sha, Path(d) / "thin.bundle")
        return base64.b64encode(bundle.read_bytes()).decode("ascii")


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

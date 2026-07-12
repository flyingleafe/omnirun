"""Shared machinery for GPU-marketplace backends (RunPod, Vast.ai, Thunder Compute).

Lifecycle: probe = price/availability query (never raises; errors become a single
unfit offer). submit = create instance via provider REST API -> poll until it has
an ssh endpoint and reports running -> wait for sshd to actually accept -> stage
the job (client-side git push + bootstrap.sh) -> detached launch (setsid+nohup,
pidfile), exactly like the plain ssh backend.

Termination policy (billable resources must not leak):
- **Provisioning stub** (issue #7): the instant ``submit`` rents an instance —
  before the minutes-long provisioning wait — it calls the ``on_provisioning``
  hook with a partial handle carrying the instance id, so the client persists a
  recovery record *first*. A submit killed mid-wait then stays visible to
  ``ps``/``status`` (reported as PROVISIONING with a billing warning) and
  reclaimable by ``omnirun gc --all``, instead of orphaning a running instance
  with no local trace. The final handle overwrites the stub on success.
- **Client-side auto-terminate** (``auto_terminate``, default true): the instance
  is destroyed in ``pull_outputs`` (after a successful pull), in ``cancel``, and
  in ``gc``. ``status`` never terminates — when the job is terminal it appends a
  reminder that the instance is still billing.  The worker itself cannot safely
  call the provider API (that would require shipping the API key to a rented box).
- **On-instance idle failsafe** (``idle_failsafe``, default true): a second
  detached watcher waits for ``result.json``, sleeps ``failsafe_grace_s``
  (default 24h, so you have time to pull outputs), then runs ``shutdown -h now``.
  Honesty note — *shutdown is not terminate*:

  * RunPod: the pod lands in EXITED; GPU billing stops but container/volume disk
    keeps billing until the pod is terminated (DELETE).
  * Vast.ai: the instance shows ``exited``; **storage still bills until the
    instance is destroyed** (DELETE), stop is not enough.
  * Thunder: billing is per-minute and only while running, so a stopped instance
    stops compute charges.

  The failsafe therefore caps runaway GPU burn if the client disappears, but
  ``omnirun gc`` is still the thing that ends all billing.

SSH keys: providers differ. RunPod and Vast use **account-level** public keys
(register once in their console/API — ``check()`` reminds you); Thunder takes the
public key in the create call. The key read here is ``config.extra
("ssh_public_key")`` or ``~/.ssh/id_ed25519.pub``; the matching private key (same
path minus ``.pub``) is passed to ssh as the identity when it exists.

Backend config knobs (``[backends.<name>]`` extras): ``image``,
``provision_timeout_s`` (600), ``ssh_wait_timeout_s`` (120), ``auto_terminate``
(true), ``idle_failsafe`` (true), ``failsafe_grace_s`` (86400),
``ssh_public_key``, plus provider-specific ones documented per module.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from omnirun.backends import jobdir
from omnirun.backends.base import Backend, BackendError, ProvisioningSink
from omnirun.bootstrap import BootstrapParams
from omnirun.execlayer.base import Exec, shell_quote
from omnirun.repo import local_root_of
from omnirun.models import (
    CancelMode,
    JobHandle,
    JobSpec,
    JobStatus,
    Offer,
    ResourceSpec,
    StatusReport,
)

try:  # built on a parallel track; absent only during early development
    from omnirun.execlayer.ssh import SSHExec
except ImportError:  # pragma: no cover
    SSHExec = None  # type: ignore[assignment]

HTTP_TIMEOUT_S = 15.0
PROVISION_POLL_S = 10.0
SSH_WAIT_POLL_S = 5.0
DEFAULT_PROVISION_TIMEOUT_S = 600.0
DEFAULT_SSH_WAIT_TIMEOUT_S = 120.0
DEFAULT_FAILSAFE_GRACE_S = 24 * 3600
WAIT_ESTIMATE_S = 150.0  # honest ballpark: instance provisioning + image pull
WAIT_NOTE = "instance provisioning + image pull, typically 2-3 min"

_sleep = time.sleep  # test seam


class HTTPBackendError(BackendError):
    """BackendError carrying the HTTP status code for callers that branch on it."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class Instance(BaseModel):
    """Provider-neutral view of one rented machine."""

    provider: str
    instance_id: str
    ssh_target: str | None = None  # host/IP (bare, or "user@host")
    ssh_port: int | None = None
    status: str = ""  # provider status, lowercased ("running" == ready)
    cost_per_hour: float | None = None
    gpu_type: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


def spec_matches_gpu(
    res: ResourceSpec, gpu_type: str | None, vram_gb: float | None
) -> bool:
    """Does a provider GPU (normalized name + per-GPU VRAM) satisfy the spec?"""
    if res.gpu_type is not None:
        return gpu_type == res.gpu_type
    floor = res.vram_floor_gb()
    if floor is not None:
        return vram_gb is not None and vram_gb >= floor
    return True


class MarketplaceBackend(Backend, ABC):
    """ABC implementing the provision -> ssh -> run -> terminate lifecycle.

    Subclasses implement the five provider primitives plus ``_check_api``.
    """

    #: env var holding the API key when config.api_key_env is unset
    default_key_env: str = ""
    #: short provider tag stored in Instance.provider
    provider: str = ""

    # ---- provider primitives (subclass surface) -------------------------------

    @abstractmethod
    def _query_offers(self, res: ResourceSpec) -> list[Offer]:
        """Live price/availability query. May raise; probe() wraps it."""

    @abstractmethod
    def _create_instance(self, spec: JobSpec, offer: Offer) -> Instance:
        """Rent the machine described by offer.details. Returns the new instance."""

    @abstractmethod
    def _get_instance(self, instance_id: str) -> Instance | None:
        """Current state, or None if the instance no longer exists."""

    @abstractmethod
    def _terminate(self, instance_id: str) -> None:
        """Destroy the instance (the variant that ends ALL billing)."""

    @abstractmethod
    def _default_ssh_user(self) -> str: ...

    @abstractmethod
    def _check_api(self) -> str:
        """Cheap authenticated call; return a short human summary."""

    # ---- API key / HTTP helpers ------------------------------------------------

    def _key_env(self) -> str:
        return self.config.api_key_env or self.default_key_env

    def _api_key(self) -> str | None:
        return os.environ.get(self._key_env()) or None

    def _require_key(self) -> str:
        key = self._api_key()
        if not key:
            raise BackendError(f"{self.name}: set {self._key_env()} (API key)")
        return key

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        auth: bool = True,
    ) -> httpx.Response:
        hdrs = dict(headers or {})
        if auth:
            hdrs["Authorization"] = f"Bearer {self._require_key()}"
        try:
            resp = httpx.request(
                method, url, json=json_body, headers=hdrs, timeout=HTTP_TIMEOUT_S
            )
        except httpx.HTTPError as e:
            raise BackendError(f"{self.name}: {method} {url} failed: {e}") from e
        if resp.status_code >= 400:
            raise HTTPBackendError(
                f"{self.name}: {method} {url} -> HTTP {resp.status_code}: "
                f"{resp.text[:300]}",
                status_code=resp.status_code,
            )
        return resp

    # ---- ssh key handling --------------------------------------------------------

    def _public_key_path(self) -> Path:
        p = self.config.extra("ssh_public_key")
        return Path(p).expanduser() if p else Path.home() / ".ssh" / "id_ed25519.pub"

    def _read_public_key(self) -> str:
        p = self._public_key_path()
        if not p.exists():
            raise BackendError(
                f"{self.name}: ssh public key {p} not found "
                f"(set backends.{self.name}.ssh_public_key in config)"
            )
        return p.read_text().strip()

    def _identity_file(self) -> str | None:
        pub = self._public_key_path()
        if pub.suffix == ".pub":
            priv = pub.with_suffix("")
            if priv.exists():
                return str(priv)
        return None

    def _make_exec(self, target: str, port: int | None) -> Exec:
        cls = SSHExec
        if cls is None:  # pragma: no cover
            raise BackendError("ssh exec layer missing (omnirun.execlayer.ssh)")
        # Freshly provisioned hosts are never in known_hosts; accept-new keeps
        # background polling non-interactive without disabling checking entirely.
        # Attached `-oKEY=VALUE` (one token) — see SSHExec._control_opts: a lone
        # `-o` before the host breaks PATH ssh-wrappers that scan argv for it.
        return cls(
            target,
            port=port,
            identity=self._identity_file(),
            extra_opts=["-oStrictHostKeyChecking=accept-new"],
        )

    def _exec_from_handle(self, handle: JobHandle) -> Exec:
        return self._make_exec(handle.data["ssh_target"], handle.data.get("ssh_port"))

    # ---- Backend protocol ----------------------------------------------------------

    def probe(self, res: ResourceSpec) -> list[Offer]:
        if not self._api_key():
            return [self._unfit_offer(f"set {self._key_env()} to enable {self.name}")]
        if not res.wants_gpu():
            return [
                self._unfit_offer(
                    "no GPU requested — marketplace backends only provision GPU instances"
                )
            ]
        try:
            offers = self._query_offers(res)
        except Exception as e:  # probe must never raise
            return [self._unfit_offer(f"probe failed: {e}")]
        cutoff = self.config.max_hourly
        for o in offers:
            if o.wait_estimate_s is None:
                o.wait_estimate_s = WAIT_ESTIMATE_S
            if not o.wait_note:
                o.wait_note = WAIT_NOTE
            if (
                cutoff is not None
                and o.cost_per_hour is not None
                and o.cost_per_hour > cutoff
            ):
                o.fits = False
                o.unfit_reasons.append(
                    f"${o.cost_per_hour:.2f}/hr exceeds max_hourly ${cutoff:.2f}"
                )
        return offers

    def _unfit_offer(self, reason: str) -> Offer:
        return Offer(
            backend=self.name,
            label=f"{self.name}: unavailable",
            fits=False,
            unfit_reasons=[reason],
        )

    def submit(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        inst = self._create_instance(spec, offer)
        instance_id = inst.instance_id
        # The instance is now billing but the handle isn't ready yet. Hand the
        # client a stub carrying the instance id *before* the provisioning wait,
        # so an interrupted submit (a kill mid-wait) stays visible to ps/gc
        # instead of orphaning a running instance with no local record (#7).
        if on_provisioning is not None:
            on_provisioning(
                JobHandle(
                    backend=self.name,
                    job_id=spec.job_id,
                    data={"instance_id": instance_id, "provisioning": True},
                )
            )
        try:
            inst = self._wait_provisioned(instance_id)
            assert inst.ssh_target is not None
            user = self._default_ssh_user()
            target = (
                inst.ssh_target
                if "@" in inst.ssh_target
                else f"{user}@{inst.ssh_target}"
            )
            ex = self._make_exec(target, inst.ssh_port)
            self._wait_ssh(ex)
            root = jobdir.remote_root(ex, self.config.root)
            project_root = jobdir.resolve_project_root(
                ex, root, spec.repo.slug, self.config.project_root_for(spec.repo.slug)
            )
            params = BootstrapParams(
                omnirun_root=root,
                project_root=project_root,
                setup_lines=list(self.config.env_setup),
            )
            job_dir = jobdir.stage_job(ex, spec, local_root_of(spec.repo), params, root)
            self._launch_detached(ex, job_dir)
        except BaseException as e:
            try:  # never leak a billable instance behind a failed submit
                self._terminate(instance_id)
            except Exception:
                pass
            if isinstance(e, BackendError) or not isinstance(e, Exception):
                raise
            raise BackendError(
                f"{self.name}: submit failed after creating instance "
                f"{instance_id} (instance terminated): {e}"
            ) from e
        return JobHandle(
            backend=self.name,
            job_id=spec.job_id,
            data={
                "ssh_target": target,
                "ssh_port": inst.ssh_port,
                "instance_id": instance_id,
                "job_dir": job_dir,
                "root": root,
                "slug": spec.repo.slug,
            },
        )

    def _wait_provisioned(self, instance_id: str) -> Instance:
        timeout = float(
            self.config.extra("provision_timeout_s", DEFAULT_PROVISION_TIMEOUT_S)
        )
        deadline = time.monotonic() + timeout
        last: Instance | None = None
        while True:
            inst = self._get_instance(instance_id)
            if inst is not None:
                last = inst
                if inst.ssh_target and inst.status.lower() == "running":
                    return inst
            if time.monotonic() >= deadline:
                raise BackendError(
                    f"{self.name}: instance {instance_id} not ready after "
                    f"{timeout:.0f}s (last status: "
                    f"{last.status if last else 'not visible yet'})"
                )
            _sleep(PROVISION_POLL_S)

    def _wait_ssh(self, ex: Exec) -> None:
        timeout = float(
            self.config.extra("ssh_wait_timeout_s", DEFAULT_SSH_WAIT_TIMEOUT_S)
        )
        deadline = time.monotonic() + timeout
        while True:
            try:
                if ex.run("true", timeout=15).ok:
                    return
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise BackendError(
                    f"{self.name}: sshd on {ex.describe()} did not accept "
                    f"connections within {timeout:.0f}s"
                )
            _sleep(SSH_WAIT_POLL_S)

    def _launch_detached(self, ex: Exec, job_dir: str) -> None:
        q = shell_quote(job_dir)
        ex.run(
            f"cd {q} && {{ setsid nohup bash bootstrap.sh </dev/null "
            f">/dev/null 2>&1 & echo $! > pid; }}",
            check=True,
        )
        if self.config.extra("idle_failsafe", True):
            grace = int(self.config.extra("failsafe_grace_s", DEFAULT_FAILSAFE_GRACE_S))
            watcher = (
                f"while [ ! -f {q}/result.json ]; do sleep 300; done; "
                f"sleep {grace}; shutdown -h now"
            )
            # Failsafe only: caps GPU burn if the client never comes back.
            # shutdown != terminate — see module docstring.
            ex.run(
                f"setsid nohup bash -c {shell_quote(watcher)} "
                f"</dev/null >/dev/null 2>&1 &"
            )

    def status(self, handle: JobHandle) -> StatusReport:
        instance_id = handle.data["instance_id"]
        if not handle.data.get("job_dir"):
            # A provisioning stub: submit was interrupted before it staged the
            # job, so only the instance id was persisted. Report whether the
            # rented instance is still billing so the user knows to reclaim it.
            try:
                inst = self._get_instance(instance_id)
            except BackendError as e:
                return StatusReport(
                    status=JobStatus.PROVISIONING,
                    detail=f"submit was interrupted while provisioning instance "
                    f"{instance_id}; could not reach the {self.name} API to check "
                    f"it ({e}) — verify in the provider console, run `omnirun gc`",
                )
            if inst is None:
                return StatusReport(
                    status=JobStatus.LOST,
                    detail=f"instance {instance_id} never finished provisioning and "
                    "no longer exists",
                )
            return StatusReport(
                status=JobStatus.PROVISIONING,
                detail=f"submit was interrupted after renting instance {instance_id} "
                f"(now {inst.status or 'unknown'}) — it is still billing; run "
                f"`omnirun gc --all` (or cancel {handle.job_id}) to destroy it",
            )
        inst: Instance | None = None
        api_error: str | None = None
        try:
            inst = self._get_instance(instance_id)
        except BackendError as e:
            api_error = str(e)
        if inst is None and api_error is None:
            return StatusReport(
                status=JobStatus.LOST,
                detail=f"instance {instance_id} no longer exists "
                "(auto-terminated after pull, gc'd, or reaped by the provider)",
            )
        report = jobdir.derive_status(
            self._exec_from_handle(handle), handle.data["job_dir"]
        )
        if (
            report.status.terminal
            and inst is not None
            and self.config.extra("auto_terminate", True)
        ):
            note = (
                f"instance {instance_id} still running — pull outputs to "
                "auto-terminate, or run `omnirun gc`"
            )
            report.detail = f"{report.detail}; {note}" if report.detail else note
        return report

    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        ex = self._exec_from_handle(handle)
        job_dir = handle.data["job_dir"]
        is_terminal: Callable[[], bool] | None = None
        if follow:

            def _terminal() -> bool:
                return jobdir.derive_status(ex, job_dir).status.terminal

            is_terminal = _terminal

        return jobdir.tail_logs(ex, job_dir, follow=follow, is_terminal=is_terminal)

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        ex = self._exec_from_handle(handle)
        paths = jobdir.pull_outputs(ex, handle.data["job_dir"], dest)
        if self.config.extra("auto_terminate", True):
            instance_id = handle.data["instance_id"]
            try:
                if self._get_instance(instance_id) is not None:
                    self._terminate(instance_id)
            except BackendError as e:
                raise BackendError(
                    f"{self.name}: outputs pulled to {dest}, but auto-terminate of "
                    f"instance {instance_id} failed — it is still billing, run "
                    f"`omnirun gc`. ({e})"
                ) from e
        return paths

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        if handle.data.get("job_dir"):  # a provisioning stub has nothing to kill
            try:  # best-effort remote kill; instance dies right after anyway
                ex = self._exec_from_handle(handle)
                q = shell_quote(handle.data["job_dir"])
                ex.run(
                    f"if [ -f {q}/pid ]; then p=$(cat {q}/pid); "
                    f'kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null; fi; true',
                    timeout=30,
                )
            except Exception:
                pass
        instance_id = handle.data["instance_id"]
        if self._get_instance(instance_id) is not None:
            self._terminate(instance_id)

    def gc(self, handle: JobHandle) -> None:
        instance_id = handle.data.get("instance_id")
        if instance_id and self._get_instance(instance_id) is not None:
            self._terminate(instance_id)

    def check(self) -> str:
        if not self._api_key():
            raise BackendError(
                f"{self.name}: set {self._key_env()} (API key from the "
                f"{self.provider or self.type_name} console)"
            )
        return f"ok: {self._check_api()}"

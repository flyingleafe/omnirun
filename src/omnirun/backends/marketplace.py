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

import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field

from omnirun.backends import jobdir
from omnirun.backends.base import (
    Backend,
    BackendError,
    BackendUnreachable,
    CapacityError,
    OfferGoneError,
    ProvisioningSink,
)
from omnirun.bootstrap import BootstrapParams
from omnirun.execlayer.base import Exec, ExecError, shell_quote
from omnirun.repo import local_root_of
from omnirun.models import (
    CancelMode,
    JobHandle,
    JobSpec,
    JobStatus,
    Offer,
    ReapPolicy,
    ResourceSpec,
    StatusReport,
)

if TYPE_CHECKING:
    from omnirun.config import BackendConfig
    from omnirun.execlayer.ssh import SSHExec as _SSHExecClass

SSHExec: type[_SSHExecClass] | None
try:  # built on a parallel track; absent only during early development
    from omnirun.execlayer.ssh import SSHExec
except ImportError:  # pragma: no cover
    SSHExec = None

HTTP_TIMEOUT_S = 15.0
PROVISION_POLL_S = 10.0
SSH_WAIT_POLL_S = 5.0
DEFAULT_PROVISION_TIMEOUT_S = 600.0
DEFAULT_SSH_WAIT_TIMEOUT_S = 120.0
DEFAULT_FAILSAFE_GRACE_S = 24 * 3600
# How many times submit will destroy a dead-on-arrival instance and rent a fresh
# one before giving up. Marketplace instances (esp. vast) not uncommonly rent but
# never boot ssh; a bad rental is the instance's fault, not the job's, so the
# backend re-provisions rather than failing the whole placement.
DEFAULT_PROVISION_ATTEMPTS = 3
# How many times a rate-limited (HTTP 429) provider API call is retried, honoring
# the server's Retry-After, before giving up. Parallel provisioning bursts calls;
# a generous retry count keeps a placement from failing on a transient 429.
DEFAULT_API_429_RETRIES = 6
DEFAULT_RETRY_AFTER_S = 5.0
# No-progress watchdog on the staged provisioning wait (COST-4): the provider
# status string unchanged AND not ready for this long → the rental is dead;
# destroy it and fail the stage instead of billing until the overall timeout.
DEFAULT_PROVISION_STALL_S = 90.0
WAIT_ESTIMATE_S = 150.0  # honest ballpark: instance provisioning + image pull
WAIT_NOTE = "instance provisioning + image pull, typically 2-3 min"

_sleep = time.sleep  # test seam
_monotonic = time.monotonic  # test seam (the stall watchdog's clock)
_log = logging.getLogger("omnirun.backends.marketplace")


def instance_label(job_id: str) -> str:
    """The deterministic instance name/label for a job (SCHED-8 adopt key)."""
    return f"omnirun-{job_id}"


def _retry_after_s(resp: "httpx.Response") -> float:
    """Seconds to wait before retrying a 429, from the ``Retry-After`` header or a
    ``retry_after`` field in the JSON body, else a sensible default. A small jitter
    is added so parallel placement threads do not all retry on the same beat."""
    import random

    candidates: list[float] = []
    header = resp.headers.get("Retry-After")
    if header:
        try:
            candidates.append(float(header))
        except ValueError:
            pass
    try:
        body = resp.json()
        if isinstance(body, dict) and body.get("retry_after") is not None:
            candidates.append(float(body["retry_after"]))
    except (ValueError, TypeError):
        pass
    base = max(candidates) if candidates else DEFAULT_RETRY_AFTER_S
    return base + random.uniform(0.0, 1.0)


class HTTPBackendError(BackendError):
    """BackendError carrying the HTTP status code for callers that branch on it."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class InstanceUnreachable(BackendError):
    """The rented instance never became usable (never reported running, or sshd
    never accepted connections). Distinct from a job/staging failure because the
    fault is the RENTAL, not the work — so ``submit`` destroys it and re-provisions
    a fresh instance instead of failing the placement."""


class Instance(BaseModel):
    """Provider-neutral view of one rented machine."""

    provider: str
    instance_id: str
    ssh_target: str | None = None  # host/IP (bare, or "user@host")
    ssh_port: int | None = None
    status: str = ""  # provider status, lowercased ("running" == ready)
    cost_per_hour: float | None = None
    gpu_type: str | None = None
    label: str | None = None  # our deterministic name (omnirun-<job_id>), if set
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

    def __init__(self, name: str, config: "BackendConfig") -> None:
        super().__init__(name, config)
        # A finished or abandoned instance must not keep billing: with
        # auto_terminate (the default) a terminal job is collected to the durable
        # cache and then the instance is released, and a LOST placement is
        # force-released — both on the next reconcile tick. ``auto_terminate=false``
        # opts out of ALL automatic teardown (fully manual control). The existing
        # auto-terminate in pull_outputs/cancel/gc stays as an idempotent second
        # guard (the ``_get_instance`` check makes double-terminate safe).
        auto = bool(self.config.extra("auto_terminate", True))
        self.reap = ReapPolicy(hold_on_terminal=auto, release_lost=auto)
        # Provider API throttle. Parallel provisioning fans many concurrent API
        # calls at one provider; vast caps at ~3 req/s and answers a burst with
        # HTTP 429. ``api_min_interval_s`` spaces the calls to stay under the
        # ceiling; 429s that still slip through are retried with backoff below.
        # The spacing STATE lives in the shared EndpointManager keyed by
        # provider name, so the limit covers every backend section (and every
        # placement thread) talking to this provider API, not just this one.
        self._min_interval_s = float(self.config.extra("api_min_interval_s", 0.0))

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
    def _list_instances(self) -> list[Instance]:
        """Every live instance on the account (one list call — the batched
        observation and the adopt-by-label probe both ride it)."""

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
            raise BackendUnreachable(f"{self.name}: set {self._key_env()} (API key)")
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
        retries = int(self.config.extra("api_429_retries", DEFAULT_API_429_RETRIES))
        for attempt in range(retries + 1):
            self._throttle()
            try:
                resp = httpx.request(
                    method, url, json=json_body, headers=hdrs, timeout=HTTP_TIMEOUT_S
                )
            except httpx.HTTPError as e:
                raise BackendUnreachable(
                    f"{self.name}: {method} {url} failed: {e}"
                ) from e
            if resp.status_code == 429:
                if attempt < retries:
                    wait = _retry_after_s(resp)
                    _log.info(
                        "%s: %s %s rate-limited (429); retry %d/%d in %.1fs",
                        self.name,
                        method,
                        url,
                        attempt + 1,
                        retries,
                        wait,
                    )
                    _sleep(wait)
                    continue
                # Retries exhausted: the provider is saturated RIGHT NOW — a
                # transient capacity condition, not a defect. The scheduler
                # defers quietly and retries later (JOB-4).
                raise CapacityError(
                    f"{self.name}: {method} {url} still rate-limited (429) after "
                    f"{retries} retries — provider API saturated, retry later"
                )
            if resp.status_code >= 400:
                raise HTTPBackendError(
                    f"{self.name}: {method} {url} -> HTTP {resp.status_code}: "
                    f"{resp.text[:300]}",
                    status_code=resp.status_code,
                )
            return resp
        raise CapacityError(  # pragma: no cover - loop always returns/raises
            f"{self.name}: {method} {url} exhausted 429 retries"
        )

    def _throttle(self) -> None:
        """Space provider-API calls to ``api_min_interval_s`` so a burst of
        parallel provisioning never trips the provider's rate limit. A no-op
        when the interval is 0 (the default). The last-call state is the
        SHARED per-provider :class:`~omnirun.endpoints.manager.Throttle`, so
        the ceiling holds across all backend sections on this provider."""
        if self._min_interval_s <= 0:
            return
        self.endpoint_manager().throttle(self.provider or self.type_name).wait(
            self._min_interval_s
        )

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
        # A provisioning stub (interrupted submit, or ssh details never
        # materialized) carries no ssh_target. Fail with an actionable message
        # instead of leaking a raw KeyError('ssh_target') to `logs`/`pull` (#24).
        target = handle.data.get("ssh_target")
        if not target:
            instance_id = handle.data.get("instance_id", "?")
            raise BackendError(
                f"{self.name}: instance {instance_id} has no ssh yet — it is still "
                "provisioning or never came up (check `omnirun status`; if the "
                f"account's ssh key isn't registered on {self.name}, ssh never "
                f"materializes). Cancel {handle.job_id} or `omnirun gc` to reclaim it."
            )
        return self._make_exec(target, handle.data.get("ssh_port"))

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
        attempts = max(
            1, int(self.config.extra("provision_attempts", DEFAULT_PROVISION_ATTEMPTS))
        )
        current = offer
        tried: set[str] = set()
        last: BackendError | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self._submit_once(spec, current, on_provisioning)
            except (InstanceUnreachable, OfferGoneError) as e:
                # The RENTAL was dead-on-arrival (never ran / no sshd) or the offer
                # churned before we could rent it; the failed instance is already
                # terminated. Re-PROBE for a fresh cheapest offer (re-renting the
                # same, now-stale ask just fails again) and try that — unless we've
                # exhausted our attempts or nothing else is on the market.
                last = e
                tried.add(self._offer_key(current))
                if attempt >= attempts:
                    break
                fresh = self._reprobe_offer(spec, tried)
                if fresh is None:
                    break
                _log.warning(
                    "%s: %s — re-provisioning on a fresh offer %s (attempt %d/%d)",
                    self.name,
                    e,
                    self._offer_key(fresh),
                    attempt + 1,
                    attempts,
                )
                current = fresh
        raise BackendError(
            f"{self.name}: could not get a usable instance after {attempts} "
            f"attempts (last: {last})"
        )

    # ---- staged placement seam (v2 engine) ------------------------------------
    # rent = create-or-adopt the instance ONLY; boot = the provisioning + sshd
    # wait (per-stage budgets + the COST-4 stall watchdog); launch = stage the
    # job dir and start the bootstrap detached. Re-shopping a churned ask
    # belongs to the CALLER (the engine re-shops on capacity contention) — the
    # staged path never loops internally.

    def find_resource(self, spec: JobSpec) -> JobHandle | None:
        """Adopt by the deterministic instance name/label ``omnirun-<job_id>``
        (SCHED-8): an account instance carrying our label is OUR rental from a
        prior placer — adopt it, never rent a duplicate."""
        label = instance_label(spec.job_id)
        for inst in self._list_instances():
            if inst.label == label:
                return self._instance_handle(spec.job_id, inst)
        return None

    def _instance_handle(self, job_id: str, inst: Instance) -> JobHandle:
        """The handle for an adopted instance: ssh coordinates when it already
        has them, else a provisioning stub the boot stage completes."""
        data: dict[str, Any] = {"instance_id": inst.instance_id}
        if inst.ssh_target and inst.status.lower() == "running":
            user = self._default_ssh_user()
            target = (
                inst.ssh_target
                if "@" in inst.ssh_target
                else f"{user}@{inst.ssh_target}"
            )
            data.update({"ssh_target": target, "ssh_port": inst.ssh_port})
        else:
            data["provisioning"] = True
        return JobHandle(backend=self.name, job_id=job_id, data=data)

    def rent_resource(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
        *,
        attempt: int = 1,
    ) -> JobHandle:
        """Create the instance for *offer* and return a provisioning handle.

        A churned/taken ask surfaces as ``OfferGoneError`` (capacity
        contention at the seam) so the caller re-shops; nothing is retried
        here."""
        inst = self._create_instance(spec, offer)
        handle = JobHandle(
            backend=self.name,
            job_id=spec.job_id,
            data={"instance_id": inst.instance_id, "provisioning": True},
        )
        if on_provisioning is not None:
            on_provisioning(handle)
        return handle

    def resource_ready(self, handle: JobHandle) -> JobHandle:
        """Wait for the rented instance to run and accept ssh (per-stage
        budgets: ``provision_timeout_s`` then ``ssh_wait_timeout_s``), with a
        no-progress watchdog (``provision_stall_s``, COST-4). A rental that
        never becomes usable is DESTROYED first, then the failure propagates
        (an infra failure — the rental's fault, not the job's)."""
        instance_id = handle.data["instance_id"]
        try:
            inst = self._wait_provisioned_watched(instance_id)
            assert inst.ssh_target is not None
            user = self._default_ssh_user()
            target = (
                inst.ssh_target
                if "@" in inst.ssh_target
                else f"{user}@{inst.ssh_target}"
            )
            ex = self._make_exec(target, inst.ssh_port)
            self._wait_ssh(ex)
        except InstanceUnreachable:
            try:  # destroy FIRST — a dead rental must not bill (COST-4)
                self._terminate(instance_id)
            except Exception:
                _log.warning(
                    "%s: could not destroy dead-on-arrival instance %s; "
                    "the release/reap will retry",
                    self.name,
                    instance_id,
                )
            raise
        data = dict(handle.data)
        data.pop("provisioning", None)
        data.update({"ssh_target": target, "ssh_port": inst.ssh_port})
        return JobHandle(backend=self.name, job_id=handle.job_id, data=data)

    def launch_job(
        self, spec: JobSpec, handle: JobHandle, *, attempt: int = 1
    ) -> JobHandle:
        ex = self._exec_from_handle(handle)
        root = jobdir.remote_root(ex, self.config.root)
        job_dir = jobdir.job_dir_of(root, spec.job_id)
        # Idempotent (SCHED-8): a bootstrap already started on this instance
        # (pid recorded) is adopted, never re-executed.
        if not ex.run(f"test -f {shell_quote(f'{job_dir}/pid')}").ok:
            project_root = jobdir.resolve_project_root(
                ex, root, spec.repo.slug, self.config.project_root_for(spec.repo.slug)
            )
            params = BootstrapParams(
                omnirun_root=root,
                project_root=project_root,
                setup_lines=list(self.config.env_setup),
            )
            job_dir = jobdir.stage_job(
                ex, spec, local_root_of(spec.repo), params, root, attempt=attempt
            )
            self._launch_detached(ex, job_dir)
        data = dict(handle.data)
        data.update({"job_dir": job_dir, "root": root, "slug": spec.repo.slug})
        return JobHandle(backend=self.name, job_id=spec.job_id, data=data)

    def observe_status(self, handle: JobHandle) -> StatusReport:
        """Observer poll (COST-3): the instance list lacking our id is POSITIVE
        death evidence (LOST); an unreachable provider API or a worker that
        cannot be ssh'd while its instance still exists is UNKNOWN — raise
        ``BackendUnreachable`` and freeze rather than mis-reap a live rental."""
        instance_id = handle.data["instance_id"]
        try:
            inst = self._get_instance(instance_id)
        except BackendUnreachable:
            raise
        except BackendError as e:
            # API reachable but erroring: no information either way.
            raise BackendUnreachable(str(e)) from e
        if inst is None:
            return StatusReport(
                status=JobStatus.LOST,
                detail=f"instance {instance_id} no longer exists",
            )
        if not handle.data.get("job_dir"):
            return StatusReport(
                status=JobStatus.PROVISIONING,
                detail=f"instance {instance_id} is {inst.status or 'unknown'}; "
                "job not launched yet",
            )
        try:
            return jobdir.derive_status(
                self._exec_from_handle(handle),
                handle.data["job_dir"],
                raise_unreachable=True,
            )
        except ExecError as e:
            raise BackendUnreachable(str(e)) from e

    def observe_batch(self, handles: list[JobHandle]) -> list[StatusReport]:
        """ONE account-wide instance list per cycle (never one GET per
        instance — the v1 pathology), then a status triple per still-live
        instance over its own ssh endpoint. An id missing from the list is
        positive death evidence; an unreachable API or worker freezes."""
        if not handles:
            return []
        instances = {i.instance_id: i for i in self._list_instances()}
        reports: list[StatusReport] = []
        for h in handles:
            instance_id = str(h.data.get("instance_id"))
            inst = instances.get(instance_id)
            if inst is None:
                reports.append(
                    StatusReport(
                        status=JobStatus.LOST,
                        detail=f"instance {instance_id} no longer exists",
                    )
                )
                continue
            if not h.data.get("job_dir"):
                reports.append(
                    StatusReport(
                        status=JobStatus.PROVISIONING,
                        detail=f"instance {instance_id} is "
                        f"{inst.status or 'unknown'}; job not launched yet",
                    )
                )
                continue
            try:
                reports.append(
                    jobdir.derive_status(
                        self._exec_from_handle(h),
                        h.data["job_dir"],
                        raise_unreachable=True,
                    )
                )
            except ExecError as e:
                raise BackendUnreachable(str(e)) from e
        return reports

    def _wait_provisioned_watched(self, instance_id: str) -> Instance:
        """``_wait_provisioned`` plus the COST-4 no-progress watchdog: the
        provider's progress signature (status string + ssh endpoint) unchanged
        AND not ready for ``provision_stall_s`` (default 90 s) means the rental
        is wedged — fail now instead of billing until the overall timeout."""
        timeout = float(
            self.config.extra("provision_timeout_s", DEFAULT_PROVISION_TIMEOUT_S)
        )
        stall = float(self.config.extra("provision_stall_s", DEFAULT_PROVISION_STALL_S))
        deadline = _monotonic() + timeout
        signature: str | None = None
        signature_since = _monotonic()
        last: Instance | None = None
        while True:
            inst = self._get_instance(instance_id)
            if inst is not None:
                last = inst
                if inst.ssh_target and inst.status.lower() == "running":
                    return inst
            sig = (
                f"{inst.status}|{inst.ssh_target}|{inst.ssh_port}"
                if inst is not None
                else "<not visible>"
            )
            now = _monotonic()
            if sig != signature:
                signature, signature_since = sig, now
            elif now - signature_since >= stall:
                raise InstanceUnreachable(
                    f"{self.name}: instance {instance_id} made no provisioning "
                    f"progress for {stall:.0f}s (status {sig!r} unchanged) — "
                    "treating the rental as dead"
                )
            if _monotonic() >= deadline:
                raise InstanceUnreachable(
                    f"{self.name}: instance {instance_id} not ready after "
                    f"{timeout:.0f}s (last status: "
                    f"{last.status if last else 'not visible yet'})"
                )
            _sleep(PROVISION_POLL_S)

    @staticmethod
    def _offer_key(offer: Offer) -> str:
        """A stable identifier for an offer so re-provisioning can avoid re-picking
        the exact rental that just failed."""
        d = offer.details or {}
        return str(d.get("ask_id") or d.get("gpu_type_id") or offer.label)

    def _reprobe_offer(self, spec: JobSpec, exclude: set[str]) -> Offer | None:
        """Re-probe the market and return the cheapest fitting offer, preferring one
        not already tried (probe returns cheapest-first). If every fitting offer has
        been tried — e.g. a provider that lists one reusable instance type rather
        than per-host asks — fall back to the cheapest so the retry still happens.
        None only when nothing fits. Best-effort: a probe failure ends the loop."""
        try:
            fitting = [o for o in self.probe(spec.resources) if o.fits]
        except Exception:
            return None
        if not fitting:
            return None
        for o in fitting:
            if self._offer_key(o) not in exclude:
                return o
        return fitting[0]

    def _submit_once(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None,
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
            # A dead rental (unreachable) propagates so submit re-provisions a fresh
            # instance; a BackendError or interrupt propagates as-is; anything else
            # is the job's/config's fault and is wrapped (not retried).
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
                _log.debug(
                    "%s: instance %s status=%r ssh=%s:%s",
                    self.name,
                    instance_id,
                    inst.status,
                    inst.ssh_target,
                    inst.ssh_port,
                )
                if inst.ssh_target and inst.status.lower() == "running":
                    return inst
            if time.monotonic() >= deadline:
                raise InstanceUnreachable(
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
        attempts = 0
        # The reason ssh is not up yet — connection refused (sshd not started),
        # auth failure (key not on the instance/account), timeout, etc. Kept so a
        # dead rental is reported with WHY, not just "did not accept connections".
        last_reason = "no attempt completed"
        while True:
            attempts += 1
            try:
                res = ex.run("true", timeout=15)
                if res.ok:
                    _log.debug(
                        "%s: ssh to %s ready after %d attempt(s)",
                        self.name,
                        ex.describe(),
                        attempts,
                    )
                    return
                last_reason = (
                    f"exit {res.returncode}: {(res.stderr or '').strip()[:200]}"
                )
            except Exception as e:
                last_reason = f"{type(e).__name__}: {e}"
            # Every probe result is logged at DEBUG so a stuck placement shows the
            # real handshake error each cycle, not silence.
            _log.debug(
                "%s: ssh to %s not ready (attempt %d): %s",
                self.name,
                ex.describe(),
                attempts,
                last_reason,
            )
            if time.monotonic() >= deadline:
                raise InstanceUnreachable(
                    f"{self.name}: sshd on {ex.describe()} did not accept connections "
                    f"within {timeout:.0f}s ({attempts} attempts; last: {last_reason})"
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
            report.status.settled
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
        return jobdir.tail_logs(ex, job_dir, follow=follow)

    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        ex = self._exec_from_handle(handle)
        paths = jobdir.pull_outputs(ex, handle.data["job_dir"], dest)
        if self.config.extra("auto_terminate", True):
            instance_id = handle.data["instance_id"]
            try:
                if self._get_instance(instance_id) is not None:
                    self._terminate(instance_id)
            except BackendError as e:
                # The pull SUCCEEDED — never trade that for a raise. The core's
                # reap stage retries the terminate (and reports honestly).
                _log.warning(
                    "%s: outputs pulled to %s, but auto-terminate of instance %s "
                    "failed — it is still billing; the reap will retry (%s)",
                    self.name,
                    dest,
                    instance_id,
                    e,
                )
        return paths

    def capture_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        """Pull-only variant for the engine's capture work item: never
        auto-terminates — capture precedes release (I6), the reap tears down."""
        ex = self._exec_from_handle(handle)
        return jobdir.pull_outputs(ex, handle.data["job_dir"], dest)

    def cancel(self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL) -> None:
        if handle.data.get("job_dir"):  # a provisioning stub has nothing to kill
            sig = "KILL" if mode is CancelMode.FORCE else "TERM"
            try:  # best-effort remote signal; the instance dies right after anyway
                jobdir.signal_job(
                    self._exec_from_handle(handle), handle.data["job_dir"], sig
                )
            except Exception:
                pass
        instance_id = handle.data["instance_id"]
        # Idempotent reap: terminate the billing instance if it still exists,
        # REGARDLESS of the job's own state — a finished job can still be billing.
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

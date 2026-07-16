"""Backend protocol and registry.

A backend turns a JobSpec into a running job somewhere and answers questions
about it afterwards. Implementations register with @register("type-name") and
are constructed from their BackendConfig section by config.load_backends().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from omnirun.models import (
    CancelMode,
    Capabilities,
    Health,
    JobHandle,
    JobSpec,
    Offer,
    ProviderFacts,
    ReapPolicy,
    ResourceSpec,
    StatusReport,
)

if TYPE_CHECKING:
    from omnirun.config import BackendConfig


class BackendError(RuntimeError):
    """Raised for backend-level failures with a user-actionable message."""


class CapacityError(BackendError):
    """Raised by ``submit`` when the backend has no room to place a job *right
    now* — a concurrent-session / quota cap (e.g. Colab's free-tier one-session
    limit), not a defect. It is transient and expected: the caller should release
    the reservation and retry on a later tick rather than fail the job."""


class BackendUnreachable(BackendError):
    """The backend service cannot be contacted or authenticated from this
    environment (missing/invalid API key, network transport failure).

    The true state of the backend's jobs and resources is UNKNOWN — callers
    must treat any operation that raises this as "no information" and change
    nothing (no requeue, no release, no reaped flag), leaving the record for
    an environment that can reach the backend.
    """


@dataclass
class SSHEndpoint:
    """SSH connection parameters for ``omnirun ssh <job>``.

    Returned by ``Backend.ssh_endpoint(handle)`` when the job is reachable
    over SSH.  None means the job is not currently ssh-reachable (not yet
    provisioned, already torn down, or the backend does not support it).

    Attributes:
        host:     Hostname or IP to connect to.
        port:     SSH (or bore tunnel) port number.
        user:     Remote login user (typically ``root`` for notebook workers).
        key_path: Path to the private key to use for authentication; the
                  omnirun-managed key at ``<state_dir>/ssh/id_ed25519``.
    """

    host: str
    port: int
    user: str
    key_path: Path


#: Optional hook submit() calls the instant it creates a *billable* resource,
#: before the (possibly minutes-long) wait for it to become usable. It hands the
#: client a partial JobHandle so a recovery stub can be persisted immediately —
#: an interrupted submit then stays visible to `ps`/`gc` instead of orphaning a
#: running instance with no local record.
ProvisioningSink = Callable[[JobHandle], None]


class Backend(ABC):
    """One configured execution target (a cluster, a machine, a provider account).

    Rules for implementations:
    - probe() must be fast (called speculatively with a ~10s budget), must not
      mutate anything, and must NEVER raise: on error return a single not-fit
      Offer carrying the error in unfit_reasons.
    - submit() may take long (repo push, provisioning). It must either return a
      working JobHandle or raise BackendError after cleaning up anything billable.
    - status()/logs()/cancel()/pull_outputs() take the handle produced by submit
      and must tolerate being called from a fresh process (no in-memory state).
    """

    #: registry type name, set by @register
    type_name: str = ""

    #: The teardown contract this backend hands the core reconciler: whether a
    #: terminal job still holds a capacity-occupying resource that must be
    #: collected-then-released, and whether a LOST placement is safe to
    #: force-release. Backends override it — as a class attribute for a fixed
    #: policy, or per-instance in ``__init__`` when it depends on config. The
    #: default (all-False) is the plain transport case: nothing lingers to
    #: reclaim, and a LOST poll may be a transient blip on a live job.
    reap: ReapPolicy = ReapPolicy()

    def __init__(self, name: str, config: "BackendConfig") -> None:
        self.name = name  # config key, e.g. "uni"
        self.config = config

    @abstractmethod
    def probe(self, res: ResourceSpec) -> list[Offer]: ...

    @abstractmethod
    def submit(
        self,
        spec: JobSpec,
        offer: Offer,
        on_provisioning: ProvisioningSink | None = None,
    ) -> JobHandle:
        """Run the job and return a handle. If a billable resource is created
        before the handle is ready, call on_provisioning with a partial handle
        first so the client can persist a recovery stub (see ProvisioningSink)."""
        ...

    @abstractmethod
    def status(self, handle: JobHandle) -> StatusReport: ...

    @abstractmethod
    def logs(self, handle: JobHandle, follow: bool = False) -> Iterator[str]:
        """Yield log lines (stdout+stderr merged). follow=True tails until the
        job reaches a terminal state."""
        ...

    @abstractmethod
    def cancel(
        self, handle: JobHandle, mode: CancelMode = CancelMode.GRACEFUL
    ) -> None: ...

    @abstractmethod
    def pull_outputs(self, handle: JobHandle, dest: Path) -> list[Path]:
        """Copy the job's collected outputs into dest; return copied paths."""
        ...

    def gc(self, handle: JobHandle) -> None:
        """Release everything the job holds remotely (worktrees, instances).

        Called by `omnirun gc` for terminal jobs. Default: nothing to do.
        Backends with billable resources MUST override."""

    def ssh_endpoint(self, handle: JobHandle) -> SSHEndpoint | None:
        """Return SSH connection parameters for this job, or None.

        None is returned when:
        - the job is not yet provisioned or has been torn down;
        - the backend does not support direct SSH access (e.g. Slurm, marketplace).

        The default implementation returns None.  Notebook backends (colab,
        kaggle) return an ``SSHEndpoint`` pointing at the bore tunnel port
        assigned at submit time when bore is configured.  SSH/local backends
        return the direct target they already use.
        """
        return None

    def check(self) -> str:
        """Connectivity/config sanity check for `omnirun backends check`.

        Return a short human 'ok: ...' description or raise BackendError."""
        return "ok"

    def discover(self) -> ProviderFacts:
        """Gather live facts about this backend (capabilities, limits, quota, health).

        Default: capabilities from statically declared config GPUs, health from
        check(). Backends with queryable limits/quota override this. Must NOT raise.
        """
        try:
            caps = Capabilities(gpu_types=[g.normalized() for g in self.config.gpus])
            detail = self.check()
            health, health_detail = Health.OK, detail
        except Exception as e:  # discover never raises
            caps = Capabilities()
            health, health_detail = Health.UNREACHABLE, str(e)
        return ProviderFacts(
            backend=self.name,
            discovered_at=datetime.now(timezone.utc),
            capabilities=caps,
            health=health,
            health_detail=health_detail,
        )


_REGISTRY: dict[str, type[Backend]] = {}

_B = TypeVar("_B", bound=Backend)


def register(type_name: str) -> Callable[[type[_B]], type[_B]]:
    def deco(cls: type[_B]) -> type[_B]:
        cls.type_name = type_name
        _REGISTRY[type_name] = cls
        return cls

    return deco


def backend_class(type_name: str) -> type[Backend]:
    # Import concrete backend modules lazily so optional deps (kaggle, colab)
    # don't break unrelated commands.
    if type_name not in _REGISTRY:
        import importlib

        mod = {
            "local": "omnirun.backends.local",
            "ssh": "omnirun.backends.ssh",
            "slurm": "omnirun.backends.slurm",
            "kaggle": "omnirun.backends.kaggle",
            "colab": "omnirun.backends.colab",
            "runpod": "omnirun.backends.runpod",
            "vast": "omnirun.backends.vast",
            "thunder": "omnirun.backends.thunder",
        }.get(type_name)
        if mod is None:
            raise BackendError(f"unknown backend type {type_name!r}")
        importlib.import_module(mod)
    return _REGISTRY[type_name]


def make_backend(name: str, config: Any) -> Backend:
    return backend_class(config.type)(name, config)

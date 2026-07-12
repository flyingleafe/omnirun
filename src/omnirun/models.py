"""Core data models. Every other module builds against these contracts."""

from __future__ import annotations

import enum
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Normalized GPU type names used across all backends. Backend configs map these
# to site-/provider-specific identifiers (gres strings, RunPod gpuTypeIds, ...).
# Matching is case-insensitive and dash/space-insensitive ("A100-80" == "a100 80").
KNOWN_GPU_VRAM_GB: dict[str, float] = {
    "T4": 16,
    "2xT4": 32,  # Kaggle's paired T4s
    "P100": 16,
    "V100": 16,
    "V100-32": 32,
    "L4": 24,
    "L40": 48,
    "A6000": 48,
    "A100": 40,
    "A100-80": 80,
    "H100": 80,
    "H200": 141,
    "4090": 24,
    "3090": 24,
    "5090": 32,
    "RTX-PRO-6000": 96,
}


def normalize_gpu_type(name: str) -> str:
    key = re.sub(r"[\s_-]+", "", name).lower()
    for known in KNOWN_GPU_VRAM_GB:
        if re.sub(r"[\s_-]+", "", known).lower() == key:
            return known
    return name  # unknown types pass through verbatim


class ResourceSpec(BaseModel):
    """What the job needs. Everything optional; empty spec = 'any CPU box'."""

    gpus: int = 0
    gpu_type: str | None = None  # normalized name, see KNOWN_GPU_VRAM_GB
    min_vram_gb: float | None = None  # alternative to gpu_type
    min_cuda: str | None = None  # min CUDA version, e.g. "12.4"
    cpus: int | None = None
    mem_gb: float | None = None
    time: timedelta | None = None  # estimated duration; drives cost math + --time
    disk_gb: float | None = None

    @field_validator("gpu_type")
    @classmethod
    def _norm(cls, v: str | None) -> str | None:
        return normalize_gpu_type(v) if v else None

    def wants_gpu(self) -> bool:
        return (
            self.gpus > 0 or self.gpu_type is not None or self.min_vram_gb is not None
        )

    def effective_gpus(self) -> int:
        return max(self.gpus, 1) if self.wants_gpu() else 0

    def vram_floor_gb(self) -> float | None:
        """Minimum acceptable per-GPU VRAM implied by the spec."""
        if self.min_vram_gb is not None:
            return self.min_vram_gb
        if self.gpu_type is not None:
            return KNOWN_GPU_VRAM_GB.get(self.gpu_type)
        return None


def _cuda_tuple(v: str | float) -> tuple[int, ...]:
    out: list[int] = []
    for part in str(v).strip().split("."):
        if part.isdigit():
            out.append(int(part))
        else:
            break
    return tuple(out) or (0,)


def cuda_at_least(have: str | float | None, need: str | float | None) -> bool:
    """True if CUDA ``have`` >= ``need``. Unknown/unparseable side -> True (don't block)."""
    if have is None or need is None:
        return True
    return _cuda_tuple(have) >= _cuda_tuple(need)


class Health(str, enum.Enum):
    OK = "ok"
    DEGRADED = "degraded"  # reachable but constrained (quota low, partition busy)
    UNREACHABLE = "unreachable"


class Capabilities(BaseModel):
    """What a backend can offer, discovered or declared. A None/empty field is
    'unknown' and never used to reject a job."""

    gpu_types: list[str] = Field(default_factory=list)  # normalized names available
    max_vram_gb: float | None = None
    max_gpus_per_job: int | None = None
    cuda_version: str | None = None  # max CUDA the host driver supports
    max_walltime: timedelta | None = None
    max_parallel_jobs: int | None = None

    def satisfies(self, res: ResourceSpec) -> list[str]:
        """Unfit reasons for a job with requirements ``res``; empty list = fits."""
        reasons: list[str] = []
        if res.gpu_type and self.gpu_types and res.gpu_type not in self.gpu_types:
            have = ", ".join(self.gpu_types) or "none"
            reasons.append(f"GPU {res.gpu_type} not available (offers: {have})")
        floor = res.vram_floor_gb()
        if (
            floor is not None
            and self.max_vram_gb is not None
            and floor > self.max_vram_gb
        ):
            reasons.append(
                f"needs >={floor:g}GB VRAM, max here is {self.max_vram_gb:g}GB"
            )
        if (
            res.time is not None
            and self.max_walltime is not None
            and res.time > self.max_walltime
        ):
            reasons.append(f"time {res.time} exceeds max walltime {self.max_walltime}")
        if not cuda_at_least(self.cuda_version, res.min_cuda):
            reasons.append(f"CUDA {self.cuda_version} < required {res.min_cuda}")
        want_gpus = res.effective_gpus()
        if (
            want_gpus
            and self.max_gpus_per_job is not None
            and want_gpus > self.max_gpus_per_job
        ):
            reasons.append(
                f"wants {want_gpus} GPUs, max {self.max_gpus_per_job} per job"
            )
        return reasons


class ProviderFacts(BaseModel):
    """Discovered metadata about a backend, cached with a TTL."""

    backend: str
    discovered_at: datetime
    ttl_s: float = 3600.0
    capabilities: Capabilities = Field(default_factory=Capabilities)
    health: Health = Health.OK
    health_detail: str = ""
    budget_state: dict[str, Any] = Field(default_factory=dict)

    def is_fresh(self, now: datetime) -> bool:
        return (now - self.discovered_at).total_seconds() < self.ttl_s


class EnvKind(str, enum.Enum):
    AUTO = "auto"  # detect from repo contents at bootstrap time
    UV = "uv"  # uv sync (uv.lock / pyproject.toml)
    PIP = "pip"  # uv venv + uv pip install -r requirements.txt
    CONDA = "conda"  # micromamba create -f environment.yml
    SYSTEM = "system"  # install deps into the ambient interpreter (notebooks: keep their torch)
    NONE = "none"  # run bare; user's setup lines do everything


class EnvSpec(BaseModel):
    kind: EnvKind = EnvKind.AUTO
    # Raw shell lines run before env creation (module loads, exports).
    # Backend config's env_setup is prepended to these.
    setup: list[str] = Field(default_factory=list)
    # Extra shell lines run inside the activated env, before the command.
    pre_run: list[str] = Field(default_factory=list)


class RepoRef(BaseModel):
    """Immutable snapshot of the repo state a job runs against."""

    remote_url: str
    sha: str
    branch: str
    slug: str  # filesystem-safe repo name, e.g. "omnirun"
    # Client-side only: absolute path of the local repo root at capture time, so
    # submit paths don't have to assume Path.cwd() is the repo root. Meaningless
    # on workers (the worker materializes its own tree from the sha).
    local_root: str | None = None


class Deadline(BaseModel):
    """Optional window in which a job must start and/or finish."""

    start_by: datetime | None = None
    finish_by: datetime | None = None


class JobPolicy(BaseModel):
    """Request-level scheduling policy; travels with the spec via serialization."""

    deadline: Deadline | None = None
    max_cost: float | None = None  # USD ceiling for a single job run
    priority: int = 0  # higher = scheduled sooner; reprioritizable after submission


class JobSpec(BaseModel):
    job_id: str  # "<name>-<hex6>", globally unique per client
    name: str
    command: str  # executed via bash -c in the worktree root
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    env: EnvSpec = Field(default_factory=EnvSpec)
    outputs: list[str] = Field(default_factory=list)  # globs relative to repo root
    repo: RepoRef
    env_vars: dict[str, str] = Field(default_factory=dict)  # forwarded to the job
    policy: JobPolicy = Field(default_factory=JobPolicy)

    @staticmethod
    def make_job_id(name: str) -> str:
        safe = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "job"
        return f"{safe[:24]}-{secrets.token_hex(3)}"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"  # accepted by backend, waiting to start
    PROVISIONING = "provisioning"  # machine being created (marketplaces)
    STARTING = "starting"  # bootstrap running (code/env setup)
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOST = "lost"  # can't reach worker / handle stale

    @property
    def terminal(self) -> bool:
        return self in (
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.LOST,
        )


class Offer(BaseModel):
    """One concrete way to run a job, produced by Backend.probe()."""

    backend: str  # config key ("uni", "runpod", ...)
    label: str  # human line: "runpod: H100 SXM (secure) $2.79/hr"
    fits: bool = True
    unfit_reasons: list[str] = Field(default_factory=list)
    gpu_type: str | None = None  # normalized, what you'd actually get
    gpus: int = 0
    cost_per_hour: float | None = None  # USD; None = free
    wait_estimate_s: float | None = None  # None = unknown
    wait_note: str = ""  # honesty label: "backfill estimate, often pessimistic"
    notes: str = ""
    details: dict[str, Any] = Field(default_factory=dict)  # backend-specific

    def total_cost(self, time: timedelta | None) -> float | None:
        """Estimated total USD for the job, None if free or unknowable."""
        if self.cost_per_hour is None:
            return None
        hours = (time.total_seconds() / 3600) if time else 1.0
        return self.cost_per_hour * hours


class JobHandle(BaseModel):
    """Everything a backend needs to find a submitted job again."""

    backend: str
    job_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    # ssh family: {"host": ..., "job_dir": ...} (+ slurm: {"slurm_job_id": ...})
    # kaggle: {"kernel_ref": "user/slug", "dataset_ref": ...}
    # colab: {"session": ..., "job_dir": ...}
    # marketplaces: ssh keys + {"instance_id": ..., "provider": ...}


class StatusReport(BaseModel):
    status: JobStatus
    exit_code: int | None = None
    detail: str = ""  # e.g. slurm reason, kernel status string
    started_at: datetime | None = None
    finished_at: datetime | None = None


# ---------------------------------------------------------------------------
# Phase-3 scheduler domain types (JobState + Placement defined early so that
# JobRecord can reference them as field types and default values)
# ---------------------------------------------------------------------------


class JobState(str, enum.Enum):
    """Scheduler-level lifecycle state of a job (distinct from backend ``JobStatus``)."""

    QUEUED = "queued"
    HELD = "held"  # admitted but no suitable slot found; retried each tick
    PLACING = "placing"  # tick emitted a ``place`` Decision; provider.place in flight
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED)


class Link(BaseModel):
    """Human-facing URL for a running job (dashboard, notebook, kernel, etc.).

    DISPLAY ONLY — never used by the pure scheduler tick for decisions.
    """

    label: str
    url: str


class Placement(BaseModel):
    """Record of a job placed on a specific provider slot."""

    provider_name: str  # DISPLAY ONLY
    job_id: str
    handle: dict[str, Any] = Field(default_factory=dict)
    links: list[Link] = Field(default_factory=list)
    cost_actual: float | None = None
    state: JobStatus = JobStatus.QUEUED
    placed_at: datetime | None = None
    ended_at: datetime | None = None


class JobRecord(BaseModel):
    """What the local store persists per job."""

    spec: JobSpec
    handle: JobHandle | None = None
    offer: Offer | None = None
    submitted_at: datetime | None = None
    last_status: StatusReport | None = None
    outputs_pulled_to: str | None = None
    schema_version: int = 0  # 0 = written before state versioning; Store.save_job stamps the current version
    # --- scheduler runtime state (Task 2) ---
    attempts: int = 0  # number of placement attempts so far
    state: JobState = JobState.QUEUED  # scheduler-level lifecycle state
    placement: Placement | None = None  # active or most-recent placement

    def urgency(self, now: datetime) -> float:
        """Higher value = more urgent; used to rank QUEUED jobs within a priority tier.

        If no ``finish_by`` deadline is set, returns 0.0 (no deadline pressure).

        Otherwise computes the latest safe start = ``finish_by`` minus the
        estimated run time (``spec.resources.time``, defaulting to zero).
        Urgency = ``-(latest_safe_start - now).total_seconds()``, so:

        * A job whose latest-safe-start is far in the future has a large negative
          value (low urgency).
        * A job whose latest-safe-start is at ``now`` has urgency 0.0.
        * A job whose latest-safe-start has already passed has positive urgency
          (highest urgency — act immediately).

        Monotonic: less slack ⇒ strictly higher urgency.
        """
        deadline = self.spec.policy.deadline
        if deadline is None or deadline.finish_by is None:
            return 0.0
        est_runtime = self.spec.resources.time or timedelta(0)
        latest_safe_start = deadline.finish_by - est_runtime
        # Defensive: a naive/aware datetime mix would make subtraction raise
        # TypeError, which would propagate out of the pure scheduler tick.
        # If exactly one side is naive, treat the naive one as UTC.
        if (latest_safe_start.tzinfo is None) != (now.tzinfo is None):
            if latest_safe_start.tzinfo is None:
                latest_safe_start = latest_safe_start.replace(tzinfo=timezone.utc)
            else:
                now = now.replace(tzinfo=timezone.utc)
        return -(latest_safe_start - now).total_seconds()


# ---------------------------------------------------------------------------
# Remaining Phase-3 scheduler domain types
# ---------------------------------------------------------------------------


class Cost(BaseModel):
    """Pricing for a slot.  ``per_hour is None`` means free (returns 0.0 from total)."""

    setup: float | None = None
    per_hour: float | None = None

    def total(self, dur: timedelta | None) -> float | None:
        """Estimated total USD for ``dur``.

        * Free (``per_hour is None``) → 0.0.
        * ``dur`` is None but slot is paid → None (unknowable).
        * Otherwise ``(setup or 0) + per_hour * hours``.
        """
        if self.per_hour is None:
            return 0.0
        if dur is None:
            return None
        hours = dur.total_seconds() / 3600
        return (self.setup or 0.0) + self.per_hour * hours


class Availability(BaseModel):
    """When a slot can start a job."""

    kind: Literal["ready_now", "queued", "provision"] = "ready_now"
    wait_s: float | None = None  # queued/provision wait estimate in seconds
    note: str = ""


class Slot(BaseModel):
    """One concrete way to run a job, offered by a Provider.

    ``provider_name`` and any ``Link`` attached to a ``Placement`` are
    DISPLAY ONLY — ``tick`` never reads them for routing decisions.
    Fit is determined solely by ``capabilities.satisfies(req)``.
    """

    provider_name: str  # DISPLAY ONLY
    capabilities: Capabilities
    cost: Cost = Field(default_factory=Cost)
    availability: Availability = Field(default_factory=Availability)
    capacity: int = 1  # remaining concurrent jobs this slot can accept
    provider_ref: dict[str, Any] = Field(
        default_factory=dict
    )  # opaque; echoed to provider.place

    def fits(self, req: ResourceSpec) -> bool:
        """True when ``capabilities`` satisfy all requirements in ``req``."""
        return not self.capabilities.satisfies(req)


class Status(BaseModel):
    """Uniform provider → scheduler signal; wraps the backend ``JobStatus`` enum."""

    state: JobStatus
    exit_code: int | None = None
    detail: str = ""


class Decision(BaseModel):
    """Output of one ``tick`` call for a single job."""

    kind: Literal["place", "hold", "requeue", "noop"]
    job_id: str
    slot: Slot | None = None
    reason: str = ""


class CancelMode(str, enum.Enum):
    """How firmly to cancel a placement.

    ``GRACEFUL`` asks the job to stop and lets it clean up; ``FORCE`` tears the
    resource down immediately. Phase 3's ``BackendProvider`` treats both as a
    best-effort delegate to ``Backend.cancel``; the graceful→force reaping
    distinction is deepened in Phase 4.
    """

    GRACEFUL = "graceful"
    FORCE = "force"

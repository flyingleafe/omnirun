"""EndpointManager: one shared session/throttle/discovery-cache per physical
target (DESIGN-V2 §3.1, CONN-*).

Motivation (tick-anatomy findings 4-5): three slurm backends configured against
one login host each built their own ``SSHExec`` and ran identical discovery
queries (``sacctmgr``/``sinfo``/``scontrol``) in the same second, and every
marketplace backend instance throttled its provider API calls independently.
The manager deduplicates all three kinds of per-target state:

- :meth:`ssh_exec` — a shared ``SSHExec`` per physical ssh target, so every
  backend pointed at one host multiplexes over the SAME instance (and its one
  ControlMaster + serialized-auth machinery in ``omnirun.execlayer.ssh``).
- :meth:`throttle` — a shared :class:`Throttle` per provider name, so the
  rate-limit spacing covers ALL concurrent callers of that provider's API,
  not each backend instance separately.
- :meth:`cached` — a per-``(endpoint, query)`` TTL cache with single-flight
  locking, so concurrent identical discovery calls coalesce into ONE remote
  round per TTL window.

There are NO module-level singletons: whoever owns the process's backends
(``LocalClient``; the daemon through its ``LocalClient``) constructs one
manager and injects it via ``Backend.endpoints``; tests construct their own.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar, cast

from omnirun.execlayer.ssh import SSHExec

_T = TypeVar("_T")

_sleep = time.sleep  # test seam

#: Conservative default TTLs (seconds) for the discovery cache. Slow-moving
#: cluster facts (partition caps, associations, QOS) hold for 5 minutes; a
#: start-time estimate and a quota read are fresher signals and hold for 1.
DISCOVER_TTL_S = 300.0
ESTIMATE_TTL_S = 60.0
QUOTA_TTL_S = 60.0

CacheKey = tuple[object, ...]


class Throttle:
    """Shared min-interval spacing for one provider API.

    Semantics match the old per-backend ``MarketplaceBackend._throttle``: each
    ``wait`` sleeps just enough to keep successive calls ``min_interval_s``
    apart — but the *last-call* state now lives here, shared by every backend
    section (and every placement thread) talking to the same provider, so a
    burst across sections cannot exceed the provider's global ceiling.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._last_call_at = 0.0

    def wait(self, min_interval_s: float) -> None:
        """Block until ``min_interval_s`` has passed since the previous call
        (across ALL users of this throttle). A no-op when the interval is 0."""
        if min_interval_s <= 0:
            return
        with self._lock:
            delay = min_interval_s - (self._clock() - self._last_call_at)
            if delay > 0:
                _sleep(delay)
            self._last_call_at = self._clock()


class EndpointManager:
    """Process-wide registry of per-physical-target shared state.

    Thread-safe. One instance per placer process (``LocalClient`` /
    daemon); never a global.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._ssh: dict[CacheKey, SSHExec] = {}
        self._throttles: dict[str, Throttle] = {}
        # key -> (expires_at, value). Guarded per-key by _cache_locks so
        # concurrent identical producers coalesce (single flight) without one
        # slow endpoint blocking cache hits for another.
        self._cache: dict[CacheKey, tuple[float, object]] = {}
        self._cache_locks: dict[CacheKey, threading.Lock] = {}

    # --- shared ssh sessions -------------------------------------------------

    def ssh_exec(
        self,
        target: str,
        *,
        port: int | None = None,
        identity: str | None = None,
        extra_opts: Sequence[str] | None = None,
        control_dir: Path | None = None,
        login_shell: bool = False,
        ssh_command: Sequence[str] = ("ssh",),
        control_master: bool = True,
        batch_mode: bool = True,
        control_persist: str = "10m",
    ) -> SSHExec:
        """The ONE ``SSHExec`` for this physical ssh target + option set.

        Keyed by every construction parameter: backends with identical config
        (the three-partitions-one-login-host case) get the same instance and
        therefore one ControlMaster lifecycle; backends that genuinely differ
        (e.g. ``login_shell``) keep distinct instances — those still share the
        OS-level master socket via ``ControlPath`` and the serialized-auth lock
        in ``omnirun.execlayer.ssh``, so behavior is preserved either way.
        """
        key: CacheKey = (
            "ssh-exec",
            target,
            port,
            identity,
            tuple(extra_opts or ()),
            str(control_dir) if control_dir is not None else None,
            login_shell,
            tuple(ssh_command),
            control_master,
            batch_mode,
            control_persist,
        )
        with self._lock:
            ex = self._ssh.get(key)
            if ex is None:
                ex = SSHExec(
                    target,
                    port=port,
                    identity=identity,
                    extra_opts=list(extra_opts or []),
                    control_dir=control_dir,
                    login_shell=login_shell,
                    ssh_command=ssh_command,
                    control_master=control_master,
                    batch_mode=batch_mode,
                    control_persist=control_persist,
                )
                self._ssh[key] = ex
            return ex

    # --- shared provider throttles -------------------------------------------

    def throttle(self, provider: str) -> Throttle:
        """The ONE :class:`Throttle` for this provider API (keyed by name)."""
        with self._lock:
            th = self._throttles.get(provider)
            if th is None:
                th = Throttle(clock=self._clock)
                self._throttles[provider] = th
            return th

    # --- shared discovery cache ----------------------------------------------

    def cached(
        self,
        key: CacheKey,
        ttl_s: float,
        producer: Callable[[], _T],
        *,
        should_cache: Callable[[_T], bool] | None = None,
    ) -> _T:
        """A per-``(endpoint, query)`` TTL cache with single-flight locking.

        Concurrent calls with the same *key* coalesce: one caller runs
        *producer* while the rest block on the key's lock and then read the
        fresh value — N backends discovering one host make ONE remote round
        per query per TTL window. A producer that raises caches nothing (the
        exception propagates; the next caller retries), and *should_cache*
        lets callers keep failure-shaped values (e.g. a not-ok ``ExecResult``)
        out of the cache so a transient error never sticks for a TTL.
        """
        lock = self._key_lock(key)
        with lock:
            hit = self._cache.get(key)
            if hit is not None and self._clock() < hit[0]:
                return cast(_T, hit[1])
            value = producer()
            if should_cache is None or should_cache(value):
                self._cache[key] = (self._clock() + ttl_s, value)
            else:
                self._cache.pop(key, None)
            return value

    def invalidate(self, key: CacheKey | None = None) -> None:
        """Drop one cached entry (or all of them when *key* is None)."""
        with self._lock:
            if key is None:
                self._cache.clear()
            else:
                self._cache.pop(key, None)

    def _key_lock(self, key: CacheKey) -> threading.Lock:
        with self._lock:
            lock = self._cache_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._cache_locks[key] = lock
            return lock

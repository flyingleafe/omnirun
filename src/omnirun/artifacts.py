"""Where a job's captured outputs come to rest.

Capture always lands on local disk first: a provider pulls over its own
transport, so the bytes must touch a file before anything else can happen.
What happens *after* that is this seam's business. The local store leaves the
bytes where they are; the object store uploads them and drops the local copy.

Each store returns a *pointer*, and that pointer is what ``outputs_cached_to``
carries on the job row. The same store reads the pointer back for ``pull``.
Pointers are therefore stable identifiers, not paths — a local store returns a
directory path, an object store returns a URL. A store must always read a
pointer written by an earlier configuration, because a tree migrated to an
object store keeps company with jobs captured before the move.

Only outputs move. The capture sink keeps ``log.txt`` on local disk, and
``logs_cached_to`` keeps pointing at the sink, so the log paths stay untouched.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from omnirun.backends.base import BackendError
from omnirun.config import Config, ConfigError

#: The marker that separates a URL pointer from a plain filesystem path.
_SCHEME = "://"


class ArtifactStore(Protocol):
    """The durable home of captured outputs (DESIGN §9).

    An implementation must be idempotent: ``publish`` of a sink that was
    already published, and ``fetch`` of the same pointer twice, must both be
    safe. The engine retries capture, and a restarted engine adopts.
    """

    def owns(self, pointer: str) -> bool:
        """True when this store wrote *pointer* and can read it back."""
        ...

    def publish(self, job_id: str, sink: Path) -> str:
        """Move the outputs under *sink* to their durable home. Returns the
        pointer to record in ``outputs_cached_to``."""
        ...

    def fetch(self, pointer: str, dest: Path) -> list[Path]:
        """Materialize the outputs named by *pointer* into *dest*."""
        ...


def _missing(pointer: str) -> BackendError:
    return BackendError(
        f"cached outputs are missing at {pointer} "
        "(session already reaped, nothing to re-fetch)"
    )


class LocalArtifactStore:
    """Outputs stay in the capture sink on local disk — the behavior omnirun
    had before object storage existed, and still the default."""

    def owns(self, pointer: str) -> bool:
        return _SCHEME not in pointer

    def publish(self, job_id: str, sink: Path) -> str:
        return str(sink)

    def fetch(self, pointer: str, dest: Path) -> list[Path]:
        cache = Path(pointer)
        src = cache / "outputs" if (cache / "outputs").is_dir() else cache
        if not src.is_dir():
            raise _missing(pointer)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest, dirs_exist_ok=True)
        return sorted(p for p in dest.rglob("*") if p.is_file())


class ObjectArtifactStore:
    """Outputs live in an S3-compatible bucket; the local copy is dropped as
    soon as the upload is complete.

    Reads fall back to *local* for any pointer this store did not write, so a
    tree that predates the move still pulls.
    """

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        prefix: str = "",
        local: LocalArtifactStore | None = None,
        client: Any | None = None,
    ) -> None:
        self._bucket = bucket
        self._endpoint = endpoint_url
        self._prefix = prefix.strip("/")
        self._local = local or LocalArtifactStore()
        self._client_obj = client

    # -- infra --

    def _client(self) -> Any:
        if self._client_obj is None:
            try:
                import boto3
            except ImportError as e:  # pragma: no cover - packaging guard
                raise ConfigError(
                    "object artifact storage needs boto3 — install omnirun[s3]"
                ) from e
            self._client_obj = boto3.client("s3", endpoint_url=self._endpoint)
        return self._client_obj

    def _key(self, job_id: str) -> str:
        return "/".join(p for p in (self._prefix, job_id, "outputs") if p)

    def _url(self, key: str) -> str:
        return f"s3{_SCHEME}{self._bucket}/{key}"

    # -- the seam --

    def owns(self, pointer: str) -> bool:
        return pointer.startswith(f"s3{_SCHEME}")

    def publish(self, job_id: str, sink: Path) -> str:
        out = sink / "outputs"
        if not out.is_dir():
            # Nothing was captured; leave the pointer local so `pull` reports
            # the same "nothing to re-fetch" it always did.
            return self._local.publish(job_id, sink)
        key = self._key(job_id)
        client = self._client()
        for f in sorted(p for p in out.rglob("*") if p.is_file()):
            rel = f.relative_to(out).as_posix()
            client.upload_file(str(f), self._bucket, f"{key}/{rel}")
        shutil.rmtree(out)
        return self._url(key)

    def fetch(self, pointer: str, dest: Path) -> list[Path]:
        if not self.owns(pointer):
            return self._local.fetch(pointer, dest)
        parsed = urlparse(pointer)
        bucket, key = parsed.netloc, parsed.path.strip("/")
        client = self._client()
        dest.mkdir(parents=True, exist_ok=True)
        root = dest.resolve()
        got: list[Path] = []
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=f"{key}/"
        ):
            for obj in page.get("Contents", ()):
                rel = obj["Key"][len(key) + 1 :]
                if not rel or rel.endswith("/"):
                    continue
                target = (dest / rel).resolve()
                if not target.is_relative_to(root):
                    raise BackendError(f"object key escapes the pull dir: {obj['Key']}")
                target.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, obj["Key"], str(target))
                got.append(target)
        if not got:
            raise _missing(pointer)
        return sorted(got)


def make_artifact_store(cfg: Config) -> ArtifactStore:
    """The configured store. Absent an ``[artifacts]`` section every job keeps
    its outputs on local disk, which is what a laptop wants."""
    ac = cfg.artifacts
    if ac.store == "local":
        return LocalArtifactStore()
    if not ac.bucket:
        raise ConfigError("[artifacts] store = 's3' needs a bucket")
    return ObjectArtifactStore(
        ac.bucket, endpoint_url=ac.endpoint_url, prefix=ac.prefix
    )

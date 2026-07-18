"""The replay validator (DEPLOY-V2 §1) — production conformance checking.

Tails ``job_events`` via ``Store.events_after`` (cursor persisted in
``meta['validator_cursor']``), maintains BOTH validation views
(CONFORMANCE.md §2) with the trace exporter, and runs the compiled
``trace-check`` binary incrementally: each round with new events re-exports
every view **from init** and replays it whole — traces are small (thousands
of lines), and the full re-run keeps the validator stateless and the
checker's valid-from-``init`` guarantee intact.

On a VIOLATION a GitHub issue is filed through the daemon host's own ``gh``:

* fingerprint = sha256 over (view, violating line content, action, the
  trace's nid → job_id mapping) — one identity per distinct violation;
* dedup: if an OPEN issue already carries the fingerprint marker (found via
  ``gh issue list --search``), a counter comment is appended instead — at
  most once per hour per fingerprint (the cap rides ``meta``);
* ``--dry-run`` prints the would-be issue instead of filing anything.

Runs as ``omnirun validate-replay [--once|--interval S]`` — daemon-host only
(needs store access); deployed as its own systemd service so a validator
crash never touches the daemon.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from omnirun.state.store import Store
from omnirun.state.traceexport import export_global_trace, export_provider_trace

_log = logging.getLogger("omnirun.validator")

CURSOR_KEY = "validator_cursor"
COMMENT_KEY_PREFIX = "validator_comment."
ISSUE_LABEL = "model-violation"
FINGERPRINT_MARKER = "omnirun-violation-fingerprint:"

# Same non-binding init parameters the test trace gate uses: the validator
# checks the structural/lifecycle/resource invariants; the budget and capacity
# caps are effectively unbounded so a config change mid-history can never
# manufacture a false violation.
BUDGET_CENTS = 1_000_000_000
CAP = 1_000_000

_TAIL_LINES = 50  # trace window attached to a filed issue
_COMMENT_CAP = timedelta(hours=1)  # per-fingerprint comment pacing


class ValidatorError(RuntimeError):
    """A validator infrastructure failure (missing checker, broken gh)."""


def default_trace_check() -> Path:
    """Locate the ``trace-check`` binary: ``$OMNIRUN_TRACE_CHECK`` first, then
    the in-repo build, then ``trace-check`` on PATH."""
    env = os.environ.get("OMNIRUN_TRACE_CHECK")
    if env:
        return Path(env)
    repo_build = (
        Path(__file__).resolve().parents[2]
        / "formal"
        / ".lake"
        / "build"
        / "bin"
        / "trace-check"
    )
    if repo_build.is_file():
        return repo_build
    on_path = shutil.which("trace-check")
    if on_path:
        return Path(on_path)
    raise ValidatorError(
        "trace-check binary not found: set $OMNIRUN_TRACE_CHECK, build it with "
        "`nix build .#trace-check`, or run `lake build` in formal/"
    )


@dataclass(frozen=True)
class Violation:
    """One checker rejection: where, what, and its dedup identity."""

    view: str  # "global" or the provider name
    line_no: int
    line: str  # the violating trace line, verbatim
    reason: str
    action: str  # first token of the violating line
    job_id: str | None  # nid resolved through the trace's alias map
    fingerprint: str
    trace_tail: str  # the last _TAIL_LINES lines up to the violation

    @property
    def title(self) -> str:
        return (
            f"model violation: {self.action} on {self.job_id or '?'} ({self.view} view)"
        )

    @property
    def body(self) -> str:
        return (
            f"The replay validator rejected the `{self.view}` trace view at "
            f"line {self.line_no}:\n\n"
            f"```\n{self.line}\n```\n\n"
            f"Reason: {self.reason}\n\n"
            f"Job: `{self.job_id or 'unknown'}`\n\n"
            f"Trace window (last {_TAIL_LINES} lines up to the violation):\n\n"
            f"```\n{self.trace_tail}\n```\n\n"
            f"<!-- {FINGERPRINT_MARKER} {self.fingerprint} -->\n"
        )


def _parse_checker_output(output: str) -> tuple[int, str, str] | None:
    """``(line_no, line, reason)`` from the checker's VIOLATION stderr."""
    lines = [ln for ln in output.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if ln.startswith("VIOLATION line "):
            head, _, rest = ln.partition(": ")
            try:
                line_no = int(head.removeprefix("VIOLATION line ").strip())
            except ValueError:
                line_no = 0
            reason = lines[i + 1] if i + 1 < len(lines) else "rejected"
            return line_no, rest, reason
    return None


class ReplayValidator:
    """Tail the event log, re-validate both views, file deduplicated issues."""

    def __init__(
        self,
        store: Store,
        *,
        trace_check: Path | None = None,
        gh: str = "gh",
        dry_run: bool = False,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._checker = trace_check or default_trace_check()
        self._gh = gh
        self._dry_run = dry_run
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._stop = threading.Event()

    # -- rounds ----------------------------------------------------------

    def run_once(self) -> list[Violation]:
        """One validation round. Skipped (returns ``[]``) when no event has
        arrived since the persisted cursor; otherwise both views are
        re-exported from init and replayed; every violation found is reported
        (filed/commented, or printed in dry-run) and returned."""
        cursor = int(self._store.get_meta(CURSOR_KEY) or 0)
        page = self._store.events_after(cursor, limit=1)
        if not page:
            return []
        violations = self.validate_all()
        for violation in violations:
            self._report(violation)
        # Advance the cursor even on violation: the trace is re-validated only
        # when NEW events arrive; the dedup marker (not the cursor) is what
        # prevents duplicate issues for a persisting violation.
        last = self._last_event_id()
        self._store.set_meta(CURSOR_KEY, str(last))
        return violations

    def run_forever(self, interval_s: float = 60.0) -> None:
        """The service loop: a round every *interval_s* until stopped. A round
        that raises is logged and retried — the validator must outlive
        transient store/gh hiccups (systemd restarts cover crashes)."""
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:
                _log.exception("validation round failed; retrying")
            self._stop.wait(interval_s)

    def stop(self) -> None:
        self._stop.set()

    def _last_event_id(self) -> int:
        return self._store.last_event_id()

    # -- validation ------------------------------------------------------

    def _providers(self) -> list[str]:
        seen: set[str] = set()
        cursor = 0
        while True:
            page = self._store.events_after(cursor, limit=1000)
            if not page:
                break
            for ev in page:
                if ev.action == "reserve":
                    provider = (ev.data or {}).get("provider")
                    if provider:
                        seen.add(str(provider))
            cursor = page[-1].id
        return sorted(seen)

    def validate_all(self) -> list[Violation]:
        """Export + check the global view and one view per provider."""
        providers = self._providers()
        out: list[Violation] = []
        alias: dict[int, str] = {}
        trace = export_global_trace(
            self._store,
            budget_cents=BUDGET_CENTS,
            caps=dict.fromkeys(providers, CAP),
            with_asserts=True,
            alias_out=alias,
        )
        v = self._check_view("global", trace, alias)
        if v is not None:
            out.append(v)
        for provider in providers:
            alias = {}
            trace = export_provider_trace(
                self._store,
                provider,
                budget_cents=BUDGET_CENTS,
                cap=CAP,
                alias_out=alias,
            )
            v = self._check_view(provider, trace, alias)
            if v is not None:
                out.append(v)
        return out

    def _check_view(
        self, view: str, trace: str, alias: dict[int, str]
    ) -> Violation | None:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".trace", prefix=f"omnirun-{view}-", delete=False
        ) as f:
            f.write(trace)
            path = f.name
        try:
            proc = subprocess.run(
                [str(self._checker), path],
                capture_output=True,
                text=True,
                timeout=120,
            )
        finally:
            Path(path).unlink(missing_ok=True)
        output = proc.stdout + proc.stderr
        if proc.returncode == 0 and "VIOLATION" not in output:
            return None
        parsed = _parse_checker_output(output)
        if parsed is None:
            line_no, line, reason = 0, "", output.strip()[:500]
        else:
            line_no, line, reason = parsed
        tokens = line.split()
        action = tokens[0] if tokens else "?"
        job_id: str | None = None
        if len(tokens) > 1 and tokens[1].isdigit():
            job_id = alias.get(int(tokens[1]))
        # The dedup identity: view + violating line + action + the violating
        # nid resolved through the alias map to its real job_id. Deliberately
        # NOT the whole alias map — unrelated later jobs re-alias the trace,
        # and a persisting violation must keep ONE fingerprint across rounds.
        fingerprint = hashlib.sha256(
            json.dumps([view, line, action, job_id], sort_keys=True).encode()
        ).hexdigest()
        trace_lines = trace.splitlines()
        upto = line_no if 0 < line_no <= len(trace_lines) else len(trace_lines)
        tail = "\n".join(trace_lines[max(0, upto - _TAIL_LINES) : upto])
        return Violation(
            view=view,
            line_no=line_no,
            line=line,
            reason=reason,
            action=action,
            job_id=job_id,
            fingerprint=fingerprint,
            trace_tail=tail,
        )

    # -- reporting (gh) --------------------------------------------------

    def _report(self, violation: Violation) -> None:
        if self._dry_run:
            print(f"[dry-run] would file: {violation.title}")
            print(violation.body)
            return
        existing = self._find_issue(violation.fingerprint)
        if existing is None:
            self._run_gh(
                "issue",
                "create",
                "--title",
                violation.title,
                "--label",
                ISSUE_LABEL,
                "--body",
                violation.body,
            )
            _log.warning("filed issue for %s", violation.title)
            return
        # Never a duplicate issue: append a counter comment, at most once per
        # hour per fingerprint.
        key = COMMENT_KEY_PREFIX + violation.fingerprint
        last_raw = self._store.get_meta(key)
        now = self._now()
        if last_raw:
            try:
                if now - datetime.fromisoformat(last_raw) < _COMMENT_CAP:
                    return
            except ValueError:
                pass
        self._run_gh(
            "issue",
            "comment",
            str(existing),
            "--body",
            f"still violating as of {now.isoformat()} "
            f"({FINGERPRINT_MARKER} {violation.fingerprint})",
        )
        self._store.set_meta(key, now.isoformat())

    def _find_issue(self, fingerprint: str) -> int | None:
        proc = self._run_gh(
            "issue",
            "list",
            "--state",
            "open",
            "--label",
            ISSUE_LABEL,
            "--search",
            fingerprint,
            "--json",
            "number",
        )
        try:
            rows = json.loads(proc.stdout or "[]")
        except ValueError:
            return None
        if isinstance(rows, list) and rows:
            first = rows[0]
            if isinstance(first, dict) and "number" in first:
                return int(first["number"])
        return None

    def _run_gh(self, *args: str) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            [self._gh, *args], capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0:
            raise ValidatorError(
                f"gh {' '.join(args[:2])} failed: {proc.stderr.strip()[:300]}"
            )
        return proc

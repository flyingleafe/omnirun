"""Tests for the scoped progress-narration sink (progress.py)."""

from __future__ import annotations

from omnirun.progress import report, reporting


def test_report_is_noop_outside_a_scope() -> None:
    # Must not raise when no sink is installed (the daemon / plain library use).
    report("nobody is listening")


def test_reporting_captures_messages_in_order() -> None:
    seen: list[str] = []
    with reporting(seen.append):
        report("one")
        report("two")
    assert seen == ["one", "two"]


def test_sink_is_reset_after_scope() -> None:
    seen: list[str] = []
    with reporting(seen.append):
        report("inside")
    report("outside")  # sink reset -> dropped
    assert seen == ["inside"]


def test_nested_scopes_restore_the_outer_sink() -> None:
    outer: list[str] = []
    inner: list[str] = []
    with reporting(outer.append):
        report("a")
        with reporting(inner.append):
            report("b")
        report("c")
    assert outer == ["a", "c"]
    assert inner == ["b"]


def test_a_raising_sink_never_breaks_the_operation() -> None:
    def boom(_msg: str) -> None:
        raise RuntimeError("render failed")

    with reporting(boom):
        report("this must not propagate")  # no exception escapes

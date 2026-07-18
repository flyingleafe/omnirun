"""Group/matrix expansion (FUT-1) and the wire version handshake (CLI-6)."""

from __future__ import annotations

import pytest

from omnirun import wire
from omnirun.groups import expand_cells, make_group_name, parse_matrix
from omnirun.models import CodePlan, JobSpec, RepoRef
from omnirun.state.store import EventRow

_REPO = RepoRef(remote_url="", sha="a" * 40, branch="main", slug="proj")


def _base_spec() -> JobSpec:
    return JobSpec(
        job_id="base-000000",
        name="train",
        command="python train.py",
        repo=_REPO,
        env_vars={"BASE": "1"},
        code=CodePlan(kind="remote", clone_url="https://x/y.git", origin="o"),
        env_dotenv="SECRET=s\n",
    )


# --------------------------------------------------------------------------- matrix


def test_parse_matrix_cross_product() -> None:
    cells = parse_matrix("lr=0.1,0.3×seed=0,1")
    assert cells == [
        {"lr": "0.1", "seed": "0"},
        {"lr": "0.1", "seed": "1"},
        {"lr": "0.3", "seed": "0"},
        {"lr": "0.3", "seed": "1"},
    ]


def test_parse_matrix_star_separator_and_spaces() -> None:
    assert parse_matrix("a=1 * b=x, y") == [
        {"a": "1", "b": "x"},
        {"a": "1", "b": "y"},
    ]


def test_parse_matrix_single_dimension() -> None:
    assert parse_matrix("seed=1,2,3") == [{"seed": "1"}, {"seed": "2"}, {"seed": "3"}]


@pytest.mark.parametrize("bad", ["", "lr", "lr=", "=0.1", "a=1×a=2"])
def test_parse_matrix_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_matrix(bad)


def test_expand_cells_shares_plan_and_stamps_group() -> None:
    spec = _base_spec()
    cells = parse_matrix("lr=0.1,0.3")
    out = expand_cells(spec, cells, "sweep-1")
    assert len(out) == 2
    assert len({s.job_id for s in out}) == 2  # unique ids
    for s, cell in zip(out, cells):
        assert s.group == "sweep-1"
        assert s.code == spec.code  # ONE resolved plan, shared
        assert s.env_dotenv == spec.env_dotenv
        assert s.env_vars["BASE"] == "1"
        assert s.env_vars["lr"] == cell["lr"]
        assert "lr" in s.name
    # The base spec is untouched.
    assert spec.group is None and "lr" not in spec.env_vars


def test_expand_cells_count_mode() -> None:
    out = expand_cells(_base_spec(), [{} for _ in range(3)], "g")
    assert len(out) == 3
    assert len({s.job_id for s in out}) == 3
    assert all(s.name == "train" for s in out)


def test_make_group_name_is_safe_and_fresh() -> None:
    a, b = make_group_name("My Train!"), make_group_name("My Train!")
    assert a != b
    assert a.startswith("my-train-")


# --------------------------------------------------------------------------- handshake


def test_version_tuple() -> None:
    assert wire.version_tuple("0.5.18") == (0, 5, 18)
    assert wire.version_tuple("1.2") == (1, 2)
    assert wire.version_tuple("2.0.0rc1") == (2, 0)
    assert wire.version_tuple("garbage") == (0,)


def test_check_peer_version_compatible_and_absent() -> None:
    assert wire.check_peer_version(wire.PROTOCOL_VERSION, wire.MIN_SUPPORTED_PEER) is (
        None
    )
    assert wire.check_peer_version(None, None) is None


def test_check_peer_version_client_too_old() -> None:
    err = wire.check_peer_version("99.0", "99.0")
    assert err is not None and "upgrade the client" in err


def test_check_peer_version_daemon_too_old() -> None:
    err = wire.check_peer_version("0.0.1", None)
    assert err is not None and "daemon host" in err


def test_check_client_version() -> None:
    assert wire.check_client_version(None) is None
    assert wire.check_client_version(wire.PROTOCOL_VERSION) is None
    err = wire.check_client_version("0.0.1")
    assert err is not None and "upgrade the client" in err


def test_event_row_codec_roundtrip() -> None:
    ev = EventRow(
        id=7,
        job_id="j-1",
        seq=3,
        at="2026-07-01T00:00:00+00:00",
        actor="scheduler",
        action="reserve",
        cause=None,
        data={"provider": "prov", "est_cost": 1.5},
    )
    assert wire.event_row_from_json(wire.event_row_to_json(ev)) == ev

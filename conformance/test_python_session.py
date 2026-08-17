import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from quip_proto import session  # noqa: E402
from quip_proto import miner_pb2  # noqa: E402


def test_hello_requires_token(monkeypatch):
    monkeypatch.delenv("QUIP_SESSION_TOKEN", raising=False)
    with pytest.raises(session.MissingToken):
        session.build_hello("qpu-0", "dwave-qpu", "quantum-anneal", [miner_pb2.ISING_SAMPLE])
    monkeypatch.setenv("QUIP_SESSION_TOKEN", "tok-abc")
    h = session.build_hello("qpu-0", "dwave-qpu", "quantum-anneal", [miner_pb2.ISING_SAMPLE])
    assert h.session_token == "tok-abc"
    assert h.protocol_version == 1


def test_configure_defaults():
    c = miner_pb2.Configure()  # all zero
    cfg = session.session_config_from_configure("qpu-0", c)
    assert (cfg.queue_depth, cfg.idle_timeout_s, cfg.heartbeat_s, cfg.reconnect_window_s) == (3, 300, 15, 60)


def test_welcome_rejects_non_v1_protocol_version():
    session.check_welcome(miner_pb2.Welcome(protocol_version=1))
    with pytest.raises(session.BadWelcome) as ei:
        session.check_welcome(miner_pb2.Welcome(protocol_version=2))
    assert ei.value.version == 2


def test_documented_exit_codes():
    assert session.EXIT_CONFIG_INVALID == 64
    assert session.EXIT_ENV_INCOMPATIBLE == 69
    assert session.EXIT_INTERNAL_FATAL == 70
    assert session.EXIT_TOKEN_REJECTED == 77

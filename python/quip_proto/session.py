import os
from dataclasses import dataclass

from quip_proto import miner_pb2

# Protocol version this SDK speaks. Welcome.protocol_version must equal this.
PROTOCOL_VERSION = 1


class MissingToken(Exception):
    pass


class BadWelcome(Exception):
    """Welcome.protocol_version is not what this SDK speaks."""

    def __init__(self, version: int):
        self.version = version
        super().__init__(f"unexpected protocol version in Welcome: {version}")


# Sysexits-style process exit codes (also carried on Fatal.exit_code).
EXIT_CLEAN = 0
EXIT_CONFIG_INVALID = 64
EXIT_ENV_INCOMPATIBLE = 69
EXIT_INTERNAL_FATAL = 70
EXIT_TOKEN_REJECTED = 77


@dataclass
class SessionConfig:
    miner_id: str
    queue_depth: int
    idle_timeout_s: int
    heartbeat_s: int
    reconnect_window_s: int


def build_hello(miner_id, backend, algorithm, supported_kinds):
    token = os.environ.get("QUIP_SESSION_TOKEN")
    if not token:
        raise MissingToken("QUIP_SESSION_TOKEN unset")
    return miner_pb2.Hello(
        miner_id=miner_id,
        session_token=token,
        protocol_version=PROTOCOL_VERSION,
        backend=backend,
        algorithm=algorithm,
        supported_kinds=list(supported_kinds),
    )


def check_welcome(welcome) -> None:
    """Reject a Welcome whose protocol_version is not what this SDK speaks."""
    version = welcome.protocol_version
    if version != PROTOCOL_VERSION:
        raise BadWelcome(version)


def session_config_from_configure(miner_id, configure):
    def d(v, default):
        return default if v == 0 else v
    return SessionConfig(
        miner_id=miner_id,
        queue_depth=d(configure.queue_depth, 3),
        idle_timeout_s=d(configure.idle_timeout_s, 300),
        heartbeat_s=d(configure.heartbeat_s, 15),
        reconnect_window_s=d(configure.reconnect_window_s, 60),
    )

"""Log module for structured event capture during Tutte polynomial synthesis.

Provides a module-level singleton EventLog that any module in tutte/ can
import and use to record timestamped events.

Usage:
    from tutte.log.event_log import get_log, reset_log
    from tutte.log.event_types import EventType

    reset_log()
    engine.synthesize(graph)
    get_log().print_timeline()
"""

from tutte.logs.event_log import EventLog, get_log, reset_log
from tutte.logs.event_types import Event, EventType, LogLevel

__all__ = [
    "EventLog",
    "EventType",
    "Event",
    "LogLevel",
    "get_log",
    "reset_log",
]

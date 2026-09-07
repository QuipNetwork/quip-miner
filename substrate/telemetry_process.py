"""Telemetry sibling process — sole telemetry surface for quip-miner.

`telemetry_main` is the child entry: it runs an asyncio loop hosting
aiohttp.web with the full `/api/v1/*` route surface that dashboards
consume. State that lives in the controller process (counters,
hardware descriptor, miner survey, identity) is delivered via the
stats snapshot file the controller writes via
`shared.stats_snapshot.StatsSnapshotWriter`; chain queries are issued
from this process's own `SubstrateClient` with URL failover.

Process isolation rationale: telemetry serves HTTP requests of
arbitrary cost (operator dashboards, ad-hoc queries). Co-locating
that on the controller's main loop risks starving the event drain
during request spikes. The original silent-subscription-death bug
taught us to be paranoid about shared event loops; telemetry
isolation is the follow-through.

Response envelope: every handler returns either
``{"success": True, "data": ..., "timestamp": int}`` or
``{"success": False, "error": str, "code": str, "timestamp": int}``.
Dashboards depend on this exact shape; do not change it.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import multiprocessing.synchronize
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

from aiohttp import web
from aiohttp.web import middleware

from shared.mining_attempt_log import (
    DEFAULT_LOG_DIR,
    query_by_solution_id,
    query_by_solution_number,
    query_stored_solutions,
)
from shared.stats_snapshot import (
    merge_snapshots,
    read_all_snapshots,
    read_snapshot,
)
from shared.version import get_version
from substrate.client import SubstrateClient
from substrate.url_failover import AllUrlsDown, SubstrateUrlFailover

logger = logging.getLogger(__name__)


# Snapshot age (seconds) under which we consider the controller "alive"
# for the legacy `is_mining` field on `/api/v1/status`. The writer
# interval is 1s, so 5s gives slack for transient FS hiccups and slow
# scheduler ticks.
_SNAPSHOT_FRESHNESS_S = 5.0

# Snapshot age (seconds) above which the aggregator drops a per-kind
# file from the merged view entirely. A controller that stops (config
# change, restart, crash) leaves its snapshot behind; without this gate
# telemetry keeps reporting that dead child's miners/modes forever. Set
# well above the writer's 1s interval so a briefly-stalled live child
# is never flapped out of the inventory.
_STALE_SNAPSHOT_MAX_AGE_S = 30.0


def telemetry_main(
    listen_host: str,
    listen_port: int,
    validator_urls: list[str],
    shutdown_event: mp.synchronize.Event,
    *,
    stats_snapshot_path: Optional[str] = None,
    snapshot_dir: Optional[str] = None,
) -> None:
    """Child process entry point for telemetry.

    Args:
        listen_host: aiohttp bind address.
        listen_port: aiohttp listen port.
        validator_urls: URLs to try for direct chain queries. Telemetry
            owns its own `SubstrateClient`; no pool/event-manager
            involvement.
        shutdown_event: Set by the parent (or by SIGTERM handler) to
            request graceful exit.
        stats_snapshot_path: single-snapshot mode — file path the
            controller writes via ``StatsSnapshotWriter``. Used by the
            legacy single-controller spawn (controller spawns sibling
            on its own port).
        snapshot_dir: multi-snapshot aggregator mode — directory the
            entrypoint creates and per-mode controllers write to. The
            sibling globs ``telemetry-stats-*.json`` here and merges
            via ``merge_snapshots`` on every API hit. Used by the
            Docker multi-process supervisor (one telemetry sibling
            per container, N controller children).

    Exactly one of ``stats_snapshot_path`` / ``snapshot_dir`` must be
    provided. The aggregator mode degrades gracefully when individual
    snapshots are missing or mid-write — those snapshots are skipped
    for the duration of one request.
    """
    if (stats_snapshot_path is None) == (snapshot_dir is None):
        raise ValueError(
            "telemetry_main: pass exactly one of stats_snapshot_path "
            "(single-snapshot legacy mode) or snapshot_dir (aggregator)"
        )

    def _sigterm(signum, frame):
        logger.info("telemetry process received SIGTERM")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _sigterm)

    asyncio.run(_run(
        listen_host=listen_host,
        listen_port=listen_port,
        stats_snapshot_path=Path(stats_snapshot_path) if stats_snapshot_path else None,
        snapshot_dir=Path(snapshot_dir) if snapshot_dir else None,
        validator_urls=validator_urls,
        shutdown_event=shutdown_event,
    ))


# ----------------------------------------------------------------------
# Middlewares
# ----------------------------------------------------------------------


@middleware
async def _cors_middleware(request: web.Request, handler) -> web.Response:
    """Permissive CORS for the dev dashboard. Tighten if exposed externally."""
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        try:
            response = await handler(request)
        except web.HTTPException as exc:
            response = exc
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


@middleware
async def _error_middleware(request: web.Request, handler) -> web.Response:
    """Convert unhandled exceptions into the legacy error envelope."""
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "unhandled exception in handler for %s %s: %s",
            request.method,
            request.path,
            exc,
        )
        return web.json_response(
            {
                "success": False,
                "error": str(exc),
                "code": "INTERNAL_ERROR",
                "timestamp": int(time.time()),
            },
            status=500,
        )


# ----------------------------------------------------------------------
# Run loop
# ----------------------------------------------------------------------


async def _run(
    listen_host: str,
    listen_port: int,
    stats_snapshot_path: Optional[Path],
    snapshot_dir: Optional[Path],
    validator_urls: list[str],
    shutdown_event: mp.synchronize.Event,
) -> None:
    """Set up the aiohttp app, connect a `SubstrateClient`, serve until shutdown."""
    # No validator URLs is a legitimate config (snapshot-only deployments,
    # tests). The chain-backed endpoints degrade to 503; everything that
    # reads the snapshot keeps working.
    failover: Optional[SubstrateUrlFailover] = None
    if validator_urls:
        failover = SubstrateUrlFailover(
            validator_urls, initial_backoff_s=1.0, max_backoff_s=60.0,
        )

    client: Optional[SubstrateClient] = None
    if failover is not None:
        client = SubstrateClient(url=failover.current())
        try:
            await client.connect()
        except Exception as exc:  # noqa: BLE001 — initial connect best-effort
            logger.warning(
                "telemetry: initial SubstrateClient connect to %s failed (%s: %s); "
                "chain-backed endpoints will reconnect on first request",
                failover.current(),
                type(exc).__name__,
                exc,
            )

    started_at = time.time()

    app = web.Application(middlewares=[_cors_middleware, _error_middleware])
    # Exactly one of these is set; `_read_snapshot_or_503` picks the
    # right path. Aggregator mode (snapshot_dir) merges every per-mode
    # file the children write; legacy mode (stats_snapshot_path) reads
    # a single file as before.
    app["stats_snapshot_path"] = stats_snapshot_path
    app["snapshot_dir"] = snapshot_dir
    app["failover"] = failover
    app["client"] = client
    app["started_at"] = started_at

    app.router.add_get("/health", _handle_health)
    app.router.add_get("/api/v1/status", _handle_status)
    app.router.add_get("/api/v1/system", _handle_system)
    app.router.add_get("/api/v1/miner/survey", _handle_miner_survey)
    app.router.add_get("/api/v1/stats", _handle_stats)
    app.router.add_get("/api/v1/block/latest", _handle_block_latest)
    app.router.add_get("/api/v1/block/{block_number}", _handle_block)
    app.router.add_get(
        "/api/v1/block/{block_number}/header", _handle_block_header,
    )
    app.router.add_post("/api/v1/solve", _handle_solve)
    app.router.add_get("/api/v1/mining/attempts", _handle_mining_attempts)
    app.router.add_get("/api/v1/mining/solutions", _handle_mining_solutions)
    app.router.add_get("/", _handle_index)
    app.router.add_route("*", "/{path:.*}", _handle_not_found)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, listen_host, listen_port)
    await site.start()

    logger.info("telemetry process listening on %s:%d", listen_host, listen_port)

    try:
        while not shutdown_event.is_set():
            await asyncio.sleep(0.1)
    finally:
        logger.info("telemetry process shutting down")
        await runner.cleanup()
        if client is not None:
            try:
                await client.close()
            except Exception:  # noqa: BLE001 — cleanup best-effort
                logger.exception("telemetry: SubstrateClient close failed")


# ----------------------------------------------------------------------
# Response helpers
# ----------------------------------------------------------------------


def _success(data: Any, status: int = 200) -> web.Response:
    """Wrap `data` in the success envelope dashboards expect."""
    return web.json_response(
        {"success": True, "data": data, "timestamp": int(time.time())},
        status=status,
    )


def _error(message: str, code: str, status: int = 400) -> web.Response:
    """Wrap an error in the failure envelope dashboards expect."""
    return web.json_response(
        {
            "success": False,
            "error": message,
            "code": code,
            "timestamp": int(time.time()),
        },
        status=status,
    )


def _snapshot_freshest_mtime(request: web.Request) -> float:
    """Most-recent mtime across the snapshot source(s), 0.0 if unknown.

    In single-snapshot mode this is just `stats_snapshot_path.stat().st_mtime`.
    In aggregator mode we take the MAX mtime across every per-mode file —
    `is_mining` should report True as long as *any* child is writing
    snapshots (a dead cpu child shouldn't mark the gpu child as offline).
    """
    snapshot_dir: Optional[Path] = request.app["snapshot_dir"]
    if snapshot_dir is not None:
        pattern = "telemetry-stats-*.json"
        latest = 0.0
        for path in snapshot_dir.glob(pattern):
            try:
                mt = path.stat().st_mtime
            except OSError:
                continue
            if mt > latest:
                latest = mt
        return latest
    path: Optional[Path] = request.app["stats_snapshot_path"]
    if path is None:
        return 0.0
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _read_snapshot_or_503(
    request: web.Request,
) -> Tuple[Optional[dict], Optional[web.Response]]:
    """Read the stats snapshot or return a (None, 503-response) tuple.

    In single-snapshot mode (legacy) reads the path the controller
    writes. In aggregator mode (Docker entrypoint) globs every
    per-mode file in the snapshot dir and merges them — handlers see
    a single virtual snapshot with summed counters + per-mode
    breakdown under the new `modes` key.
    """
    snapshot_dir: Optional[Path] = request.app["snapshot_dir"]
    if snapshot_dir is not None:
        snaps = read_all_snapshots(snapshot_dir, max_age_s=_STALE_SNAPSHOT_MAX_AGE_S)
        snap = merge_snapshots(snaps)
    else:
        path: Path = request.app["stats_snapshot_path"]
        snap = read_snapshot(path)
    if snap is None:
        return None, _error(
            "stats snapshot not yet available",
            "STATS_UNAVAILABLE",
            status=503,
        )
    return snap, None


async def _get_client(request: web.Request) -> Optional[SubstrateClient]:
    """Return the cached SubstrateClient, reconnecting if needed.

    Returns ``None`` if no validator URLs were configured. On connection
    failure the failover is advanced and a new client is wired in; the
    next handler call retries from there.
    """
    client: Optional[SubstrateClient] = request.app["client"]
    failover: Optional[SubstrateUrlFailover] = request.app["failover"]
    if client is None or failover is None:
        return None
    if client._iface is not None:  # noqa: SLF001 — already connected
        return client
    try:
        await client.connect()
        return client
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "telemetry: SubstrateClient reconnect to %s failed (%s: %s); "
            "rotating to next validator",
            failover.current(),
            type(exc).__name__,
            exc,
        )
        try:
            next_url = failover.advance_after_failure(failover.current())
        except AllUrlsDown:
            return None
        new_client = SubstrateClient(url=next_url)
        try:
            await new_client.connect()
        except Exception as exc2:  # noqa: BLE001
            logger.warning(
                "telemetry: SubstrateClient connect to %s also failed (%s: %s)",
                next_url, type(exc2).__name__, exc2,
            )
            return None
        request.app["client"] = new_client
        return new_client


async def _require_client(
    request: web.Request,
) -> Tuple[Optional[SubstrateClient], Optional[web.Response]]:
    """Return ``(client, None)`` when a chain client is available, or
    ``(None, 503-error-response)`` when it is not.

    Matches the ``(value, error)`` convention used throughout this module.
    """
    client = await _get_client(request)
    if client is None:
        return None, _error("chain client unavailable", "CHAIN_UNAVAILABLE", status=503)
    return client, None


# ----------------------------------------------------------------------
# Handlers
# ----------------------------------------------------------------------


async def _handle_health(request: web.Request) -> web.Response:
    """Liveness probe — always returns the success envelope."""
    return _success({"status": "ok"})


async def _handle_index(request: web.Request) -> web.Response:
    """Endpoint directory page used as a smoke probe by dashboards."""
    return _success(
        {
            "service": "quip-miner",
            "version": get_version(),
            "endpoints": [
                {"GET": "/health"},
                {"GET": "/api/v1/status"},
                {"GET": "/api/v1/system"},
                {"GET": "/api/v1/miner/survey"},
                {"GET": "/api/v1/stats"},
                {"GET": "/api/v1/block/latest"},
                {"GET": "/api/v1/block/{block_number}"},
                {"GET": "/api/v1/block/{block_number}/header"},
                {"POST": "/api/v1/solve"},
                {"GET": "/api/v1/mining/attempts"},
                {"GET": "/api/v1/mining/solutions"},
            ],
        }
    )


async def _handle_not_found(request: web.Request) -> web.Response:
    """Catch-all 404 wrapped in the failure envelope."""
    return _error("not found", "NOT_FOUND", status=404)


async def _handle_stats(request: web.Request) -> web.Response:
    """Return the latest stats snapshot the controller has written.

    The snapshot is the controller's serialized counters + identity
    (see ``build_stats_snapshot_for_telemetry``); dashboards read it
    verbatim under ``data``.
    """
    snapshot, error_resp = _read_snapshot_or_503(request)
    if error_resp is not None:
        return error_resp
    return _success(snapshot)


async def _handle_system(request: web.Request) -> web.Response:
    """Return the hardware descriptor from the controller's snapshot."""
    snapshot, error_resp = _read_snapshot_or_503(request)
    if error_resp is not None:
        return error_resp
    descriptor = snapshot.get("descriptor")
    if descriptor is None:
        return _error(
            "descriptor not present in snapshot",
            "DESCRIPTOR_UNAVAILABLE",
            status=503,
        )
    return _success(descriptor)


async def _handle_miner_survey(request: web.Request) -> web.Response:
    """Return the versioned `MinerSurveyV1` payload from the snapshot.

    Schema: ``quip.miner_survey.v1``. Prefer this over
    ``/api/v1/system`` — survey shape is stable; descriptor is whatever
    the legacy builder produces.
    """
    snapshot, error_resp = _read_snapshot_or_503(request)
    if error_resp is not None:
        return error_resp
    survey = snapshot.get("miner_survey")
    if survey is None:
        return _error(
            "miner_survey not present in snapshot",
            "SURVEY_UNAVAILABLE",
            status=503,
        )
    return _success(survey)


def _epoch_to_iso(ts: Any) -> Optional[str]:
    """Render an epoch-seconds value as an ISO-8601 UTC string, or None.

    The controller snapshot stores ``last_successful_submission`` as a
    ``time.time()`` float; gh-27 asks for an ISO timestamp on the status
    endpoint. Anything that is not a finite number maps to None so a
    corrupt or missing value never breaks the whole status response.
    """
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


async def _handle_status(request: web.Request) -> web.Response:
    """Aggregate chain head + miner identity status.

    Identity (`ss58_address`, `node_id`, `miners`, `account_id_hex`)
    comes from the controller's snapshot. The chain bits
    (`head_hash`, `head_number`, `miner_registered`, `miner_info`)
    are fetched from this process's own `SubstrateClient`. `is_mining`
    is inferred from snapshot file mtime (within
    ``_SNAPSHOT_FRESHNESS_S``).
    """
    snapshot, error_resp = _read_snapshot_or_503(request)
    if error_resp is not None:
        return error_resp

    snapshot_mtime = _snapshot_freshest_mtime(request)
    is_mining = (time.time() - snapshot_mtime) < _SNAPSHOT_FRESHNESS_S

    started_at = request.app.get("started_at") or time.time()
    uptime_seconds = int(time.time() - started_at)

    chain_payload: dict[str, Any] = {}
    miner_registered = False
    miner_info_dict: Optional[dict] = None

    client = await _get_client(request)
    account_hex = snapshot.get("account_id_hex")
    if client is not None:
        try:
            head_hash = await client.get_head()
            head_number = await client.get_block_number(at=head_hash)
            chain_payload = {
                "head_hash": "0x" + head_hash.hex(),
                "head_number": head_number,
            }
            if account_hex:
                account_bytes = bytes.fromhex(
                    account_hex[2:] if account_hex.startswith("0x") else account_hex
                )
                miner_info = await client.query_miner(account_bytes)
                if miner_info is not None:
                    miner_registered = True
                    miner_info_dict = _miner_info_dict(miner_info)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "telemetry status: chain query failed (%s: %s)",
                type(exc).__name__, exc,
            )

    last_submit_epoch = snapshot.get("controller", {}).get(
        "last_successful_submission"
    )

    return _success(
        {
            "ss58_address": snapshot.get("ss58_address"),
            "account_id_hex": account_hex,
            "node_id": snapshot.get("node_id"),
            "is_mining": is_mining,
            # ISO-8601 UTC timestamp of the last *accepted* submit, or None
            # if the node has never landed one (gh-27). A node reporting
            # is_mining True with a stale/None value here is mining but not
            # winning (QUI-829 / gh-18) — the one field that makes that
            # self-evident. The raw epoch value is kept alongside for
            # consumers that compute staleness arithmetically.
            "last_successful_submission": _epoch_to_iso(last_submit_epoch),
            "last_successful_submission_epoch": last_submit_epoch,
            # Consecutive failed submits since the last landed proof, and a
            # reason string when this build is too old for the chain runtime.
            # A large counter or a non-null reason means "mining but landing
            # nothing" even while is_mining reads true (gh-20).
            "consecutive_submit_failures": snapshot.get("controller", {}).get(
                "consecutive_submit_failures"
            ),
            "runtime_incompatible": snapshot.get("controller", {}).get(
                "runtime_incompatible"
            ),
            "uptime_seconds": uptime_seconds,
            "chain": chain_payload,
            "miner_registered": miner_registered,
            "miner_info": miner_info_dict,
            "miners": snapshot.get("miners", []),
            # Node sync progress (pool sync-wait) — None when healthy.
            "sync_state": snapshot.get("sync_state"),
            # Per-mode breakdown — populated in aggregator mode (one
            # entry per active backend group: cpu / gpu / qpu); empty
            # dict in legacy single-process mode where there's only
            # one set of counters and no need to distinguish them.
            "modes": snapshot.get("modes", {}),
        }
    )


async def _handle_block_latest(request: web.Request) -> web.Response:
    """Return the chain head block (header + extrinsic count)."""
    client, err = await _require_client(request)
    if err is not None:
        return err
    try:
        head_hash = await client.get_head()
        return _success(await _block_payload(client, at=head_hash))
    except Exception as exc:  # noqa: BLE001
        return _error(f"chain query failed: {exc}", "CHAIN_ERROR", status=502)


async def _fetch_block_payload(
    request: web.Request, block_number: int,
) -> Tuple[Optional[dict], Optional[web.Response]]:
    """Resolve *block_number* to its payload, returning ``(payload, None)`` on
    success or ``(None, error_response)`` on any failure."""
    client, err = await _require_client(request)
    if err is not None:
        return None, err
    try:
        block_hash = await _block_hash_for_number(client, block_number)
        if block_hash is None:
            return None, _error(
                f"block {block_number} not found",
                "BLOCK_NOT_FOUND",
                status=404,
            )
        return await _block_payload(client, at=block_hash), None
    except Exception as exc:  # noqa: BLE001
        return None, _error(f"chain query failed: {exc}", "CHAIN_ERROR", status=502)


async def _handle_block(request: web.Request) -> web.Response:
    """Return the block at a given block number."""
    block_number, err = _parse_block_number(request)
    if err is not None:
        return err
    payload, err = await _fetch_block_payload(request, block_number)
    if err is not None:
        return err
    return _success(payload)


async def _handle_block_header(request: web.Request) -> web.Response:
    """Return just the header + hash for the block at a given number."""
    block_number, err = _parse_block_number(request)
    if err is not None:
        return err
    payload, err = await _fetch_block_payload(request, block_number)
    if err is not None:
        return err
    return _success({"header": payload.get("header"), "hash": payload.get("hash")})


async def _handle_solve(request: web.Request) -> web.Response:
    """POST /api/v1/solve — disabled on quip-miner.

    Returns 503 with code ``SOLVE_DISABLED``; operators that need a
    direct DWave sampler should deploy a dedicated solve service.
    """
    return _error(
        "/api/v1/solve is not enabled on quip-miner; deploy a dedicated "
        "solve service for direct DWave sampling",
        "SOLVE_DISABLED",
        status=503,
    )


def _attempts_dir_from_snapshot(request: web.Request) -> Path:
    """Resolve the attempt/solution JSONL dir from the controller snapshot.

    Falls back to ``DEFAULT_LOG_DIR`` when no snapshot (or no ``attempts_dir``
    field) is available. Shared by the attempts and solutions handlers, which
    both ignore the 503 error response.
    """
    snapshot, _err = _read_snapshot_or_503(request)
    return (
        Path(snapshot["attempts_dir"])
        if snapshot is not None and snapshot.get("attempts_dir")
        else DEFAULT_LOG_DIR
    )


async def _handle_mining_attempts(request: web.Request) -> web.Response:
    """GET /api/v1/mining/attempts — query the attempt + submission log.

    Query params (exactly one of the first two must be supplied):
      - ``solution_number`` (int; ``solution_id`` accepted as an alias):
        returns ``{"submission": {...}, "attempts": [...]}`` — the
        submission record for that solution and the winning miner's attempts.
      - ``miner_id`` + ``solution_number``: returns just the attempt list
        for that (miner, solution).
      - ``limit`` (int, default 1000): caps the attempts list.

    The store is the JSONL files under ``~/.quip-miner/mining_attempts/``
    (overridable via the ``attempts_dir`` field of the controller's
    snapshot) produced by the worker and controller. See
    ``shared/mining_attempt_log.py`` for the schema.
    """
    attempts_dir = _attempts_dir_from_snapshot(request)

    params = request.rel_url.query
    try:
        limit = int(params.get("limit", "1000"))
    except ValueError:
        return _error("limit must be an integer", "BAD_PARAM")
    if limit < 1 or limit > 100_000:
        return _error("limit out of range (1..100000)", "BAD_PARAM")

    sol_raw = params.get("solution_number", params.get("solution_id"))
    miner_id = params.get("miner_id")

    if miner_id and sol_raw is not None:
        try:
            solution_number = int(sol_raw)
        except ValueError:
            return _error("solution_number must be an integer", "BAD_PARAM")
        attempts = query_by_solution_number(
            miner_id, solution_number, log_dir=attempts_dir, limit=limit,
        )
        return _success({"attempts": attempts})

    if sol_raw is not None:
        try:
            solution_number = int(sol_raw)
        except ValueError:
            return _error("solution_number must be an integer", "BAD_PARAM")
        result = query_by_solution_id(solution_number, log_dir=attempts_dir)
        if result is None:
            return _error(
                f"solution_number {solution_number} not found",
                "NOT_FOUND",
                status=404,
            )
        result["attempts"] = result["attempts"][:limit]
        return _success(result)

    return _error(
        "supply ?solution_number=N or ?miner_id=X&solution_number=N",
        "BAD_PARAM",
    )


async def _handle_mining_solutions(request: web.Request) -> web.Response:
    """GET /api/v1/mining/solutions — list archived top-5 spin configs.

    Query params:
      - ``solution_number`` (int; ``solution_id`` accepted as an alias):
        returns the stored solutions for that solution across all miners
        (or filtered by ``miner_id``).
      - ``miner_id`` (str, optional): filters to one miner's stored
        solutions.

    Returns ``{stored: [{iter, nonce_hex, salt_hex, result_kind,
    top_5_solutions_hex, top_5_energies}, ...]}`` sorted by iter.

    ``top_5_solutions_hex`` are 1-bit-per-spin packed (4578 nodes →
    1146 hex chars / solution). Decode with numpy.unpackbits +
    transformation 0→-1, 1→+1 to recover Ising spin vectors.
    """
    attempts_dir = _attempts_dir_from_snapshot(request)

    params = request.rel_url.query
    miner_id_filter = params.get("miner_id")
    sol_raw = params.get("solution_number", params.get("solution_id"))

    if sol_raw is not None:
        try:
            solution_number = int(sol_raw)
        except ValueError:
            return _error("solution_number must be an integer", "BAD_PARAM")
        stored = query_stored_solutions(
            solution_number, log_dir=attempts_dir, miner_id=miner_id_filter,
        )
        return _success(
            {"solution_number": solution_number, "stored": stored}
        )

    return _error("supply ?solution_number=N", "BAD_PARAM")


# ----------------------------------------------------------------------
# Substrate-backed helpers
# ----------------------------------------------------------------------


async def _block_hash_for_number(
    client: SubstrateClient, block_number: int,
) -> Optional[bytes]:
    """Resolve a block number to its on-chain block hash, or None."""

    def _resolve() -> Optional[str]:
        return client._iface.get_block_hash(block_id=block_number)  # noqa: SLF001

    result = await client._run(_resolve)  # noqa: SLF001
    if not result or result in ("0x" + "00" * 32, None):
        return None
    h = result[2:] if result.startswith("0x") else result
    try:
        return bytes.fromhex(h)
    except ValueError:
        return None


async def _block_payload(client: SubstrateClient, *, at: bytes) -> dict:
    """Substrate block → legacy-shaped JSON dict.

    Substrate-interface's ``get_block`` returns a structure that mixes
    plain dicts with `ScaleType` wrappers; ``_to_jsonable`` coerces
    those to JSON-safe primitives. We expose the header, hash, and
    extrinsic count — dashboards that consumed the v0.1 chain-block
    fields directly need updating.
    """

    def _query() -> dict:
        return client._iface.get_block(block_hash="0x" + at.hex())  # noqa: SLF001

    block = await client._run(_query)  # noqa: SLF001
    header = block.get("header", {}) if isinstance(block, dict) else {}
    extrinsics = block.get("extrinsics", []) if isinstance(block, dict) else []
    return {
        "hash": "0x" + at.hex(),
        "header": _to_jsonable(header),
        "extrinsic_count": len(extrinsics or []),
    }


def _to_jsonable(obj: Any) -> Any:
    """Recursively coerce substrate-interface output to JSON-safe primitives.

    ``SubstrateInterface.get_block`` returns dicts whose leaves can be
    raw ``ScaleType`` instances (decoded ``scale_info::N`` objects);
    ``json.dumps`` rejects those. Bytes render as ``0x`` hex to match
    the rest of the API surface. Falls back to ``repr()`` so the
    endpoint stays observable even if substrate-interface changes its
    return shape.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return "0x" + bytes(obj).hex()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "value"):
        val = obj.value
        if val is not obj:
            return _to_jsonable(val)
    return repr(obj)


def _parse_block_number(
    request: web.Request,
) -> Tuple[Optional[int], Optional[web.Response]]:
    """Extract `block_number` from URL match info as a non-negative int.

    Returns ``(block_number, None)`` on success, ``(None, error)``
    on bad input.
    """
    raw = request.match_info.get("block_number", "")
    try:
        block_number = int(raw)
    except ValueError:
        return None, _error(
            "block_number must be an integer",
            "INVALID_BLOCK_NUMBER",
        )
    if block_number < 0:
        return None, _error(
            "block_number must be non-negative",
            "INVALID_BLOCK_NUMBER",
        )
    return block_number, None


def _miner_info_dict(info) -> dict:
    """Map ``substrate.types.MinerInfo`` to a JSON-safe dict."""
    return {
        "registered_at": info.registered_at,
        "deposit": info.deposit,
        "proofs_submitted": info.proofs_submitted,
        "proofs_won": info.proofs_won,
        "rewards_earned": info.rewards_earned,
    }


__all__ = ["telemetry_main"]

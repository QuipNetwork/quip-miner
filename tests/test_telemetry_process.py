"""Tests for substrate.telemetry_process.telemetry_main.

The telemetry sibling process owns its own aiohttp app, a SubstrateClient
with URL failover, and serves /api/v1/stats from the file the controller
writes via StatsSnapshotWriter.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import socket
import time
from pathlib import Path



def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_telemetry_process_starts_and_serves_stats(tmp_path: Path):
    """Spawn telemetry process; verify /api/v1/stats returns the file contents."""
    from substrate.telemetry_process import telemetry_main

    stats_path = tmp_path / "telemetry-stats.json"
    stats_path.write_text(json.dumps(
        {"controller": {"heads_observed": 42, "active_url": "http://a"}}
    ))

    port = _free_port()
    shutdown_event = mp.Event()
    proc = mp.Process(
        target=telemetry_main,
        kwargs={
            "listen_host": "127.0.0.1",
            "listen_port": port,
            "stats_snapshot_path": str(stats_path),
            "validator_urls": ["http://example.invalid"],  # not actually used in this test
            "shutdown_event": shutdown_event,
        },
    )
    proc.start()
    try:
        # Wait for server to come up
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                import urllib.request
                resp = urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/v1/stats", timeout=0.5,
                ).read()
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("telemetry process did not start in 5s")

        data = json.loads(resp)
        assert data["success"] is True
        assert data["data"]["controller"]["heads_observed"] == 42
        assert data["data"]["controller"]["active_url"] == "http://a"
    finally:
        shutdown_event.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join()


def test_telemetry_status_passes_through_sync_state(tmp_path: Path):
    """/api/v1/status surfaces the snapshot's sync_state verbatim."""
    from substrate.telemetry_process import telemetry_main

    stats_path = tmp_path / "telemetry-stats.json"
    stats_path.write_text(json.dumps({
        "controller": {"heads_observed": 1},
        "sync_state": {"is_syncing": True, "current_block": 5_000, "url": "ws://a"},
    }))

    port = _free_port()
    shutdown_event = mp.Event()
    proc = mp.Process(
        target=telemetry_main,
        kwargs={
            "listen_host": "127.0.0.1",
            "listen_port": port,
            "stats_snapshot_path": str(stats_path),
            "validator_urls": ["http://example.invalid"],
            "shutdown_event": shutdown_event,
        },
    )
    proc.start()
    try:
        import urllib.request
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                resp = urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/v1/status", timeout=0.5,
                ).read()
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("telemetry process did not start in 5s")

        data = json.loads(resp)
        assert data["success"] is True
        assert data["data"]["sync_state"] == {
            "is_syncing": True,
            "current_block": 5_000,
            "url": "ws://a",
        }
    finally:
        shutdown_event.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join()


def test_epoch_to_iso_renders_utc_or_none():
    """Epoch floats become ISO-8601 UTC; missing/invalid values become None."""
    from substrate.telemetry_process import _epoch_to_iso

    assert _epoch_to_iso(0) == "1970-01-01T00:00:00+00:00"
    assert _epoch_to_iso(1_700_000_000.5) == "2023-11-14T22:13:20.500000+00:00"
    assert _epoch_to_iso(None) is None
    assert _epoch_to_iso("1700000000") is None
    assert _epoch_to_iso(True) is None
    assert _epoch_to_iso(float("nan")) is None
    assert _epoch_to_iso(1e20) is None


def test_telemetry_status_reports_last_successful_submission_as_iso(tmp_path: Path):
    """/api/v1/status renders last_successful_submission as ISO-8601 (gh-27).

    The controller snapshot carries the raw ``time.time()`` value; the
    endpoint must expose it as an ISO timestamp (None before the first
    accepted submission) and keep the epoch alongside.
    """
    from substrate.telemetry_process import telemetry_main

    epoch = 1_700_000_000.0
    stats_path = tmp_path / "telemetry-stats.json"
    stats_path.write_text(json.dumps({
        "controller": {
            "heads_observed": 1,
            "last_successful_submission": epoch,
        },
    }))

    port = _free_port()
    shutdown_event = mp.Event()
    proc = mp.Process(
        target=telemetry_main,
        kwargs={
            "listen_host": "127.0.0.1",
            "listen_port": port,
            "stats_snapshot_path": str(stats_path),
            "validator_urls": ["http://example.invalid"],
            "shutdown_event": shutdown_event,
        },
    )
    proc.start()
    try:
        import urllib.request
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                resp = urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/v1/status", timeout=0.5,
                ).read()
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("telemetry process did not start in 5s")

        data = json.loads(resp)["data"]
        assert data["last_successful_submission"] == "2023-11-14T22:13:20+00:00"
        assert data["last_successful_submission_epoch"] == epoch

        # Before the first accepted submission both fields are null.
        stats_path.write_text(json.dumps({"controller": {"heads_observed": 1}}))
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/v1/status", timeout=2,
        ).read()
        data = json.loads(resp)["data"]
        assert data["last_successful_submission"] is None
        assert data["last_successful_submission_epoch"] is None
    finally:
        shutdown_event.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join()


def test_telemetry_process_returns_503_when_snapshot_missing(tmp_path: Path):
    """If the snapshot file doesn't exist yet, /api/v1/stats returns 503."""
    from substrate.telemetry_process import telemetry_main

    port = _free_port()
    shutdown_event = mp.Event()
    proc = mp.Process(
        target=telemetry_main,
        kwargs={
            "listen_host": "127.0.0.1",
            "listen_port": port,
            "stats_snapshot_path": str(tmp_path / "missing.json"),
            "validator_urls": ["http://example.invalid"],
            "shutdown_event": shutdown_event,
        },
    )
    proc.start()
    try:
        deadline = time.time() + 5.0
        last_status = None
        while time.time() < deadline:
            try:
                import urllib.request
                import urllib.error
                try:
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/api/v1/stats", timeout=0.5,
                    )
                    last_status = 200
                except urllib.error.HTTPError as e:
                    last_status = e.code
                break
            except Exception:
                time.sleep(0.1)
        assert last_status == 503
    finally:
        shutdown_event.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join()


def test_telemetry_process_responds_to_shutdown_event(tmp_path: Path):
    """Setting shutdown_event causes the process to exit cleanly."""
    from substrate.telemetry_process import telemetry_main

    stats_path = tmp_path / "telemetry-stats.json"
    stats_path.write_text("{}")
    port = _free_port()
    shutdown_event = mp.Event()
    proc = mp.Process(
        target=telemetry_main,
        kwargs={
            "listen_host": "127.0.0.1",
            "listen_port": port,
            "stats_snapshot_path": str(stats_path),
            "validator_urls": ["http://example.invalid"],
            "shutdown_event": shutdown_event,
        },
    )
    proc.start()
    try:
        time.sleep(0.5)  # let it come up
        shutdown_event.set()
        proc.join(timeout=5)
        assert not proc.is_alive()
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join()


def test_telemetry_process_aggregator_mode_merges_per_kind_snapshots(tmp_path: Path):
    """End-to-end: start telemetry_main in aggregator mode with two
    per-kind snapshot files; verify /api/v1/stats returns the merged
    view (summed counters, unioned miners, per-mode breakdown)."""
    from shared.stats_snapshot import snapshot_filename_for
    from substrate.telemetry_process import telemetry_main

    snap_dir = tmp_path / "runtime"
    snap_dir.mkdir()
    (snap_dir / snapshot_filename_for("cpu")).write_text(json.dumps({
        "mode": "cpu",
        "controller": {"heads_observed": 50, "proofs_submitted": 3, "active_url": "ws://a"},
        "node_id": "rig-01",
        "ss58_address": "5GPP",
        "miners": [{"id": "rig-CPU-1", "type": "CPU"}],
        "descriptor": {"cpus": 8},
        "miner_survey": {"v": "quip.miner_survey.v1"},
    }))
    (snap_dir / snapshot_filename_for("qpu")).write_text(json.dumps({
        "mode": "qpu",
        "controller": {"heads_observed": 50, "proofs_submitted": 2, "active_url": "ws://b"},
        "node_id": "rig-01",
        "ss58_address": "5GPP",
        "miners": [{"id": "rig-QPU-DWAVE-1", "type": "QPU"}],
        "descriptor": {"cpus": 8},
        "miner_survey": {"v": "quip.miner_survey.v1"},
    }))

    port = _free_port()
    shutdown_event = mp.Event()
    proc = mp.Process(
        target=telemetry_main,
        kwargs={
            "listen_host": "127.0.0.1",
            "listen_port": port,
            "snapshot_dir": str(snap_dir),
            "validator_urls": [],
            "shutdown_event": shutdown_event,
        },
    )
    proc.start()
    try:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                import urllib.request
                stats_raw = urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/v1/stats", timeout=0.5,
                ).read()
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("aggregator did not start in 5s")

        stats = json.loads(stats_raw)
        assert stats["success"] is True
        ctrl = stats["data"]["controller"]
        # Summed across the two snapshots.
        assert ctrl["heads_observed"] == 100
        assert ctrl["proofs_submitted"] == 5
        # First-found active_url (cpu snapshot comes first alphabetically).
        assert ctrl["active_url"] == "ws://a"
        # Unioned miners, dedup by id.
        miner_ids = {m["id"] for m in stats["data"]["miners"]}
        assert miner_ids == {"rig-CPU-1", "rig-QPU-DWAVE-1"}
        # Per-mode breakdown is present and bucketed correctly.
        modes = stats["data"]["modes"]
        assert set(modes.keys()) == {"cpu", "qpu"}
        assert modes["cpu"]["controller"]["proofs_submitted"] == 3
        assert modes["qpu"]["controller"]["proofs_submitted"] == 2

        # /api/v1/status also surfaces `modes` for the dashboard.
        status_raw = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/v1/status", timeout=2.0,
        ).read()
        status = json.loads(status_raw)
        assert status["success"] is True
        # is_mining was the bug — snapshot_dir mode would crash on
        # `stats_snapshot_path.stat()`. Now derives from newest file
        # mtime across the dir; freshly written snapshots are <5s old.
        assert status["data"]["is_mining"] is True
        assert set(status["data"]["modes"].keys()) == {"cpu", "qpu"}

        # Regression: /api/v1/mining/attempts must not crash in
        # snapshot_dir mode. Previously it called read_snapshot() with
        # request.app["stats_snapshot_path"] (None in aggregator mode),
        # raising "expected str, bytes or os.PathLike object, not NoneType".
        attempts_raw = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/v1/mining/attempts?miner_id=rig-CPU-1&solution_number=0",
            timeout=2.0,
        ).read()
        attempts = json.loads(attempts_raw)
        assert attempts["success"] is True
        assert "attempts" in attempts["data"]
    finally:
        shutdown_event.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join()

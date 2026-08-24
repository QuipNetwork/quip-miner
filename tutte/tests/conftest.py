"""Pytest configuration for Tutte polynomial test suite."""

import json
import os
import time

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "perf: marks performance regression tests")
    config.addinivalue_line(
        "markers",
        "probe: open-ended measurement probes (multi-hour worst case); "
        "skipped unless --run-probes is given",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--update-rainbow-table",
        action="store_true",
        default=False,
        help="Update rainbow table with newly computed polynomials",
    )
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="Collect benchmark timings and write benchmark_results.json",
    )
    parser.addoption(
        "--run-probes",
        action="store_true",
        default=False,
        help="Run open-ended measurement probes (e.g. K_n scaling)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-probes"):
        return
    skip_probe = pytest.mark.skip(reason="measurement probe; use --run-probes")
    for item in items:
        if "probe" in item.keywords:
            item.add_marker(skip_probe)


@pytest.fixture(scope="session")
def default_table():
    from tutte.lookup import load_default_table
    return load_default_table()


@pytest.fixture(scope="session")
def engine(default_table):
    from tutte.synthesis import SynthesisEngine
    return SynthesisEngine(default_table)


@pytest.fixture(scope="session")
def rainbow_updater(request, default_table):
    """Collects new polynomials during the session and saves at end."""
    collected = {}

    class Updater:
        def add(self, name, graph, polynomial):
            collected[name] = (graph, polynomial)

    updater = Updater()
    yield updater

    if request.config.getoption("--update-rainbow-table") and collected:
        from tutte.lookup import save_binary_rainbow_table

        for name, (graph, poly) in collected.items():
            if default_table.get_entry(name) is None:
                default_table.add(graph, name, poly)
        base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        default_table.save(os.path.join(base, "lookup_table.json"))
        save_binary_rainbow_table(
            default_table, os.path.join(base, "lookup_table.bin")
        )


@pytest.fixture(scope="session")
def benchmark_collector(request):
    """Collects benchmark timings and writes JSON at session end."""
    results = []
    enabled = request.config.getoption("--benchmark")

    class Collector:
        @property
        def is_enabled(self):
            return enabled

        def record(self, name, nodes, edges, spanning_trees, timings_ms):
            if enabled:
                results.append({
                    "name": name,
                    "nodes": nodes,
                    "edges": edges,
                    "spanning_trees": spanning_trees,
                    "timings_ms": timings_ms,
                })

    collector = Collector()
    yield collector

    if enabled and results:
        import subprocess
        import sys

        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            branch = "unknown"

        output = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "branch": branch,
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            },
            "results": results,
        }
        out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "benchmark_results.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nBenchmark results written to {out_path}")


def get_dwave_graph(name, builder_fn):
    """Build a D-Wave graph, skipping if dwave-networkx is unavailable."""
    dnx = pytest.importorskip("dwave_networkx")
    return builder_fn(dnx)

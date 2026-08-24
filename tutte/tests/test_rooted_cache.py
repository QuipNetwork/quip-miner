"""Rooted-lookup table persistence — save/load roundtrip + default loader.

Validates the rainbow-table-style cell T_rooted serialization shipped in
`tutte/roots/rooted_tutte.py`. Cell-quotient DPs consume the cache;
serialization lets them skip the brute-force computation on cold runs.

The persistent lookup table stores partitions in canonical (WL-refined)
labels so any isomorphic runtime graph reuses the same entry regardless
of its label scheme.
"""
from __future__ import annotations

import os
import tempfile

import networkx as nx
import pytest

from tutte.graph import Graph
from tutte.roots.rooted_tutte import (
    _ROOTED_LOOKUP,
    _T_ROOTED_CACHE,
    clear_t_rooted_cache,
    load_default_rooted_lookup,
    load_rooted_lookup,
    load_rooted_lookup_binary,
    save_rooted_lookup,
    save_rooted_lookup_binary,
    t_rooted_cached,
)


@pytest.fixture(autouse=True)
def _isolate_rooted_lookup():
    """Restore the default rooted lookup after each test so we don't
    pollute follow-on tests that depend on a clean state.
    """
    yield
    clear_t_rooted_cache()
    _ROOTED_LOOKUP.clear()
    load_default_rooted_lookup()


def _graphs_dict_for(G, boundary):
    """Helper: build the {(canon, edges, boundary): graph} dict the save_* APIs need.

    Cache key format (May 20, 2026): 3-tuple including a labeling-sensitive
    edges discriminator. See `rooted_tutte.t_rooted_cached`.
    """
    return {(
        G.canonical_key(),
        tuple(sorted(G.edges)),
        tuple(sorted(boundary)),
    ): G}


def test_save_and_load_roundtrip_k4_json():
    """K_4 T_rooted (4-vert boundary) survives JSON save/load roundtrip exactly."""
    clear_t_rooted_cache()
    _ROOTED_LOOKUP.clear()
    G = Graph.from_networkx(nx.complete_graph(4))
    boundary = [0, 1, 2, 3]
    original = t_rooted_cached(G, boundary)
    assert len(original) > 0

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
        out_path = tmp.name
    try:
        n_saved = save_rooted_lookup(out_path, _graphs_dict_for(G, boundary))
        assert n_saved == 1
        clear_t_rooted_cache()
        _ROOTED_LOOKUP.clear()
        n_loaded = load_rooted_lookup(out_path)
        assert n_loaded == n_saved
        reloaded = t_rooted_cached(G, boundary)
        assert reloaded == original
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def test_save_and_load_roundtrip_k4_binary():
    """K_4 T_rooted survives BINARY save/load roundtrip exactly."""
    clear_t_rooted_cache()
    _ROOTED_LOOKUP.clear()
    G = Graph.from_networkx(nx.complete_graph(4))
    boundary = [0, 1, 2, 3]
    original = t_rooted_cached(G, boundary)
    assert len(original) > 0

    with tempfile.NamedTemporaryFile(suffix=".bin", mode="wb", delete=False) as tmp:
        out_path = tmp.name
    try:
        n_saved = save_rooted_lookup_binary(out_path, _graphs_dict_for(G, boundary))
        assert n_saved == 1
        clear_t_rooted_cache()
        _ROOTED_LOOKUP.clear()
        n_loaded = load_rooted_lookup_binary(out_path)
        assert n_loaded == n_saved
        reloaded = t_rooted_cached(G, boundary)
        assert reloaded == original
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def test_binary_is_smaller_than_json():
    """Binary encoding should be substantially smaller than JSON."""
    clear_t_rooted_cache()
    _ROOTED_LOOKUP.clear()
    G = Graph.from_networkx(nx.complete_bipartite_graph(3, 3))
    boundary = list(G.nodes)
    t_rooted_cached(G, boundary)
    graphs = _graphs_dict_for(G, boundary)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_path = tmp.name
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        bin_path = tmp.name
    try:
        save_rooted_lookup(json_path, graphs)
        save_rooted_lookup_binary(bin_path, graphs)
        json_size = os.path.getsize(json_path)
        bin_size = os.path.getsize(bin_path)
        assert bin_size < json_size, f"binary={bin_size} should be < json={json_size}"
    finally:
        for p in (json_path, bin_path):
            if os.path.exists(p):
                os.unlink(p)


def test_load_nonexistent_file_returns_zero():
    """Loader is a no-op when the file doesn't exist."""
    _ROOTED_LOOKUP.clear()
    n = load_rooted_lookup("/tmp/this_path_definitely_does_not_exist_xyz.json")
    assert n == 0
    n = load_rooted_lookup_binary("/tmp/this_path_definitely_does_not_exist_xyz.bin")
    assert n == 0


def test_save_then_load_multiple_cells():
    """Lookup with multiple non-isomorphic entries roundtrips."""
    clear_t_rooted_cache()
    _ROOTED_LOOKUP.clear()
    G_k3 = Graph.from_networkx(nx.complete_graph(3))
    G_k4 = Graph.from_networkx(nx.complete_graph(4))
    G_k5 = Graph.from_networkx(nx.complete_graph(5))
    t_rooted_cached(G_k3, [0, 1, 2])
    t_rooted_cached(G_k4, [0, 1, 2, 3])
    t_rooted_cached(G_k5, [0, 1, 2, 3, 4])

    graphs = {}
    for G, b in [(G_k3, [0, 1, 2]), (G_k4, [0, 1, 2, 3]), (G_k5, [0, 1, 2, 3, 4])]:
        graphs[(
            G.canonical_key(),
            tuple(sorted(G.edges)),
            tuple(sorted(b)),
        )] = G

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        out_path = tmp.name
    try:
        n_saved = save_rooted_lookup_binary(out_path, graphs)
        assert n_saved == 3
        original_cache = dict(_T_ROOTED_CACHE)
        clear_t_rooted_cache()
        _ROOTED_LOOKUP.clear()
        n_loaded = load_rooted_lookup_binary(out_path)
        assert n_loaded == 3
        for (canon, _edges, boundary), partition_dict in original_cache.items():
            G = next(g for (c, _e, _b), g in graphs.items() if c == canon)
            reloaded = t_rooted_cached(G, list(boundary))
            assert reloaded == partition_dict
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def test_load_default_rooted_lookup_is_safe():
    """`load_default_rooted_lookup` doesn't crash; returns an int."""
    n = load_default_rooted_lookup()
    assert isinstance(n, int)
    assert n >= 0


def test_loaded_lookup_used_by_t_rooted_cached():
    """After loading, `t_rooted_cached` returns the cached value without recomputation."""
    clear_t_rooted_cache()
    _ROOTED_LOOKUP.clear()
    G = Graph.from_networkx(nx.complete_graph(3))
    boundary = [0, 1, 2]
    original = t_rooted_cached(G, boundary)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        out_path = tmp.name
    try:
        save_rooted_lookup_binary(out_path, _graphs_dict_for(G, boundary))
        clear_t_rooted_cache()
        _ROOTED_LOOKUP.clear()
        load_rooted_lookup_binary(out_path)
        loaded = t_rooted_cached(G, boundary)
        assert loaded == original
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def test_cross_labeling_cache_hit():
    """Save with networkx labels; load and query with a relabeled isomorph.

    Validates the labeling-aware property: an isomorphic graph with
    different vertex labels (e.g., a Chimera-style {0, 5, 6, 7} | {1, 2, 3, 4}
    K_{4,4}) hits the canonical cache and gets translated correctly.
    """
    clear_t_rooted_cache()
    _ROOTED_LOOKUP.clear()
    # Save K_{4,4} with default networkx labels (a={0,1,2,3}, b={4,5,6,7}).
    G_nx = Graph.from_networkx(nx.complete_bipartite_graph(4, 4))
    t_rooted_cached(G_nx, list(G_nx.nodes))
    graphs = _graphs_dict_for(G_nx, list(G_nx.nodes))

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        out_path = tmp.name
    try:
        save_rooted_lookup_binary(out_path, graphs)
        clear_t_rooted_cache()
        _ROOTED_LOOKUP.clear()
        load_rooted_lookup_binary(out_path)

        # Build the SAME K_{4,4} with dnx-style labels (a={0,5,6,7}, b={1,2,3,4}).
        relabel = {0: 0, 1: 5, 2: 6, 3: 7, 4: 1, 5: 2, 6: 3, 7: 4}
        nxg = nx.relabel_nodes(nx.complete_bipartite_graph(4, 4), relabel)
        G_dnx = Graph.from_networkx(nxg)
        assert G_dnx.canonical_key() == G_nx.canonical_key(), "should be isomorphic"

        # Cache hit + translate to dnx labels.
        result_dnx = t_rooted_cached(G_dnx, sorted(G_dnx.nodes))
        # Compare against fresh-computed (clear cache, recompute).
        clear_t_rooted_cache()
        _ROOTED_LOOKUP.clear()
        result_fresh = t_rooted_cached(G_dnx, sorted(G_dnx.nodes))
        assert result_dnx == result_fresh, "cache-hit must match fresh-computed bit-for-bit"
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)

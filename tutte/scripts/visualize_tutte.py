#!/usr/bin/env python
"""
Tutte Engine Visualizer — Flask + Sigma.js (WebGL) with live SSE streaming.

Layout:
  Row 1 — Input Graph + Contributing Graphs (side by side)
  Row 2 — Result
  Row 3 — Summary
  Row 4 — Timeline (live-streamed via SSE)

Usage:
    python scripts/visualize_tutte.py
    Then open http://localhost:5002/?atlas=18

URL Parameters:
    atlas=N          — NetworkX graph atlas index (0–1252)
    dwave_topo=zephyr&dwave_m=1&dwave_t=1 — D-Wave topology (zephyr/pegasus/chimera)
    edges=0-1,1-2,2-0 — Custom edge list
    rand_n=12&rand_m=18 — Random graph with n nodes and m edges
    timeout=60       — Engine timeout in seconds (default 60)
    threshold=100    — Timeline bottleneck threshold in ms (default 100)

The visualizer uses the single ``SynthesisEngine`` for all synthesis.
"""

import json
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from flask import Flask, Response, request
from tutte.graph import Graph
from tutte.logs import EventType, LogLevel, get_log, reset_log
from tutte.lookup.core import load_default_table
from tutte.synthesis.base import SynthesisResult
from tutte.synthesis.engine import SynthesisEngine

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def poly_to_html(poly_str: str) -> str:
    """Convert polynomial string to readable HTML with superscripts."""
    html = re.sub(r'\*\*(\d+)', r'<sup>\1</sup>', str(poly_str))
    html = html.replace('*', '')
    return html


def factored_poly_html(tutte_poly) -> str:
    """Try to factor a TuttePolynomial via SymPy. Returns HTML string.

    Shows factored form if non-trivial, otherwise falls back to expanded form.
    """
    from sympy import factor, symbols

    x, y = symbols('x y')
    expr = sum(coeff * x**i * y**j for (i, j), coeff in tutte_poly._coeffs.items())
    factored = factor(expr)

    # If factored form is shorter, use it; otherwise show expanded
    fact_str = str(factored)
    exp_str = str(expr)
    display = fact_str if len(fact_str) < len(exp_str) else exp_str

    return poly_to_html(display)


def _normalize_nx_graph(G):
    """Return G with nodes relabeled to consecutive ints starting at 0.

    Several networkx generators (grid_2d_graph, kneser_graph, balanced_tree
    with certain inputs) use tuple or frozenset node labels that break
    downstream rendering and spring_layout keying. Relabeling to ints keeps
    every family path uniform. Original labels are preserved under the
    'original_label' node attribute so compute_layout can use them (e.g.
    for grid positions).
    """
    if G is None:
        return G
    return nx.convert_node_labels_to_integers(G, label_attribute='original_label')


def parse_graph(args) -> tuple:
    """Parse URL parameters into a (nx.Graph, description, source_hint) tuple.

    source_hint is one of {'zephyr', 'pegasus', 'chimera', 'grid', None}
    and feeds compute_layout() so the visualizer can pick a topology-aware
    layout instead of spring layout for graphs where one exists.

    For D-Wave topologies, the layout function requires the original
    dnx-generated graph (it rejects anonymous reconstructions), so we
    compute positions here — while we still have the raw graph — and
    attach them as graph-level metadata for compute_layout to consume.

    The returned graph always has integer node labels (0..n-1).
    """
    G_raw, desc, source_hint = _parse_graph_raw(args)
    if G_raw is None:
        return None, desc, source_hint

    raw_pos = None
    if source_hint in ("zephyr", "pegasus", "chimera"):
        try:
            import dwave_networkx as dnx
            fn = {
                "zephyr": dnx.zephyr_layout,
                "pegasus": dnx.pegasus_layout,
                "chimera": dnx.chimera_layout,
            }[source_hint]
            raw_pos = fn(G_raw)
        except Exception:
            raw_pos = None

    G = _normalize_nx_graph(G_raw)
    if raw_pos is not None:
        orig = nx.get_node_attributes(G, "original_label")
        G.graph["_precomputed_pos"] = {
            new: (float(raw_pos[orig[new]][0]), float(raw_pos[orig[new]][1]))
            for new in G.nodes()
            if orig.get(new) in raw_pos
        }
    return G, desc, source_hint


def _parse_graph_raw(args) -> tuple:
    """Inner: returns (nx.Graph, description, source_hint) without normalizing labels."""
    atlas = args.get("atlas", type=int)
    if atlas is not None:
        try:
            G = nx.graph_atlas(atlas)
            return G, f"Atlas #{atlas}", None
        except Exception as e:
            return None, f"Invalid atlas index: {e}", None

    dwave_topo = args.get("dwave_topo", "").strip()
    dwave_m = args.get("dwave_m", type=int)
    dwave_n = args.get("dwave_n", type=int)
    dwave_t = args.get("dwave_t", type=int)
    if dwave_topo and dwave_m is not None:
        try:
            import dwave_networkx as dnx
            if dwave_topo == "zephyr":
                t = dwave_t if dwave_t is not None else 1
                G = dnx.zephyr_graph(dwave_m, t)
                return G, f"Zephyr Z({dwave_m},{t})", "zephyr"
            elif dwave_topo == "pegasus":
                if dwave_m < 2:
                    return None, "Pegasus requires m >= 2 (P(1) is empty)", None
                G = dnx.pegasus_graph(dwave_m)
                return G, f"Pegasus P({dwave_m})", "pegasus"
            elif dwave_topo == "chimera":
                # D-Wave Chimera is m×n tiles with shore size 4. When dwave_n
                # is omitted, default to a square m×m grid.
                n = dwave_n if dwave_n is not None else dwave_m
                G = dnx.chimera_graph(dwave_m, n=n)
                if dwave_m == n:
                    return G, f"Chimera C({dwave_m})", "chimera"
                return G, f"Chimera C({dwave_m},{n})", "chimera"
            else:
                return None, f"Unknown D-Wave topology: {dwave_topo}", None
        except ImportError:
            return None, "dwave-networkx not installed", None
        except Exception as e:
            return None, f"Invalid D-Wave params: {e}", None

    edges_str = args.get("edges", "").strip()
    if edges_str:
        try:
            G = nx.Graph()
            for part in edges_str.split(","):
                u, v = part.strip().split("-")
                G.add_edge(int(u), int(v))
            return G, f"Custom ({G.number_of_edges()} edges)", None
        except Exception as e:
            return None, f"Invalid edge list: {e}", None

    # Random graph: rand_n=12&rand_m=12
    rand_n = args.get("rand_n", type=int)
    rand_m = args.get("rand_m", type=int)
    if rand_n is not None and rand_m is not None:
        max_edges = rand_n * (rand_n - 1) // 2
        if rand_m > max_edges:
            return None, f"Too many edges: {rand_n} nodes can have at most {max_edges} edges", None
        if rand_n < 1:
            return None, "Need at least 1 node", None
        G = nx.gnm_random_graph(rand_n, rand_m)
        return G, f"Random G({rand_n},{rand_m}) — {G.number_of_nodes()}n, {G.number_of_edges()}e", None

    # Cell builder: cell_builder=1 + cb_* params
    if args.get("cell_builder", "0") == "1":
        try:
            from tutte.scripts.cell_builder import build_cell_graph

            def _cell_params(prefix: str, ctype: str) -> dict:
                """Build the param dict for one cell, including ONLY the keys
                that ctype actually consumes — otherwise stray defaults
                like n=4 silently change a Chimera m=1 cell into n=4."""
                t = ctype.lower()
                if t in ("k_n", "kn", "complete",
                         "c_n", "cn", "cycle",
                         "p_n", "pn", "path"):
                    return {"n": args.get(f"{prefix}n", 4, type=int)}
                if t in ("k_a_b", "kab", "complete_bipartite"):
                    return {
                        "a": args.get(f"{prefix}a", 4, type=int),
                        "b": args.get(f"{prefix}b", 4, type=int),
                    }
                if t in ("chimera", "cm"):
                    p = {"m": args.get(f"{prefix}m", 1, type=int)}
                    n2 = args.get(f"{prefix}n2", type=int)
                    if n2 is not None:
                        p["n"] = n2
                    return p
                if t in ("pegasus", "pm"):
                    return {"m": args.get(f"{prefix}m", 2, type=int)}
                if t in ("zephyr", "z"):
                    return {
                        "m": args.get(f"{prefix}m", 1, type=int),
                        "t": args.get(f"{prefix}t", 1, type=int),
                    }
                if t in ("z11", "z_1_1", "zephyr11",
                         "cm1", "chimera1",
                         "pm2", "pegasus2"):
                    return {}  # legacy fixed-shape cells
                return {}

            cell_type = args.get("cb_cell_type", "K_a_b")
            cell_params = _cell_params("cb_cell_", cell_type)
            junction_type = args.get("cb_junction_type", "matching")
            jt = junction_type.lower()
            if jt in ("matching", "m_k", "mk"):
                junction_params = {"k": args.get("cb_junction_k", 4, type=int)}
            elif jt in ("k_a_b_junction", "kab_junction",
                        "complete_bipartite_junction"):
                junction_params = {
                    "a": args.get("cb_junction_a", 4, type=int),
                    "b": args.get("cb_junction_b", 4, type=int),
                }
            else:
                junction_params = {}
            family_type = args.get("cb_family_type", "path")
            ft = family_type.lower()
            if ft == "grid":
                family_params = {
                    "rows": args.get("cb_family_rows", 2, type=int),
                    "cols": args.get("cb_family_cols", 2, type=int),
                }
            elif ft == "interleaved":
                family_params = {
                    "count": args.get("cb_family_count", 4, type=int),
                    "pattern": args.get("cb_family_pattern", "path"),
                }
            else:  # path / cycle
                family_params = {"count": args.get("cb_family_count", 3, type=int)}
            alt_cell_type = args.get("cb_alt_cell_type", "").strip() or None
            alt_cell_params = None
            if alt_cell_type:
                alt_cell_params = _cell_params("cb_alt_cell_", alt_cell_type)
            G, label = build_cell_graph(
                cell_type, cell_params, junction_type, junction_params,
                family_type, family_params, alt_cell_type, alt_cell_params,
            )
            return G, label, None
        except Exception as e:
            return None, f"Invalid cell builder params: {e}", None

    # Graph family: family=complete&n=5 or family=grid&n=3&m=4
    family = args.get("family", "").strip()
    if family:
        n = args.get("n", 5, type=int)
        m = args.get("m", 0, type=int)
        try:
            G, desc = _build_family_graph(family, n, m)
            hint = "grid" if family == "grid" else None
            return G, desc, hint
        except Exception as e:
            return None, f"Invalid family params: {e}", None

    return None, "", None


# Map of family name → (generator, needs_m, label_fn)
GRAPH_FAMILIES = {
    "complete": ("Complete K_n", False),
    "cycle": ("Cycle C_n", False),
    "path": ("Path P_n", False),
    "wheel": ("Wheel W_n", False),
    "star": ("Star S_n", False),
    "complete_bipartite": ("Complete Bipartite K_{n,m}", True),
    "grid": ("Grid G_{n,m}", True),
    "ladder": ("Ladder L_n", False),
    "petersen": ("Petersen", False),
    "tutte": ("Tutte", False),
    "dodecahedral": ("Dodecahedral", False),
    "icosahedral": ("Icosahedral", False),
    "octahedral": ("Octahedral", False),
    "cubical": ("Cubical", False),
    "tetrahedral": ("Tetrahedral", False),
    "heawood": ("Heawood", False),
    "moebius_kantor": ("Moebius-Kantor", False),
    "bull": ("Bull", False),
    "chvatal": ("Chvatal", False),
    "desargues": ("Desargues", False),
    "pappus": ("Pappus", False),
    "gear": ("Gear G_n", False),
    "prism": ("Prism P_n", False),
    "friendship": ("Friendship F_n", False),
    "barbell": ("Barbell B_{n,m}", True),
    "empty": ("Empty E_n", False),
    "random_tree": ("Random Tree T_n", False),
    "balanced_tree": ("Balanced Tree B_{r,h}", True),
    "kneser": ("Kneser K_{n,k}", True),
    "k_regular": ("k-Regular R_{k,n}", True),
}


def _build_family_graph(family: str, n: int, m: int):
    """Build a named graph family. Returns (nx.Graph, description)."""
    if family == "complete":
        return nx.complete_graph(n), f"K_{n}"
    elif family == "cycle":
        return nx.cycle_graph(n), f"C_{n}"
    elif family == "path":
        return nx.path_graph(n), f"P_{n}"
    elif family == "wheel":
        return nx.wheel_graph(n), f"W_{n}"
    elif family == "star":
        return nx.star_graph(n), f"S_{n}"
    elif family == "complete_bipartite":
        return nx.complete_bipartite_graph(n, m or n), f"K_{{{n},{m or n}}}"
    elif family == "grid":
        return nx.grid_2d_graph(n, m or n), f"Grid({n},{m or n})"
    elif family == "ladder":
        return nx.ladder_graph(n), f"Ladder({n})"
    elif family == "petersen":
        return nx.petersen_graph(), "Petersen"
    elif family == "tutte":
        return nx.tutte_graph(), "Tutte"
    elif family == "dodecahedral":
        return nx.dodecahedral_graph(), "Dodecahedral"
    elif family == "icosahedral":
        return nx.icosahedral_graph(), "Icosahedral"
    elif family == "octahedral":
        return nx.octahedral_graph(), "Octahedral"
    elif family == "cubical":
        return nx.cubical_graph(), "Cubical"
    elif family == "tetrahedral":
        return nx.tetrahedral_graph(), "Tetrahedral"
    elif family == "heawood":
        return nx.heawood_graph(), "Heawood"
    elif family == "moebius_kantor":
        return nx.moebius_kantor_graph(), "Moebius-Kantor"
    elif family == "bull":
        return nx.bull_graph(), "Bull"
    elif family == "chvatal":
        return nx.chvatal_graph(), "Chvatal"
    elif family == "desargues":
        return nx.desargues_graph(), "Desargues"
    elif family == "pappus":
        return nx.pappus_graph(), "Pappus"
    elif family == "gear":
        G = nx.wheel_graph(n)
        # Insert vertex on each spoke
        gear = nx.Graph()
        hub = 0
        outer = list(range(1, n))
        for i, v in enumerate(outer):
            gear.add_edge(hub, n + i)
            gear.add_edge(n + i, v)
            gear.add_edge(v, outer[(i + 1) % len(outer)])
        return gear, f"Gear({n})"
    elif family == "prism":
        return nx.circular_ladder_graph(n), f"Prism({n})"
    elif family == "friendship":
        G = nx.Graph()
        for i in range(n):
            G.add_edges_from([(0, 2*i+1), (0, 2*i+2), (2*i+1, 2*i+2)])
        return G, f"Friendship({n})"
    elif family == "barbell":
        return nx.barbell_graph(n, m), f"Barbell({n},{m})"
    elif family == "empty":
        return nx.empty_graph(n), f"E_{n}"
    elif family == "random_tree":
        return nx.random_labeled_tree(n), f"RandomTree({n})"
    elif family == "balanced_tree":
        r = n  # branching factor
        h = m if m else 2  # height
        return nx.balanced_tree(r, h), f"BalancedTree({r},{h})"
    elif family == "kneser":
        if m >= n:
            raise ValueError(f"Kneser K(n,k) requires k < n, got n={n}, k={m}")
        return nx.kneser_graph(n, m or 1), f"Kneser({n},{m or 1})"
    elif family == "k_regular":
        k = n  # degree
        num_nodes = m if m else 10  # number of nodes
        if k >= num_nodes:
            raise ValueError(f"k-Regular requires k < n, got k={k}, n={num_nodes}")
        if k * num_nodes % 2 != 0:
            raise ValueError(f"k-Regular requires k*n even, got k={k}, n={num_nodes}")
        return nx.random_regular_graph(k, num_nodes), f"Regular({k},{num_nodes})"
    else:
        raise ValueError(f"Unknown family: {family}")


def _build_baseline_graph(choice: str) -> tuple:
    """Build a small baseline graph for cell-builder comparison.

    Returns (nx.Graph, label). All baselines are small (<= 32 vertices) so
    the second engine run completes within a few seconds. Used by /stream
    when cb_compare_baseline=1.
    """
    choice = choice.strip()
    if choice == "K_5":
        return nx.complete_graph(5), "K_5"
    if choice == "K_4_4":
        return nx.complete_bipartite_graph(4, 4), "K_{4,4}"
    if choice == "C_5":
        return nx.cycle_graph(5), "C_5"
    if choice == "Cm1":
        import dwave_networkx as dnx
        G = dnx.chimera_graph(1)
        mapping = {old: new for new, old in enumerate(sorted(G.nodes))}
        return nx.relabel_nodes(G, mapping), "Cm1 (K_{4,4})"
    if choice == "Z11":
        import dwave_networkx as dnx
        G = dnx.zephyr_graph(1, t=1)
        mapping = {old: new for new, old in enumerate(sorted(G.nodes))}
        return nx.relabel_nodes(G, mapping), "Z(1,1)"
    raise ValueError(f"Unknown baseline choice {choice!r}")


def _size_for_n(n: int) -> float:
    """Node render size based on graph magnitude — keeps Z(12,4) readable."""
    if n <= 100:
        return 8.0
    if n <= 1000:
        return 4.0
    if n <= 10000:
        return 2.0
    return 1.0


def compute_layout(G, *, source_hint=None):
    """Return {node_id: (x, y)} for a networkx graph.

    Resolution order:
    1. `G.graph['_precomputed_pos']` — positions baked in by parse_graph
       (used for D-Wave topologies, where dnx layout needs the raw graph).
    2. source_hint=='grid' → (row, col) from preserved original_label.
    3. Small connected graphs → kamada_kawai.
    4. Small graphs → spring_layout (seeded).
    5. Everything else → random_layout; user can click Re-layout for FA2.
    """
    n = G.number_of_nodes()
    if n == 0:
        return {}

    precomp = G.graph.get("_precomputed_pos") if hasattr(G, "graph") else None
    if precomp and len(precomp) == n:
        return dict(precomp)

    if source_hint == "grid":
        orig = nx.get_node_attributes(G, "original_label")
        if orig and len(orig) == n:
            pos = {}
            for nd, label in orig.items():
                if isinstance(label, tuple) and len(label) == 2:
                    pos[nd] = (float(label[0]), float(label[1]))
            if len(pos) == n:
                return pos

    if n <= 300:
        try:
            if nx.is_connected(G):
                return nx.kamada_kawai_layout(G)
        except Exception:
            pass
        return nx.spring_layout(G, seed=42)

    return nx.random_layout(G, seed=42)


def sigma_graph_json(G, *, source_hint=None, pos=None) -> str:
    """Serialize G to graphology-importable JSON string.

    Shape: {nodes: [{key, attributes:{label,x,y,size,color}}],
            edges: [{key, source, target, attributes:{size,color}}]}
    """
    if pos is None:
        pos = compute_layout(G, source_hint=source_hint)
    n_nodes = G.number_of_nodes()
    size = _size_for_n(n_nodes)
    nodes = []
    for nd in G.nodes():
        p = pos.get(nd, (0.0, 0.0))
        nodes.append({
            "key": nd,
            "attributes": {
                "label": str(nd),
                "x": float(p[0]),
                "y": float(p[1]),
                "size": size,
                "color": "#4f8ef7",
            },
        })
    edges = []
    for i, (u, v) in enumerate(G.edges()):
        edges.append({
            "key": f"e{i}",
            "source": u,
            "target": v,
            "attributes": {"size": 1.0, "color": "#b0b0b0"},
        })
    return json.dumps({"nodes": nodes, "edges": edges})


def sigma_graph_vis(G, div_id, *, source_hint=None, pos=None, register_as_target=False) -> str:
    """Return a JS snippet that renders G into div_id via Sigma + graphology.

    When register_as_target=True, the Sigma instance and the graphology
    graph are exposed on window._targetSigma / window._targetGraph so the
    Re-layout button can operate on them. A `window._renderInputGraph()`
    closure is also installed so the Refresh button can fully recreate
    the renderer (recovering from WebGL context exhaustion).
    """
    data_json = sigma_graph_json(G, source_hint=source_hint, pos=pos)
    if register_as_target:
        # Self-recreating renderer: defines a function that builds a
        # fresh graphology Graph + Sigma instance every time it's
        # called. Installs it on window._renderInputGraph so the
        # refresh button can re-execute it on demand.
        return (
            "(function(){"
            "window._renderInputGraph = function() {"
            f"  var data = {data_json};"
            "  var container = document.getElementById('"
            f"{div_id}"
            "');"
            "  if (!container) return;"
            "  if (window._targetSigma) {"
            "    try { window._targetSigma.kill(); } catch(e) {}"
            "    window._targetSigma = null;"
            "  }"
            "  container.innerHTML = '';"
            "  var graph = new graphology.Graph();"
            "  graph.import(data);"
            "  var sigma = new Sigma(graph, container, "
            "{renderLabels: graph.order <= 1000, labelDensity: 0.5, "
            "labelGridCellSize: 60, labelRenderedSizeThreshold: 6, "
            "defaultEdgeColor: '#b0b0b0', minCameraRatio: 0.05, maxCameraRatio: 20});"
            "  window._targetSigma = sigma;"
            "  window._targetGraph = graph;"
            "  return sigma;"
            "};"
            "window._renderInputGraph();"
            "})();"
        )
    return (
        "(function(){"
        f"var graph = new graphology.Graph();"
        f"graph.import({data_json});"
        f"var sigma = new Sigma(graph, document.getElementById('{div_id}'), "
        "{renderLabels: graph.order <= 1000, labelDensity: 0.5, "
        "labelGridCellSize: 60, labelRenderedSizeThreshold: 6, "
        "defaultEdgeColor: '#b0b0b0', minCameraRatio: 0.05, maxCameraRatio: 20});"
        "})();"
    )


def _snapshot_to_nx(snapshot):
    """Convert a log graph snapshot dict to nx.Graph for rendering.

    Snapshot shape: {"nodes": [...], "edges": [[u, v, mult], ...],
                     "loops": [[n, mult], ...]}.

    Multiplicities and loops are collapsed to simple edges for the
    Contributing-Graphs cards (the same convention as
    `_synth_graph_to_nx`).
    """
    if not isinstance(snapshot, dict):
        return None
    G = nx.Graph()
    for n in snapshot.get("nodes", []):
        G.add_node(n)
    for edge in snapshot.get("edges", []) or []:
        if not edge:
            continue
        u = edge[0]
        v = edge[1] if len(edge) > 1 else edge[0]
        if u != v:
            G.add_edge(u, v)
    return G


def _snapshot_counts(snapshot):
    """Return (node_count, total_edge_count_including_multiplicities) for
    a log snapshot dict. Used to label Contributing Graphs cards."""
    if not isinstance(snapshot, dict):
        return 0, 0
    n = len(snapshot.get("nodes", []) or [])
    m = 0
    for edge in snapshot.get("edges", []) or []:
        if not edge:
            continue
        mult = edge[2] if len(edge) > 2 else 1
        m += mult
    for loop in snapshot.get("loops", []) or []:
        if not loop:
            continue
        mult = loop[1] if len(loop) > 1 else 1
        m += mult
    return n, m


def _synth_graph_to_nx(g_obj):
    """Convert a tutte Graph or MultiGraph synthesized during a run to an nx.Graph.

    Used by the visualizer to render engine-synthesized sub-problems that
    are not in the rainbow table. MultiGraphs are flattened to simple
    networkx graphs; parallel edges and loops are dropped in this view
    (node positions and connectivity are the visually useful parts).
    """
    if g_obj is None:
        return None
    if hasattr(g_obj, 'to_networkx'):
        try:
            return g_obj.to_networkx()
        except Exception:
            pass
    edge_counts = getattr(g_obj, 'edge_counts', None)
    nodes = getattr(g_obj, 'nodes', None)
    if edge_counts is not None and nodes is not None:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edge_counts.keys())
        return G
    return None


def graph_from_entry(entry):
    """Reconstruct nx.Graph from a rainbow table MinorEntry."""
    if entry.graph is not None:
        return entry.graph.to_networkx()

    from tutte.graphs.covering import _minor_to_graph
    g = _minor_to_graph(entry)
    if g is not None:
        return g.to_networkx()

    if entry.name.startswith("atlas_"):
        try:
            idx = int(entry.name[6:])
            return nx.graph_atlas(idx)
        except (ValueError, nx.NetworkXError):
            pass

    return None


# ---------------------------------------------------------------------------
# Event colors (shared between Python summary & JS)
# ---------------------------------------------------------------------------

EVENT_COLORS = {
    "cache_hit": "#2e7d32",
    "cache_miss": "#9e9e9e",
    "lookup_hit": "#2e7d32",
    "lookup_miss": "#9e9e9e",
    "base_case": "#2e7d32",
    "factorize": "#1565c0",
    "series_parallel": "#6a1b9a",
    "treewidth_dp": "#00838f",
    "ksum": "#1565c0",
    "family_recognition": "#2e7d32",
    "vf2_match": "#e65100",
    "tile_accept": "#1565c0",
    "cover_result": "#1565c0",
    "edge_add": "#c62828",
    "multigraph_op": "#9e9e9e",
    "verify": "#2e7d32",
    "theorem6": "#e65100",
    "hierarchical": "#e65100",
    "chord_rule": "#ad1457",
    "unified_formula": "#8e24aa",
    "kmatching_formula": "#d81b60",
    "cotree_dp": "#00838f",
    "cell_quotient_dp": "#ff6f00",
    "synthesis_start": "#9e9e9e",
    "candidate_filter": "#9e9e9e",
}


# ---------------------------------------------------------------------------
# HTML template — page loads immediately, timeline streams via SSE
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Tutte Engine Visualizer</title>
  <script src="https://cdn.jsdelivr.net/npm/graphology@0.25.4/dist/graphology.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/graphology-library@0.8.0/dist/graphology-library.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/sigma@3.0.1/dist/sigma.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; margin: 0; padding: 16px; background: #fafafa; }}
    h2 {{ font-size: 14px; margin: 0 0 8px 0; color: #333; }}
    .graphs-row {{ display: flex; gap: 16px; margin-bottom: 16px; }}
    .graphs-row > .panel {{ flex: 1; display: flex; flex-direction: column; }}
    .panel {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }}
    .graph-box {{ height: 320px; border: 1px solid #eee; border-radius: 4px; }}
    .small-graph {{ height: 140px; border: 1px solid #eee; border-radius: 4px; }}
    .meta {{ font-size: 11px; color: #555; line-height: 1.6; }}
    .meta b {{ color: #333; }}
    .result-grid {{ display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; font-size: 12px; }}
    .result-grid dt {{ color: #888; }}
    .result-grid dd {{ margin: 0; }}
    .badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px; color: #fff; font-size: 10px; font-weight: bold; }}
    .timeline-scroll {{ max-height: 400px; overflow-y: auto; }}
    .timeline {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
    .timeline th {{ background: #f5f5f5; position: sticky; top: 0; text-align: left; padding: 4px 6px; }}
    .timeline td {{ padding: 2px 6px; border-bottom: 1px solid #f0f0f0; white-space: nowrap; }}
    .timeline td:last-child {{ white-space: normal; word-break: break-word; }}
    .summary {{ border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }}
    .summary th {{ background: #f5f5f5; text-align: left; padding: 5px 10px; border-bottom: 2px solid #ddd; white-space: nowrap; }}
    .summary th:nth-child(2), .summary th:nth-child(3), .summary th:nth-child(4) {{ text-align: right; }}
    .summary td {{ padding: 4px 10px; border-bottom: 1px solid #f0f0f0; white-space: nowrap; }}
    .summary tr:hover {{ background: #fafafa; }}
    .summary .pct-bar {{ display: inline-block; height: 6px; border-radius: 3px; vertical-align: middle; }}
    .minors-grid {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .minor-card {{ flex: 1; min-width: 200px; border: 1px solid #eee; border-radius: 4px; padding: 6px; transition: background 0.1s, border-color 0.1s; }}
    .minor-card.has-provenance {{ border-color: #ff5722; cursor: pointer; }}
    .minor-card.has-provenance:hover {{ background: #fff3e0; }}
    .minor-card.prov-active {{ background: #ffe0b2; border-color: #d84315; border-width: 2px; padding: 5px; }}
    .prov-badge {{ display: inline-block; background: #ff5722; color: white; font-size: 9px; font-weight: bold; padding: 1px 5px; border-radius: 8px; margin-left: 4px; vertical-align: middle; cursor: pointer; user-select: none; }}
    .prov-badge.active {{ background: #d84315; box-shadow: 0 0 0 2px #ffab91; }}
    .active-subgraph-chip {{ display: inline-block; color: white; font-size: 10px; font-family: monospace; padding: 2px 7px 2px 8px; border-radius: 10px; margin: 2px 4px 2px 0; cursor: pointer; user-select: none; }}
    .active-subgraph-chip .x {{ margin-left: 6px; opacity: 0.7; }}
    .active-subgraph-chip:hover .x {{ opacity: 1; }}
    .minor-label {{ font-size: 11px; font-weight: bold; margin-bottom: 4px; }}
    input[type=number] {{ font-family: inherit; padding: 3px 6px; width: 60px; border: 1px solid #ccc; border-radius: 3px; }}
    input[type=text] {{ font-family: inherit; padding: 3px 6px; width: 100%; border: 1px solid #ccc; border-radius: 3px; }}
    select {{ font-family: inherit; padding: 3px 6px; border: 1px solid #ccc; border-radius: 3px; }}
    button {{ font-family: inherit; padding: 3px 10px; cursor: pointer; border: 1px solid #ccc; border-radius: 3px; background: #fff; }}
    button:hover {{ background: #f0f0f0; }}
    .controls {{ display: flex; gap: 6px; align-items: center; flex-wrap: wrap; font-size: 12px; margin-bottom: 12px; }}
    .poly {{ font-size: 12px; line-height: 1.5; word-break: break-word; }}
    .timeout-banner {{ background: #fff3e0; border: 1px solid #e65100; border-radius: 4px; padding: 8px; color: #e65100; font-weight: bold; }}
    .error-banner {{ background: #fce4ec; border: 1px solid #c62828; border-radius: 4px; padding: 8px; color: #c62828; }}
    .section {{ margin-bottom: 16px; }}
    .ctrl-grid {{ display: flex; gap: 32px; }}
    .ctrl-group {{ display: flex; flex-direction: column; gap: 6px; }}
    .ctrl-label {{ font-size: 11px; font-weight: bold; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }}
    .ctrl-radio {{ font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 4px; }}
    .ctrl-radio input[type="radio"] {{ margin: 0; }}
    .ctrl-indent {{ margin-left: 20px; font-size: 12px; }}
    .ctrl-row {{ display: flex; align-items: center; gap: 6px; font-size: 12px; }}
    .run-btn {{ background: #1565c0; color: #fff; border: none; padding: 6px 20px; border-radius: 4px; font-weight: bold; font-size: 12px; cursor: pointer; }}
    .run-btn:hover {{ background: #0d47a1; }}
    input:disabled, select:disabled {{ opacity: 0.4; }}
    .spinner {{ display: inline-block; width: 12px; height: 12px; border: 2px solid #ccc; border-top-color: #333; border-radius: 50%; animation: spin 0.6s linear infinite; margin-right: 6px; vertical-align: middle; }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  </style>
</head>
<body>
  <div class="panel section" style="padding:16px;">
    <h2 style="margin-bottom:12px;">Control Panel</h2>
    <form method="get" action="/" id="ctrl-form">
      <input type="hidden" name="form_submitted" value="1">
      <div class="ctrl-grid">
        <!-- Input source selection -->
        <div class="ctrl-group">
          <div class="ctrl-label">Input Graph</div>
          <label class="ctrl-radio"><input type="radio" name="source" value="atlas" {atlas_checked}> Atlas index</label>
          <div class="ctrl-indent"><input type="number" name="atlas" value="{atlas_val}" placeholder="#" min="0" max="1252" style="width:80px" {atlas_disabled}></div>
          <label class="ctrl-radio"><input type="radio" name="source" value="dwave" {dwave_checked}> D-Wave topology</label>
          <div class="ctrl-indent">
            <select name="dwave_topo" id="dwave-topo-select" {dwave_disabled}>{dwave_topo_options}</select>
            <span id="dwave-m-wrap"><span id="dwave-m-label">{dwave_m_label}</span>=<input type="number" name="dwave_m" value="{dwave_m_val}" placeholder="1" min="1" style="width:50px" {dwave_disabled}></span>
            <span id="dwave-n-wrap" style="{dwave_n_display}">n=<input type="number" name="dwave_n" value="{dwave_n_val}" placeholder="(=m)" min="1" style="width:55px" {dwave_disabled}></span>
            <span id="dwave-t-wrap" style="{dwave_t_display}"><span id="dwave-t-label">{dwave_t_label}</span>=<input type="number" name="dwave_t" value="{dwave_t_val}" placeholder="1" min="1" style="width:50px" {dwave_disabled}></span>
          </div>
          <label class="ctrl-radio"><input type="radio" name="source" value="family" {family_checked}> Graph family</label>
          <div class="ctrl-indent">
            <select name="family" id="family-select" {family_disabled}>{family_options}</select>
            <span id="n-wrap"><span id="n-label">{n_label}</span>=<input type="number" name="n" value="{n_val}" placeholder="5" min="1" style="width:50px" {family_disabled}></span>
            <span id="m-wrap" style="display:none"><span id="m-label">{m_label}</span>=<input type="number" name="m" value="{m_val}" placeholder="" min="0" style="width:50px" {family_disabled}></span>
          </div>
          <label class="ctrl-radio"><input type="radio" name="source" value="edges" {edges_checked}> Edge list</label>
          <div class="ctrl-indent"><input type="text" name="edges" value="{edges_val}" placeholder="0-1,1-2,2-3,3-0" style="width:260px" {edges_disabled}></div>
          <label class="ctrl-radio"><input type="radio" name="source" value="random" {random_checked}> Random graph</label>
          <div class="ctrl-indent">
            nodes=<input type="number" name="rand_n" value="{rand_n_val}" placeholder="12" min="1" max="50000" style="width:65px" {random_disabled}>
            edges=<input type="number" name="rand_m" value="{rand_m_val}" placeholder="12" min="0" style="width:55px" {random_disabled}>
            <span id="rand-max-edges" style="color:#999;font-size:11px">{rand_max_hint}</span>
          </div>
          <label class="ctrl-radio"><input type="radio" name="source" value="cell_builder" {cell_builder_checked}> Cell builder</label>
          <input type="hidden" name="cell_builder" id="cb-flag" value="{cb_flag_val}">
          <div class="ctrl-indent" style="display:flex;flex-direction:column;gap:3px;">
            <div class="ctrl-row">
              <span style="min-width:60px">Cell:</span>
              <select name="cb_cell_type" id="cb-cell-type" {cb_disabled}>{cb_cell_type_options}</select>
              <span id="cb-cell-n-wrap">n=<input type="number" name="cb_cell_n" value="{cb_cell_n_val}" min="1" style="width:45px" {cb_disabled}></span>
              <span id="cb-cell-ab-wrap" style="display:none">a=<input type="number" name="cb_cell_a" value="{cb_cell_a_val}" min="1" style="width:40px" {cb_disabled}> b=<input type="number" name="cb_cell_b" value="{cb_cell_b_val}" min="1" style="width:40px" {cb_disabled}></span>
              <span id="cb-cell-m-wrap" style="display:none">m=<input type="number" name="cb_cell_m" value="{cb_cell_m_val}" min="1" style="width:40px" {cb_disabled}></span>
              <span id="cb-cell-n2-wrap" style="display:none">n=<input type="number" name="cb_cell_n2" value="{cb_cell_n2_val}" placeholder="(=m)" min="1" style="width:55px" {cb_disabled}></span>
              <span id="cb-cell-t-wrap" style="display:none">t=<input type="number" name="cb_cell_t" value="{cb_cell_t_val}" min="1" style="width:40px" {cb_disabled}></span>
            </div>
            <div class="ctrl-row">
              <span style="min-width:60px">Junction:</span>
              <select name="cb_junction_type" id="cb-junction-type" {cb_disabled}>{cb_junction_type_options}</select>
              <span id="cb-junction-k-wrap">k=<input type="number" name="cb_junction_k" value="{cb_junction_k_val}" min="1" style="width:40px" {cb_disabled}></span>
              <span id="cb-junction-ab-wrap" style="display:none">a=<input type="number" name="cb_junction_a" value="{cb_junction_a_val}" min="1" style="width:40px" {cb_disabled}> b=<input type="number" name="cb_junction_b" value="{cb_junction_b_val}" min="1" style="width:40px" {cb_disabled}></span>
            </div>
            <div class="ctrl-row">
              <span style="min-width:60px">Family:</span>
              <select name="cb_family_type" id="cb-family-type" {cb_disabled}>{cb_family_type_options}</select>
              <span id="cb-family-count-wrap">count=<input type="number" name="cb_family_count" value="{cb_family_count_val}" min="2" style="width:45px" {cb_disabled}></span>
              <span id="cb-family-grid-wrap" style="display:none">rows=<input type="number" name="cb_family_rows" value="{cb_family_rows_val}" min="1" style="width:40px" {cb_disabled}> cols=<input type="number" name="cb_family_cols" value="{cb_family_cols_val}" min="1" style="width:40px" {cb_disabled}></span>
              <span id="cb-family-pattern-wrap" style="display:none">pattern=<select name="cb_family_pattern" {cb_disabled}><option value="path"{cb_pattern_path_sel}>path</option><option value="cycle"{cb_pattern_cycle_sel}>cycle</option></select></span>
            </div>
            <div class="ctrl-row" id="cb-alt-cell-wrap" style="display:none">
              <span style="min-width:60px">Alt cell:</span>
              <select name="cb_alt_cell_type" id="cb-alt-cell-type" {cb_disabled}>{cb_alt_cell_type_options}</select>
              <span id="cb-alt-cell-n-wrap">n=<input type="number" name="cb_alt_cell_n" value="{cb_alt_cell_n_val}" min="1" style="width:45px" {cb_disabled}></span>
              <span id="cb-alt-cell-ab-wrap" style="display:none">a=<input type="number" name="cb_alt_cell_a" value="{cb_alt_cell_a_val}" min="1" style="width:40px" {cb_disabled}> b=<input type="number" name="cb_alt_cell_b" value="{cb_alt_cell_b_val}" min="1" style="width:40px" {cb_disabled}></span>
              <span id="cb-alt-cell-m-wrap" style="display:none">m=<input type="number" name="cb_alt_cell_m" value="{cb_alt_cell_m_val}" min="1" style="width:40px" {cb_disabled}></span>
              <span id="cb-alt-cell-n2-wrap" style="display:none">n=<input type="number" name="cb_alt_cell_n2" value="{cb_alt_cell_n2_val}" placeholder="(=m)" min="1" style="width:55px" {cb_disabled}></span>
              <span id="cb-alt-cell-t-wrap" style="display:none">t=<input type="number" name="cb_alt_cell_t" value="{cb_alt_cell_t_val}" min="1" style="width:40px" {cb_disabled}></span>
            </div>
            <div class="ctrl-row">
              <label title="Synthesize the constructed graph and an optional baseline, then surface ✓/✗ at (0,1), (1,0), (1,1), (2,2)."><input type="checkbox" name="cb_compare_baseline" value="1" {cb_compare_checked} {cb_disabled}> Compare to baseline:</label>
              <select name="cb_baseline_choice" {cb_disabled}>{cb_baseline_options}</select>
            </div>
          </div>
        </div>
        <!-- Settings -->
        <div class="ctrl-group">
          <div class="ctrl-label">Settings</div>
          <div class="ctrl-row">
            <span>Engine:</span>
            <select name="engine">{engine_options}</select>
          </div>
          <div class="ctrl-row">
            <span>Timeout:</span>
            <input type="number" name="timeout" value="{timeout_val}" min="1" max="3600" style="width:60px">s
          </div>
          <div class="ctrl-row">
            <label><input type="checkbox" name="debug" value="1" {debug_checked}> Debug logging</label>
          </div>
          <div class="ctrl-row">
            <label title="Master switch — when off the engine consults no rainbow-table entries (top-level OR sub-problems). All polynomials are computed from scratch."><input type="checkbox" id="use-table-cb" name="use_table" value="1" {use_table_checked}> Use lookup table</label>
          </div>
          <div class="ctrl-row" style="padding-left:18px;">
            <label id="use-lookup-label" title="When the master switch is on, also consult the table for the top-level target graph by canonical key. Off → only sub-problems consult the table."><input type="checkbox" id="use-lookup-cb" name="use_lookup" value="1" {use_lookup_checked} {use_lookup_disabled}> Lookup target by canonical key</label>
          </div>
          <div class="ctrl-row" style="margin-top:12px;">
            <button type="submit" class="run-btn" style="background:#555;">Load Graph</button>
            <button type="button" class="run-btn" id="run-engine-btn" onclick="startEngine()">Run Engine</button>
            <button type="button" class="run-btn" id="stop-engine-btn" style="background:#c62828;display:none;" onclick="stopEngine()">Stop</button>
          </div>
        </div>
      </div>
    </form>
  </div>
  <script>
    // Enable/disable inputs based on radio selection
    (function() {{
      var form = document.getElementById('ctrl-form');
      var radios = form.querySelectorAll('input[name="source"]');
      function update() {{
        var sel = form.querySelector('input[name="source"]:checked').value;
        form.querySelector('input[name="atlas"]').disabled = (sel !== 'atlas');
        form.querySelector('select[name="dwave_topo"]').disabled = (sel !== 'dwave');
        form.querySelector('input[name="dwave_m"]').disabled = (sel !== 'dwave');
        form.querySelector('input[name="dwave_n"]').disabled = (sel !== 'dwave');
        form.querySelector('input[name="dwave_t"]').disabled = (sel !== 'dwave');
        form.querySelector('select[name="family"]').disabled = (sel !== 'family');
        form.querySelector('input[name="n"]').disabled = (sel !== 'family');
        form.querySelector('input[name="m"]').disabled = (sel !== 'family');
        form.querySelector('input[name="edges"]').disabled = (sel !== 'edges');
        form.querySelector('input[name="rand_n"]').disabled = (sel !== 'random');
        form.querySelector('input[name="rand_m"]').disabled = (sel !== 'random');
        // Cell builder inputs: all cb_* fields share the same disabled gate.
        var cbInputs = form.querySelectorAll('[name^="cb_"]');
        cbInputs.forEach(function(el) {{ el.disabled = (sel !== 'cell_builder'); }});
        // Toggle the cell_builder=1 hidden flag so the server only fires the
        // cell_builder branch when the radio is selected.
        var cbFlag = document.getElementById('cb-flag');
        if (cbFlag) cbFlag.value = (sel === 'cell_builder') ? '1' : '0';
        if (sel === 'dwave') updateDwave();
        if (sel === 'random') updateRandMax();
        if (sel === 'cell_builder') updateCellBuilder();
      }}
      radios.forEach(function(r) {{ r.addEventListener('change', update); }});
      update();

      // Show/hide n and m based on selected family
      var needsM = {{'complete_bipartite':1,'grid':1,'barbell':1,'balanced_tree':1,'kneser':1,'k_regular':1}};
      var fixed = {{'petersen':1,'tutte':1,'dodecahedral':1,'icosahedral':1,'octahedral':1,
        'cubical':1,'tetrahedral':1,'heawood':1,'moebius_kantor':1,'bull':1,
        'chvatal':1,'desargues':1,'pappus':1}};
      var paramLabels = {{
        'complete': ['nodes', ''],
        'cycle': ['nodes', ''],
        'path': ['nodes', ''],
        'wheel': ['nodes (incl. hub)', ''],
        'star': ['leaves', ''],
        'complete_bipartite': ['partition 1', 'partition 2'],
        'grid': ['rows', 'columns'],
        'ladder': ['rungs', ''],
        'gear': ['spokes', ''],
        'prism': ['sides', ''],
        'friendship': ['triangles', ''],
        'barbell': ['clique size', 'path length'],
        'empty': ['nodes', ''],
        'random_tree': ['nodes', ''],
        'balanced_tree': ['branching factor', 'height'],
        'kneser': ['n', 'k'],
        'k_regular': ['degree k', 'nodes'],
      }};
      var fsel = document.getElementById('family-select');
      function updateFamily() {{
        var v = fsel.value;
        document.getElementById('m-wrap').style.display = needsM[v] ? '' : 'none';
        document.getElementById('n-wrap').style.display = fixed[v] ? 'none' : '';
        var labels = paramLabels[v] || ['n', 'm'];
        document.getElementById('n-label').textContent = labels[0];
        document.getElementById('m-label').textContent = labels[1] || 'm';
      }}
      fsel.addEventListener('change', updateFamily);
      updateFamily();

      // Show/hide D-Wave params based on topology
      var dsel = document.getElementById('dwave-topo-select');
      var dwaveLabels = {{
        'zephyr':  ['grid parameter', 'tile parameter'],
        'pegasus': ['size parameter', ''],
        'chimera': ['grid parameter', ''],
      }};
      function updateDwave() {{
        var topo = dsel.value;
        // Zephyr needs (m, t). Pegasus takes just m. Chimera takes (m, n) where
        // n is optional and defaults to m.
        var showT = (topo === 'zephyr');
        var showN = (topo === 'chimera');
        document.getElementById('dwave-t-wrap').style.display = showT ? '' : 'none';
        document.getElementById('dwave-n-wrap').style.display = showN ? '' : 'none';
        var labels = dwaveLabels[topo] || ['m', 't'];
        document.getElementById('dwave-m-label').textContent = labels[0];
        document.getElementById('dwave-t-label').textContent = labels[1] || 't';
        var mInput = form.querySelector('input[name="dwave_m"]');
        if (topo === 'pegasus') {{
          mInput.min = 2;
          if (parseInt(mInput.value) < 2) mInput.value = 2;
        }} else {{
          mInput.min = 1;
        }}
      }}
      dsel.addEventListener('change', updateDwave);
      updateDwave();

      // Random graph: show max edges hint
      var randN = form.querySelector('input[name="rand_n"]');
      var randM = form.querySelector('input[name="rand_m"]');
      var randHint = document.getElementById('rand-max-edges');
      function updateRandMax() {{
        var n = parseInt(randN.value) || 0;
        var maxE = n * (n - 1) / 2;
        randHint.textContent = n > 0 ? '(max ' + maxE + ')' : '';
        if (parseInt(randM.value) > maxE && maxE > 0) randM.value = maxE;
        randM.max = maxE > 0 ? maxE : '';
      }}
      randN.addEventListener('input', updateRandMax);
      updateRandMax();

      // Cell builder: show/hide cell, junction, family, alt-cell params.
      // Cell types that take different parameter shapes:
      //   n only      → K_n, C_n, P_n
      //   a, b        → K_{{a,b}}
      //   m           → chimera (n optional), pegasus, zephyr (t separate)
      //   n (2nd)     → chimera only (optional second grid dim)
      //   t           → zephyr only
      var cellNeedsN = {{'K_n':1,'C_n':1,'P_n':1}};
      var cellNeedsAB = {{'K_a_b':1}};
      var cellNeedsM = {{'chimera':1,'pegasus':1,'zephyr':1}};
      var cellNeedsN2 = {{'chimera':1}};
      var cellNeedsT = {{'zephyr':1}};
      // Junction types: matching/k_a_b_junction takes (a,b) or k.
      // single_edge/shared_vertex take nothing.
      var junctionNeedsK = {{'matching':1}};
      var junctionNeedsAB = {{'k_a_b_junction':1}};
      // Family types: path/cycle/interleaved take count; grid takes rows,cols.
      var familyNeedsCount = {{'path':1,'cycle':1,'interleaved':1}};
      var familyNeedsGrid = {{'grid':1}};
      var familyNeedsPattern = {{'interleaved':1}};
      var familyNeedsAlt = {{'interleaved':1}};

      function _cbSetWrap(id, kind, table) {{
        var el = document.getElementById(id);
        if (el) el.style.display = table[kind] ? '' : 'none';
      }}

      function updateCellBuilder() {{
        // Guard: this function may be invoked from update() during the
        // initial IIFE run before the cellNeeds* tables below have been
        // assigned. `var` hoists the names, but the dict assignments run
        // in source order. Bail out until they exist.
        if (typeof cellNeedsN === 'undefined') return;
        var cType = document.getElementById('cb-cell-type').value;
        _cbSetWrap('cb-cell-n-wrap', cType, cellNeedsN);
        _cbSetWrap('cb-cell-ab-wrap', cType, cellNeedsAB);
        _cbSetWrap('cb-cell-m-wrap', cType, cellNeedsM);
        _cbSetWrap('cb-cell-n2-wrap', cType, cellNeedsN2);
        _cbSetWrap('cb-cell-t-wrap', cType, cellNeedsT);

        var jType = document.getElementById('cb-junction-type').value;
        _cbSetWrap('cb-junction-k-wrap', jType, junctionNeedsK);
        _cbSetWrap('cb-junction-ab-wrap', jType, junctionNeedsAB);

        var fType = document.getElementById('cb-family-type').value;
        _cbSetWrap('cb-family-count-wrap', fType, familyNeedsCount);
        _cbSetWrap('cb-family-grid-wrap', fType, familyNeedsGrid);
        _cbSetWrap('cb-family-pattern-wrap', fType, familyNeedsPattern);
        _cbSetWrap('cb-alt-cell-wrap', fType, familyNeedsAlt);

        var aType = document.getElementById('cb-alt-cell-type').value;
        _cbSetWrap('cb-alt-cell-n-wrap', aType, cellNeedsN);
        _cbSetWrap('cb-alt-cell-ab-wrap', aType, cellNeedsAB);
        _cbSetWrap('cb-alt-cell-m-wrap', aType, cellNeedsM);
        _cbSetWrap('cb-alt-cell-n2-wrap', aType, cellNeedsN2);
        _cbSetWrap('cb-alt-cell-t-wrap', aType, cellNeedsT);
      }}
      document.getElementById('cb-cell-type').addEventListener('change', updateCellBuilder);
      document.getElementById('cb-junction-type').addEventListener('change', updateCellBuilder);
      document.getElementById('cb-family-type').addEventListener('change', updateCellBuilder);
      document.getElementById('cb-alt-cell-type').addEventListener('change', updateCellBuilder);
      updateCellBuilder();
    }})();
  </script>

  <!-- Row 1: Input Graph + Contributing Graphs side by side -->
  <div class="graphs-row">
    <div class="panel">
      <h2 style="display:flex;align-items:center;justify-content:space-between;">
        <span>Input Graph — {graph_desc}</span>
        <span style="font-weight:normal;">
          <button type="button" id="clear-highlights" title="Clear all active sub-graph highlights." style="display:none;">&#x2715; subgraphs</button>
          <button type="button" id="refresh-target" title="Destroy and recreate the renderer (use after WebGL glitches).">&#x23FB;</button>
          <button type="button" id="relayout-target" title="Run ForceAtlas2 layout on the target graph (uses Barnes-Hut for large graphs).">&#x21BA;</button>
          <span id="relayout-status" style="font-size:11px;color:#888;margin-left:6px;"></span>
        </span>
      </h2>
      <div class="meta">{input_meta}</div>
      <div id="active-subgraph-labels" style="margin:4px 0;display:none;"></div>
      <div id="input-graph" class="graph-box"></div>
    </div>
    <div class="panel">
      <h2>Contributing Graphs</h2>
      <div id="minors-container" class="minors-grid">
        <div class="meta">Click "Run Engine" to start.</div>
      </div>
    </div>
  </div>

  <!-- Row 2: Result -->
  <div class="panel section">
    <h2>Result</h2>
    <div id="result-container">
      <div class="meta">Click "Run Engine" to start.</div>
    </div>
  </div>

  <!-- Row 3: Summary -->
  <div class="panel section">
    <h2>Summary</h2>
    <div id="summary-container"></div>
  </div>

  <!-- Row 4: Timeline -->
  <div class="panel section">
    <h2>Timeline <span id="event-count"></span>
      <span id="engine-elapsed" style="font-weight:normal;color:#888;font-size:12px;margin-left:12px;"></span>
    </h2>
    <div class="timeline-scroll" id="timeline-scroll">
      <table class="timeline" id="timeline-table">
        <tr><th>Time</th><th>Duration</th><th>D</th><th>Type</th><th>Module</th><th>Message</th><th>Graph</th></tr>
      </table>
    </div>
  </div>

  <!-- Row 5: Step-graph viewer -->
  <div class="panel section">
    <h2>Step Graph <span id="step-label" style="font-weight:normal;color:#888;font-size:12px;"></span></h2>
    <div class="controls" id="step-controls">
      <button type="button" id="step-first" title="Jump to first graph step.">&#x23EE;&#xFE0E;</button>
      <button type="button" id="step-prev" title="Previous graph step.">&#x23EA;&#xFE0E;</button>
      <button type="button" id="step-play" title="Auto-advance through steps.">&#x25B6;&#xFE0E;</button>
      <button type="button" id="step-next" title="Next graph step.">&#x23E9;&#xFE0E;</button>
      <button type="button" id="step-last" title="Jump to last graph step.">&#x23ED;&#xFE0E;</button>
      <button type="button" id="step-relayout" title="Re-run ForceAtlas2 on the currently-displayed step graph.">&#x21BA;</button>
      <button type="button" id="step-refresh" title="Destroy and recreate the step renderer (use after WebGL glitches).">&#x23FB;</button>
      <span style="color:#666;margin-left:8px">interval</span>
      <input type="number" id="step-interval" value="500" min="50" max="10000" step="50" style="width:70px">ms
      <span id="step-graph-count" style="color:#888;margin-left:12px;"></span>
    </div>
    <div id="step-graph" class="graph-box"></div>
  </div>

  <script>
    var EVENT_COLORS = {event_colors_json};
    // Sigma-specific utilities shared by all graph panels.
    function _sigmaNodeSize(n) {{
      if (n <= 100) return 8.0;
      if (n <= 1000) return 4.0;
      if (n <= 10000) return 2.0;
      return 1.0;
    }}
    function _sigmaSettings(order) {{
      return {{
        renderLabels: order <= 1000,
        labelDensity: 0.5,
        labelGridCellSize: 60,
        labelRenderedSizeThreshold: 6,
        defaultEdgeColor: '#b0b0b0',
        minCameraRatio: 0.05,
        maxCameraRatio: 20,
      }};
    }}
    // Cache for Sigma instances attached to minor cards so we can free WebGL
    // contexts when the contributing-graphs panel re-renders.
    window._minorSigmas = window._minorSigmas || [];
    function _killMinorSigmas() {{
      (window._minorSigmas || []).forEach(function(s) {{
        try {{ s.kill(); }} catch (e) {{}}
      }});
      window._minorSigmas = [];
    }}
    var _es = null;  // current EventSource
    var _engineTimerId = null;
    var _engineStartedAt = 0;

    function _fmtElapsed(ms) {{
      var s = ms / 1000;
      if (s < 60) return s.toFixed(1) + 's';
      var m = Math.floor(s / 60);
      var r = (s - m * 60).toFixed(1);
      return m + 'm ' + r + 's';
    }}

    function _paintElapsed() {{
      var el = _fmtElapsed(performance.now() - _engineStartedAt);
      var h = document.getElementById('engine-elapsed');
      if (h) h.textContent = 'elapsed ' + el;
      var r = document.getElementById('running-elapsed');
      if (r) r.textContent = el;
    }}

    function _startEngineClock() {{
      _engineStartedAt = performance.now();
      _paintElapsed();
      if (_engineTimerId) clearInterval(_engineTimerId);
      _engineTimerId = setInterval(_paintElapsed, 250);
    }}

    function _stopEngineClock() {{
      if (_engineTimerId) {{ clearInterval(_engineTimerId); _engineTimerId = null; }}
      _paintElapsed();
    }}

    window.addEventListener('DOMContentLoaded', function() {{
      // Render input graph
      {input_graph_script}

      // Parent/child checkbox: when "Use lookup table" is unchecked,
      // disable "Lookup target by canonical key" — the child option
      // is meaningless without the parent. When the parent re-enables,
      // the child re-enables (preserving its prior checked state).
      var _useTableCb = document.getElementById('use-table-cb');
      var _useLookupCb = document.getElementById('use-lookup-cb');
      var _useLookupLabel = document.getElementById('use-lookup-label');
      if (_useTableCb && _useLookupCb) {{
        function _syncChild() {{
          var on = _useTableCb.checked;
          _useLookupCb.disabled = !on;
          if (_useLookupLabel) {{
            _useLookupLabel.style.opacity = on ? '' : '0.5';
          }}
        }}
        _useTableCb.addEventListener('change', _syncChild);
        _syncChild();
      }}
    }});

    function stopEngine() {{
      if (_es) {{ _es.close(); _es = null; }}
      _stopEngineClock();
      document.getElementById('run-engine-btn').style.display = '';
      document.getElementById('stop-engine-btn').style.display = 'none';
    }}

    function startEngine() {{
      // Close any previous connection
      if (_es) {{ _es.close(); _es = null; }}

      // Show stop button, hide run button
      document.getElementById('run-engine-btn').style.display = 'none';
      document.getElementById('stop-engine-btn').style.display = '';

      // Free WebGL contexts held by minor cards and step viewer from a
      // prior run before we clear the containers.
      _killMinorSigmas();
      if (window._stepSigmaInstance) {{
        try {{ window._stepSigmaInstance.kill(); }} catch (e) {{}}
        window._stepSigmaInstance = null;
      }}

      // Reset UI
      var tbody = document.getElementById('timeline-table');
      while (tbody.rows.length > 1) tbody.deleteRow(1);
      document.getElementById('event-count').textContent = '';
      document.getElementById('result-container').innerHTML =
        '<div class="meta"><span class="spinner"></span>Running engine... (<span id="running-elapsed">0.0s</span>)</div>';
      document.getElementById('minors-container').innerHTML = '<div class="meta"><span class="spinner"></span>Waiting for engine...</div>';
      document.getElementById('summary-container').innerHTML = '';
      // Reset step-graph viewer
      document.getElementById('step-graph').innerHTML = '';
      document.getElementById('step-label').textContent = '';
      document.getElementById('step-graph-count').textContent = '';

      _startEngineClock();
      {sse_script}
    }}
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------

# Global state for the current engine run (single-user visualizer)
_engine_lock = threading.Lock()
_engine_state = {
    "running": False,
    "result": None,
    "error": None,
    "timed_out": False,
    "elapsed": 0.0,
    "done": False,
}


def _run_engine_thread(graph, table, timeout_sec, engine_type, skip_target_lookup=False):
    """Run the engine in a thread, storing result in _engine_state.

    If skip_target_lookup is True, the top-level synthesize() call skips the
    rainbow-table lookup for the input graph — but the engine still uses the
    full table for sub-problems encountered during decomposition.
    """
    global _engine_state
    result_holder = [None]
    error_holder = [None]

    def target():
        try:
            engine = SynthesisEngine(table=table)
            engine.skip_target_lookup = skip_target_lookup
            # Promote cache entries to the rainbow table at
            # end-of-synthesis so cache_hits become lookup_hits
            # on the next visualizer run.
            engine.promote_cache_on_finish = True
            synth_result = engine.synthesize(graph)
            result_holder[0] = SynthesisResult(
                polynomial=synth_result.polynomial,
                recipe=synth_result.recipe,
                verified=synth_result.verified,
                method=synth_result.method,
                tiles_used=getattr(synth_result, 'tiles_used', 0),
                minors_used=getattr(synth_result, 'minors_used', set()),
                synthesized_minors=getattr(synth_result, 'synthesized_minors', set()),
                synthesized_graphs=getattr(synth_result, 'synthesized_graphs', {}),
            )
        except Exception as e:
            error_holder[0] = str(e)

    t0 = time.perf_counter()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)
    elapsed = time.perf_counter() - t0

    with _engine_lock:
        _engine_state["elapsed"] = elapsed
        if thread.is_alive():
            _engine_state["timed_out"] = True
        elif error_holder[0]:
            _engine_state["error"] = error_holder[0]
        else:
            _engine_state["result"] = result_holder[0]
        _engine_state["done"] = True


@app.route("/stream")
def stream():
    """SSE endpoint: streams events as they're recorded, then final result."""
    timeout_sec = request.args.get("timeout", 60, type=int)
    engine_type = request.args.get("engine", "synthesis")
    threshold_ms = request.args.get("threshold", 100, type=float)
    debug = request.args.get("debug", "0") == "1"
    cb_compare_baseline = request.args.get("cb_compare_baseline", "0") == "1"
    cb_baseline_choice = request.args.get("cb_baseline_choice", "Cm1")
    # form_submitted sentinel distinguishes "no form submit" from "unchecked"
    form_submitted = request.args.get("form_submitted", "0") == "1"
    if form_submitted:
        use_table = request.args.get("use_table", "0") == "1"
        use_lookup = request.args.get("use_lookup", "0") == "1"
    else:
        use_table = True
        use_lookup = True
    # Child can't be on without parent.
    if not use_table:
        use_lookup = False

    G_nx, graph_desc, _source_hint = parse_graph(request.args)
    if G_nx is None:
        def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': graph_desc or 'No graph'})}\n\n"
        return Response(error_stream(), mimetype="text/event-stream")

    graph = Graph.from_networkx(G_nx)
    # Master switch: when use_table is False, pass an empty
    # RainbowTable so neither the top-level nor sub-problem lookups
    # find anything. `skip_target_lookup` then only matters when the
    # table is non-empty.
    if use_table:
        table = load_default_table()
    else:
        from tutte.lookup.core import RainbowTable
        table = RainbowTable()
    skip_target_lookup = not use_lookup

    # Reset log and enable graph snapshot capture for the viewer. We
    # always record at DEBUG level so cache_hit + other DEBUG-level
    # events (and their graph snapshots) are surfaced in Contributing
    # Graphs. The `debug` URL param controls UI filtering, not capture
    # level — the client can hide DEBUG rows without losing snapshots.
    reset_log()
    log_ref = get_log()
    log_ref.min_level = LogLevel.DEBUG
    log_ref.capture_graphs = True
    global _engine_state
    with _engine_lock:
        _engine_state = {
            "running": True, "result": None, "error": None,
            "timed_out": False, "elapsed": 0.0, "done": False,
        }

    # Start engine in background thread
    engine_thread = threading.Thread(
        target=_run_engine_thread,
        args=(graph, table, timeout_sec, engine_type, skip_target_lookup),
        daemon=True,
    )
    engine_thread.start()

    def _serialize_batch(new_events, start_idx, prev_timestamp):
        """Serialize a list of new events into a single SSE batch message.

        Returns (batch, last_timestamp). Each event dict may include a
        `graph_key` field referencing a snapshot sent in the same batch.
        """
        batch = []
        for i, ev in enumerate(new_events):
            idx = start_idx + i
            gap = ev.timestamp - prev_timestamp if prev_timestamp is not None else 0.0
            color = EVENT_COLORS.get(ev.event_type.value, "#9e9e9e")
            batch.append({
                "index": idx,
                "timestamp": f"{ev.timestamp:.3f}s",
                "ts_raw": ev.timestamp,
                "depth": ev.depth,
                "event_type": ev.event_type.value,
                "module": ev.module,
                "message": ev.message,
                "color": color,
                "gap": gap,
                "graph_key": ev.graph_key,
            })
            prev_timestamp = ev.timestamp
        return batch, prev_timestamp

    def event_stream():
        log = get_log()
        sent_count = 0
        prev_timestamp = None
        poll_interval = 0.05  # 50ms
        sent_snapshot_keys: set = set()

        def _batch_payload(events, start_idx, prev_ts):
            batch, new_prev_ts = _serialize_batch(events, start_idx, prev_ts)
            new_snapshots = log.new_graph_snapshots(sent_snapshot_keys)
            if new_snapshots:
                sent_snapshot_keys.update(new_snapshots.keys())
            payload = {'type': 'batch', 'events': batch}
            if new_snapshots:
                payload['snapshots'] = new_snapshots
            return payload, new_prev_ts

        while True:
            new_events = log.events_since(sent_count)

            if new_events:
                payload, prev_timestamp = _batch_payload(
                    new_events, sent_count, prev_timestamp
                )
                sent_count += len(new_events)
                yield f"data: {json.dumps(payload)}\n\n"

            # Check if engine is done
            with _engine_lock:
                done = _engine_state["done"]

            if done:
                # Send any remaining events
                remaining = log.events_since(sent_count)
                if remaining:
                    payload, prev_timestamp = _batch_payload(
                        remaining, sent_count, prev_timestamp
                    )
                    sent_count += len(remaining)
                    yield f"data: {json.dumps(payload)}\n\n"

                # Build final result payload
                with _engine_lock:
                    result = _engine_state["result"]
                    error = _engine_state["error"]
                    timed_out = _engine_state["timed_out"]
                    elapsed = _engine_state["elapsed"]

                # Re-send the final snapshot dict so the client picks
                # up provenance that was added AFTER the snapshot was
                # first streamed (e.g., the input graph's snapshot is
                # captured at SYNTHESIS_START with no provenance, then
                # later cells/inter-cell records augment it). Cheap
                # because snapshots are small dicts.
                final_snapshots = {
                    k: snap for k, snap in log._graph_snapshots.items()
                    if snap.get("provenance")
                }

                final = {
                    "type": "done",
                    "timed_out": timed_out,
                    "elapsed": elapsed,
                    "event_count": sent_count,
                    "threshold_ms": threshold_ms,
                    "snapshots_with_provenance": final_snapshots,
                }

                if timed_out:
                    final["result_html"] = (
                        f'<div class="timeout-banner">'
                        f'TIMEOUT after {elapsed:.1f}s (limit: {timeout_sec}s)<br>'
                        f'Check timeline for last event before timeout.'
                        f'</div>'
                    )
                    final["minors"] = []
                    final["minors_lookup"] = []
                    final["minors_synthesized"] = []
                elif error:
                    final["result_html"] = (
                        f'<div class="error-banner">Engine error: {error}</div>'
                    )
                    final["minors"] = []
                    final["minors_lookup"] = []
                    final["minors_synthesized"] = []
                elif result is not None:
                    poly_html = factored_poly_html(result.polynomial)
                    t11 = result.polynomial.num_spanning_trees()
                    verified_str = (
                        '<span style="color:#2e7d32">YES</span>' if result.verified
                        else '<span style="color:#c62828">NO</span>'
                    )
                    # Deferred: Tiles + Sub-graphs are computed after
                    # the contributing-graphs lists are built below, so
                    # the result_html is assembled at that point.
                    _result_meta = {
                        "method": result.method,
                        "verified_str": verified_str,
                        "tiles": result.tiles_used,
                        "t11": t11,
                        "elapsed": elapsed,
                        "poly_html": poly_html,
                    }

                    # Optional baseline comparison: synth a small baseline
                    # graph on a fresh engine, then compare evaluations at
                    # (0,1), (1,0), (1,1), (2,2). Synchronous: small
                    # baselines (<= 32 vertices) finish in < a few seconds.
                    _result_meta["baseline_html"] = ""
                    if cb_compare_baseline:
                        try:
                            from tutte.scripts.cell_builder import compare_to_baseline
                            from tutte.synthesis.engine import SynthesisEngine
                            base_nx, base_label = _build_baseline_graph(
                                cb_baseline_choice,
                            )
                            base_graph = Graph.from_networkx(base_nx)
                            base_engine = SynthesisEngine(verbose=False)
                            base_engine.skip_target_lookup = skip_target_lookup
                            base_result = base_engine.synthesize(base_graph)
                            comparison = compare_to_baseline(
                                result.polynomial, base_result.polynomial,
                            )
                            rows = []
                            for (x, y), info in comparison.items():
                                ok = info["dominates"]
                                mark = (
                                    '<span style="color:#2e7d32">&#x2713;</span>'
                                    if ok else
                                    '<span style="color:#c62828">&#x2717;</span>'
                                )
                                rows.append(
                                    f'<tr><td>{mark}</td>'
                                    f'<td>T({x},{y})</td>'
                                    f'<td>{info["constructed"]}</td>'
                                    f'<td>{info["baseline"]}</td></tr>'
                                )
                            _result_meta["baseline_html"] = (
                                f'<table style="font-size:11px;border-collapse:collapse;">'
                                f'<thead><tr><th></th><th>Point</th>'
                                f'<th>Constructed</th><th>Baseline ({base_label})</th></tr></thead>'
                                f'<tbody>{"".join(rows)}</tbody></table>'
                            )
                        except Exception as e:
                            _result_meta["baseline_html"] = (
                                f'<span style="color:#c62828">'
                                f'Baseline comparison failed: {e}</span>'
                            )
                    # Build contributing graphs from the EVENT STREAM so
                    # every unique graph that appeared in the Timeline /
                    # Step Graph is surfaced here too:
                    #   lookup_hit + cache_hit → "From Lookup Table"
                    #   everything else → "Synthesized During Run"
                    #
                    # Dedupe by *structural signature* rather than canonical
                    # key: Graph.canonical_key() and MultiGraph.canonical_key()
                    # differ for the same underlying structure, so without
                    # normalization treewidth_dp emits two cards for the same
                    # graph (engine's simple Graph + treewidth.py's MultiGraph).
                    # The structural signature is derived from the serialized
                    # snapshot (sorted nodes + sorted multi-edges + sorted
                    # loops) and collapses both representations to one.
                    LOOKUP_EVENT_TYPES = {"lookup_hit", "cache_hit"}

                    def _struct_sig(k: str) -> str:
                        """Return a structural signature derived from the log
                        snapshot, so simple Graph and MultiGraph of the same
                        underlying structure map to the same bucket."""
                        snap = log.graph_snapshot(k)
                        if not isinstance(snap, dict):
                            return k  # fallback: unique per canonical_key
                        nodes = tuple(sorted(snap.get("nodes", []) or []))
                        edges = []
                        for e in snap.get("edges", []) or []:
                            if not e:
                                continue
                            u = e[0]
                            v = e[1] if len(e) > 1 else e[0]
                            mult = e[2] if len(e) > 2 else 1
                            edges.append((min(u, v), max(u, v), mult))
                        loops = []
                        for ln in snap.get("loops", []) or []:
                            if not ln:
                                continue
                            n = ln[0]
                            mult = ln[1] if len(ln) > 1 else 1
                            loops.append((n, mult))
                        return repr((nodes, tuple(sorted(edges)), tuple(sorted(loops))))

                    lookup_keys_ordered: list = []
                    synth_keys_ordered: list = []
                    lookup_sigs: dict = {}   # sig → key (first-seen)
                    synth_sigs: dict = {}

                    for ev in log.events:
                        k = ev.graph_key
                        if not k:
                            continue
                        sig = _struct_sig(k)
                        etype = ev.event_type.value
                        if etype in LOOKUP_EVENT_TYPES:
                            if sig not in lookup_sigs:
                                lookup_sigs[sig] = k
                                lookup_keys_ordered.append(k)
                            # If previously bucketed as synth, promote to
                            # lookup (lookup/cache wins).
                            if sig in synth_sigs:
                                old_k = synth_sigs.pop(sig)
                                synth_keys_ordered = [
                                    kk for kk in synth_keys_ordered if kk != old_k
                                ]
                        else:
                            if sig in lookup_sigs:
                                continue
                            if sig not in synth_sigs:
                                synth_sigs[sig] = k
                                synth_keys_ordered.append(k)

                    # Augment with any rainbow-table entries from
                    # result.minors_used that didn't fire a graph-bearing
                    # event (defensive: keeps the old behavior intact).
                    for k in sorted(result.minors_used or []):
                        sig = _struct_sig(k)
                        if sig not in lookup_sigs and sig not in synth_sigs:
                            lookup_sigs[sig] = k
                            lookup_keys_ordered.append(k)

                    def _provenance_count(snap) -> int:
                        if not isinstance(snap, dict):
                            return 0
                        prov = snap.get("provenance") or []
                        return len(prov)

                    def _lookup_card(k: str) -> dict:
                        entry = table.get_entry_by_key(k)
                        snap = log.graph_snapshot(k)
                        prov_n = _provenance_count(snap)
                        if entry is not None:
                            card = {"name": entry.name, "edges": entry.edge_count, "key": k, "provenance_count": prov_n}
                            minor_nx = graph_from_entry(entry)
                            if minor_nx is not None:
                                card["sigma_json"] = sigma_graph_json(minor_nx)
                            return card
                        # No rainbow-table entry: render from snapshot.
                        nc, ec = _snapshot_counts(snap)
                        name = f"cached {nc}n {ec}e [{k[:8]}]"
                        card = {"name": name, "edges": ec, "key": k, "provenance_count": prov_n}
                        minor_nx = _snapshot_to_nx(snap) if snap else None
                        if minor_nx is not None:
                            card["sigma_json"] = sigma_graph_json(minor_nx)
                        return card

                    def _synth_card(k: str) -> dict:
                        # Prefer the richer graph object from
                        # _synth_accum_graphs when available (supports the
                        # existing `_synth_graph_to_nx` path); fall back to
                        # the log snapshot otherwise.
                        synthesized_graphs = getattr(
                            result, 'synthesized_graphs', {}
                        ) or {}
                        g_obj = synthesized_graphs.get(k)
                        snap = log.graph_snapshot(k)
                        prov_n = _provenance_count(snap)
                        if g_obj is not None:
                            minor_nx = _synth_graph_to_nx(g_obj)
                            nc = getattr(g_obj, 'node_count', lambda: '?')()
                            ec = getattr(g_obj, 'edge_count', lambda: '?')()
                            name = (
                                f"{type(g_obj).__name__} {nc}n {ec}e "
                                f"[{k[:8]}]"
                            )
                            card = {"name": name, "edges": ec, "key": k, "provenance_count": prov_n}
                            if minor_nx is not None:
                                card["sigma_json"] = sigma_graph_json(minor_nx)
                            return card
                        nc, ec = _snapshot_counts(snap)
                        name = f"snapshot {nc}n {ec}e [{k[:8]}]"
                        card = {"name": name, "edges": ec, "key": k, "provenance_count": prov_n}
                        minor_nx = _snapshot_to_nx(snap) if snap else None
                        if minor_nx is not None:
                            card["sigma_json"] = sigma_graph_json(minor_nx)
                        return card

                    lookup_list = [_lookup_card(k) for k in lookup_keys_ordered]
                    synth_list = [_synth_card(k) for k in synth_keys_ordered]

                    final["minors"] = lookup_list  # back-compat: legacy key
                    final["minors_lookup"] = lookup_list
                    final["minors_synthesized"] = synth_list

                    # Build result_html with both Tiles (top-level cell
                    # partition count) and Sub-graphs (total unique graphs
                    # contributing to the synthesis, including all cache
                    # and lookup hits).
                    total_subgraphs = len(lookup_list) + len(synth_list)
                    baseline_dt = (
                        f'<dt>Baseline</dt><dd>{_result_meta["baseline_html"]}</dd>'
                        if _result_meta["baseline_html"] else ""
                    )
                    final["result_html"] = (
                        f'<dl class="result-grid">'
                        f'<dt>Method</dt><dd>{_result_meta["method"]}</dd>'
                        f'<dt>Verified</dt><dd>{_result_meta["verified_str"]}</dd>'
                        f'<dt>Tiles</dt><dd>{_result_meta["tiles"]}'
                        f' <span style="color:#888;font-size:11px">(top-level cell partition)</span></dd>'
                        f'<dt>Sub-graphs</dt><dd>{total_subgraphs}'
                        f' <span style="color:#888;font-size:11px">'
                        f'({len(lookup_list)} lookup/cache + {len(synth_list)} synthesized)</span></dd>'
                        f'<dt>T(1,1)</dt><dd>{_result_meta["t11"]}</dd>'
                        f'<dt>Time</dt><dd>{_result_meta["elapsed"]:.3f}s</dd>'
                        f'{baseline_dt}'
                        f'<dt>Polynomial</dt><dd class="poly">{_result_meta["poly_html"]}</dd>'
                        f'</dl>'
                    )
                else:
                    final["result_html"] = (
                        '<div class="error-banner">Engine returned no result.</div>'
                    )
                    final["minors"] = []
                    final["minors_lookup"] = []
                    final["minors_synthesized"] = []

                # Build summary from log aggregation
                summary_data = log.summary()
                total_time = prev_timestamp if prev_timestamp is not None else 0.0
                if summary_data and total_time > 0:
                    sorted_items = sorted(
                        ((et.value, (c, d)) for et, (c, d) in summary_data.items()),
                        key=lambda x: -x[1][1],
                    )
                    summary_rows = []
                    for etype, (count, total_dur) in sorted_items:
                        pct = (total_dur / total_time) * 100 if total_time > 0 else 0
                        color = EVENT_COLORS.get(etype, "#9e9e9e")
                        summary_rows.append({
                            "event_type": etype, "color": color,
                            "count": count, "duration": f"{total_dur:.3f}s",
                            "pct": f"{pct:.1f}%", "pct_num": round(pct, 1),
                        })
                    final["summary"] = summary_rows
                else:
                    final["summary"] = []

                yield f"data: {json.dumps(final)}\n\n"
                return

            time.sleep(poll_interval)

    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    timeout_sec = request.args.get("timeout", 60, type=int)
    threshold_ms = request.args.get("threshold", 100, type=float)
    engine_type = request.args.get("engine", "synthesis")
    debug = request.args.get("debug", "0") == "1"
    # use_table + use_lookup default to ON. Unchecking submits without
    # the param, so we detect form submission via a hidden sentinel.
    form_submitted = request.args.get("form_submitted", "0") == "1"
    if form_submitted:
        use_table = request.args.get("use_table", "0") == "1"
        use_lookup = request.args.get("use_lookup", "0") == "1"
    else:
        use_table = True
        use_lookup = True
    # Child can't be on without parent.
    if not use_table:
        use_lookup = False

    atlas_val = request.args.get("atlas", "")
    dwave_topo_val = request.args.get("dwave_topo", "zephyr")
    dwave_m_val = request.args.get("dwave_m", "1")
    dwave_n_val = request.args.get("dwave_n", "")
    dwave_t_val = request.args.get("dwave_t", "1")
    edges_val = request.args.get("edges", "")
    family_val = request.args.get("family", "")
    n_val = request.args.get("n", "5")
    m_val = request.args.get("m", "")

    rand_n_val = request.args.get("rand_n", "12")
    rand_m_val = request.args.get("rand_m", "12")

    # Cell builder values (mirrored from request args, with sensible defaults)
    cb_cell_type_val = request.args.get("cb_cell_type", "K_a_b")
    cb_cell_n_val = request.args.get("cb_cell_n", "4")
    cb_cell_a_val = request.args.get("cb_cell_a", "4")
    cb_cell_b_val = request.args.get("cb_cell_b", "4")
    cb_cell_m_val = request.args.get("cb_cell_m", "1")
    cb_cell_n2_val = request.args.get("cb_cell_n2", "")
    cb_cell_t_val = request.args.get("cb_cell_t", "1")
    cb_junction_type_val = request.args.get("cb_junction_type", "matching")
    cb_junction_k_val = request.args.get("cb_junction_k", "4")
    cb_junction_a_val = request.args.get("cb_junction_a", "4")
    cb_junction_b_val = request.args.get("cb_junction_b", "4")
    cb_family_type_val = request.args.get("cb_family_type", "path")
    cb_family_count_val = request.args.get("cb_family_count", "3")
    cb_family_rows_val = request.args.get("cb_family_rows", "2")
    cb_family_cols_val = request.args.get("cb_family_cols", "2")
    cb_family_pattern_val = request.args.get("cb_family_pattern", "path")
    cb_alt_cell_type_val = request.args.get("cb_alt_cell_type", "C_n")
    cb_alt_cell_n_val = request.args.get("cb_alt_cell_n", "8")
    cb_alt_cell_a_val = request.args.get("cb_alt_cell_a", "4")
    cb_alt_cell_b_val = request.args.get("cb_alt_cell_b", "4")
    cb_alt_cell_m_val = request.args.get("cb_alt_cell_m", "1")
    cb_alt_cell_n2_val = request.args.get("cb_alt_cell_n2", "")
    cb_alt_cell_t_val = request.args.get("cb_alt_cell_t", "1")
    cb_compare_baseline = request.args.get("cb_compare_baseline", "0") == "1"
    cb_baseline_choice_val = request.args.get("cb_baseline_choice", "Cm1")

    # Determine which source is active
    source = request.args.get("source", "")
    if not source:
        if atlas_val:
            source = "atlas"
        elif request.args.get("dwave_topo"):
            source = "dwave"
        elif family_val:
            source = "family"
        elif edges_val:
            source = "edges"
        elif request.args.get("rand_n"):
            source = "random"
        elif request.args.get("cell_builder") == "1":
            source = "cell_builder"
        else:
            source = "atlas"

    atlas_checked = "checked" if source == "atlas" else ""
    dwave_checked = "checked" if source == "dwave" else ""
    family_checked = "checked" if source == "family" else ""
    edges_checked = "checked" if source == "edges" else ""
    random_checked = "checked" if source == "random" else ""
    cell_builder_checked = "checked" if source == "cell_builder" else ""
    atlas_disabled = "" if source == "atlas" else "disabled"
    dwave_disabled = "" if source == "dwave" else "disabled"
    family_disabled = "" if source == "family" else "disabled"
    edges_disabled = "" if source == "edges" else "disabled"
    random_disabled = "" if source == "random" else "disabled"
    cb_disabled = "" if source == "cell_builder" else "disabled"
    cb_flag_val = "1" if source == "cell_builder" else "0"

    engine_options = ""
    for opt in ["synthesis"]:
        sel = " selected" if opt == engine_type else ""
        engine_options += f'<option value="{opt}"{sel}>{opt}</option>'

    # Build family dropdown options
    family_options = '<option value="">-- select --</option>'
    for key, (label, _needs_m) in GRAPH_FAMILIES.items():
        sel = " selected" if key == family_val else ""
        family_options += f'<option value="{key}"{sel}>{label}</option>'

    # Build D-Wave topology dropdown
    dwave_topo_options = ""
    for topo, label in [("zephyr", "Zephyr Z(m, t)"), ("pegasus", "Pegasus P(m)"), ("chimera", "Chimera C(m)")]:
        sel = " selected" if topo == dwave_topo_val else ""
        dwave_topo_options += f'<option value="{topo}"{sel}>{label}</option>'

    # Build cell-builder dropdowns
    _cb_cell_choices = [
        ("K_n", "K_n (complete)"),
        ("K_a_b", "K_{a,b} (bipartite)"),
        ("C_n", "C_n (cycle)"),
        ("P_n", "P_n (path)"),
        ("chimera", "Chimera Cm(m, n)"),
        ("pegasus", "Pegasus Pm(m)"),
        ("zephyr", "Zephyr Z(m, t)"),
    ]
    cb_cell_type_options = "".join(
        f'<option value="{k}"{" selected" if k == cb_cell_type_val else ""}>{lbl}</option>'
        for k, lbl in _cb_cell_choices
    )
    cb_alt_cell_type_options = "".join(
        f'<option value="{k}"{" selected" if k == cb_alt_cell_type_val else ""}>{lbl}</option>'
        for k, lbl in _cb_cell_choices
    )
    _cb_junction_choices = [
        ("matching", "matching M_k"),
        ("single_edge", "single edge"),
        ("shared_vertex", "shared vertex"),
        ("k_a_b_junction", "K_{a,b} junction"),
    ]
    cb_junction_type_options = "".join(
        f'<option value="{k}"{" selected" if k == cb_junction_type_val else ""}>{lbl}</option>'
        for k, lbl in _cb_junction_choices
    )
    _cb_family_choices = [
        ("path", "path"),
        ("cycle", "cycle"),
        ("grid", "grid"),
        ("interleaved", "interleaved"),
    ]
    cb_family_type_options = "".join(
        f'<option value="{k}"{" selected" if k == cb_family_type_val else ""}>{lbl}</option>'
        for k, lbl in _cb_family_choices
    )
    cb_pattern_path_sel = " selected" if cb_family_pattern_val == "path" else ""
    cb_pattern_cycle_sel = " selected" if cb_family_pattern_val == "cycle" else ""
    cb_compare_checked = "checked" if cb_compare_baseline else ""
    _cb_baseline_choices = [
        ("K_5", "K_5"),
        ("K_4_4", "K_{4,4}"),
        ("C_5", "C_5"),
        ("Cm1", "Cm1 (K_{4,4})"),
        ("Z11", "Z(1,1)"),
    ]
    cb_baseline_options = "".join(
        f'<option value="{k}"{" selected" if k == cb_baseline_choice_val else ""}>{lbl}</option>'
        for k, lbl in _cb_baseline_choices
    )

    G_nx, graph_desc, source_hint = parse_graph(request.args)

    debug_checked = "checked" if debug else ""
    use_lookup_checked = "checked" if use_lookup else ""
    use_table_checked = "checked" if use_table else ""
    use_lookup_disabled = "" if use_table else "disabled"

    # Compute server-side labels for D-Wave params
    _dwave_labels = {
        "zephyr": ("grid parameter", "tile parameter"),
        "pegasus": ("size parameter", ""),
        "chimera": ("grid parameter", ""),
    }
    dwave_m_label, dwave_t_label = _dwave_labels.get(dwave_topo_val, ("m", "t"))
    # Only Zephyr exposes a tile parameter; Pegasus and Chimera hide the t input.
    dwave_t_display = "" if dwave_topo_val == "zephyr" else "display:none"
    if not dwave_t_label:
        dwave_t_label = "t"
    # Only Chimera exposes the optional second grid dimension n.
    dwave_n_display = "" if dwave_topo_val == "chimera" else "display:none"

    # Compute server-side labels for family params
    _family_labels = {
        "complete": ("nodes", ""),
        "cycle": ("nodes", ""),
        "path": ("nodes", ""),
        "wheel": ("nodes (incl. hub)", ""),
        "star": ("leaves", ""),
        "complete_bipartite": ("partition 1", "partition 2"),
        "grid": ("rows", "columns"),
        "ladder": ("rungs", ""),
        "gear": ("spokes", ""),
        "prism": ("sides", ""),
        "friendship": ("triangles", ""),
        "barbell": ("clique size", "path length"),
        "empty": ("nodes", ""),
        "random_tree": ("nodes", ""),
        "balanced_tree": ("branching factor", "height"),
        "kneser": ("n", "k"),
        "k_regular": ("degree k", "nodes"),
    }
    n_label, m_label = _family_labels.get(family_val, ("n", "m"))
    if not m_label:
        m_label = "m"

    # Compute random graph max edges hint
    try:
        _rn = int(rand_n_val)
        rand_max_hint = f"(max {_rn * (_rn - 1) // 2})" if _rn > 0 else ""
    except (ValueError, TypeError):
        rand_max_hint = ""

    ctrl_vars = dict(
        atlas_val=atlas_val, dwave_m_val=dwave_m_val,
        dwave_n_val=dwave_n_val, dwave_t_val=dwave_t_val,
        dwave_topo_options=dwave_topo_options,
        dwave_m_label=dwave_m_label, dwave_t_label=dwave_t_label,
        dwave_t_display=dwave_t_display,
        dwave_n_display=dwave_n_display,
        n_label=n_label, m_label=m_label,
        timeout_val=timeout_sec, engine_options=engine_options,
        atlas_checked=atlas_checked, dwave_checked=dwave_checked,
        family_checked=family_checked, edges_checked=edges_checked,
        random_checked=random_checked,
        atlas_disabled=atlas_disabled, dwave_disabled=dwave_disabled,
        family_disabled=family_disabled, edges_disabled=edges_disabled,
        random_disabled=random_disabled,
        family_options=family_options, n_val=n_val, m_val=m_val,
        edges_val=edges_val, debug_checked=debug_checked,
        use_lookup_checked=use_lookup_checked,
        use_table_checked=use_table_checked,
        use_lookup_disabled=use_lookup_disabled,
        rand_n_val=rand_n_val, rand_m_val=rand_m_val,
        rand_max_hint=rand_max_hint,
        cell_builder_checked=cell_builder_checked,
        cb_disabled=cb_disabled,
        cb_flag_val=cb_flag_val,
        cb_cell_type_options=cb_cell_type_options,
        cb_cell_n_val=cb_cell_n_val,
        cb_cell_a_val=cb_cell_a_val,
        cb_cell_b_val=cb_cell_b_val,
        cb_cell_m_val=cb_cell_m_val,
        cb_cell_n2_val=cb_cell_n2_val,
        cb_cell_t_val=cb_cell_t_val,
        cb_junction_type_options=cb_junction_type_options,
        cb_junction_k_val=cb_junction_k_val,
        cb_junction_a_val=cb_junction_a_val,
        cb_junction_b_val=cb_junction_b_val,
        cb_family_type_options=cb_family_type_options,
        cb_family_count_val=cb_family_count_val,
        cb_family_rows_val=cb_family_rows_val,
        cb_family_cols_val=cb_family_cols_val,
        cb_pattern_path_sel=cb_pattern_path_sel,
        cb_pattern_cycle_sel=cb_pattern_cycle_sel,
        cb_alt_cell_type_options=cb_alt_cell_type_options,
        cb_alt_cell_n_val=cb_alt_cell_n_val,
        cb_alt_cell_a_val=cb_alt_cell_a_val,
        cb_alt_cell_b_val=cb_alt_cell_b_val,
        cb_alt_cell_m_val=cb_alt_cell_m_val,
        cb_alt_cell_n2_val=cb_alt_cell_n2_val,
        cb_alt_cell_t_val=cb_alt_cell_t_val,
        cb_compare_checked=cb_compare_checked,
        cb_baseline_options=cb_baseline_options,
    )

    # No graph provided — empty state
    if G_nx is None:
        desc = graph_desc if graph_desc else "none"
        meta = graph_desc if graph_desc else "Select a graph using the controls above."
        no_graph_script = """
          document.getElementById('run-engine-btn').disabled = true;
          document.getElementById('run-engine-btn').style.opacity = '0.4';
        """
        page = HTML.format(
            **ctrl_vars,
            graph_desc=desc, input_meta=meta,
            event_colors_json=json.dumps(EVENT_COLORS),
            input_graph_script=no_graph_script, sse_script="",
        )
        return Response(page, mimetype="text/html")

    # Build input graph metadata
    n = G_nx.number_of_nodes()
    m = G_nx.number_of_edges()
    connected = nx.is_connected(G_nx) if n > 0 else False
    deg_seq = sorted([d for _, d in G_nx.degree()], reverse=True)
    circuit_rank = m - n + (nx.number_connected_components(G_nx) if n > 0 else 0)

    # Cap the rendered degree sequence so HTML stays compact on huge graphs.
    if len(deg_seq) > 40:
        _deg_display = f"{deg_seq[:20]} … {deg_seq[-20:]} (len {len(deg_seq)})"
    else:
        _deg_display = str(deg_seq)
    input_meta = (
        f"Nodes: {n} &nbsp; Edges: {m} &nbsp; Connected: {connected}<br>"
        f"Degree seq: {_deg_display}<br>"
        f"Circuit rank: {circuit_rank}"
    )

    # Build input-graph render script (Sigma.js + graphology) and embed
    # the target canonical key + positions so the step viewer can align
    # matching snapshots to the same layout.
    #
    # For large graphs we skip canonical_key — it runs Weisfeiler-Lehman
    # on the full graph, which is seconds-to-minutes at Z(12,4) scale.
    # Losing the key means the step viewer just can't skip layout work on
    # the first snapshot; the visualizer still works.
    _KEY_NODE_THRESHOLD = 1500
    input_graph_script = ""
    if n > 0:
        if n <= _KEY_NODE_THRESHOLD:
            try:
                _tutte_target = Graph.from_networkx(G_nx)
                _target_key = _tutte_target.canonical_key()
            except Exception:
                _target_key = None
        else:
            _target_key = None
        _target_pos = compute_layout(G_nx, source_hint=source_hint)
        _render_snippet = sigma_graph_vis(
            G_nx, "input-graph",
            source_hint=source_hint, pos=_target_pos,
            register_as_target=True,
        )
        # Only embed the positions dict when we also have a canonical key
        # to match snapshots against — otherwise nothing in renderStep can
        # use it, and duplicating 4,800+ coordinates in HTML just bloats
        # the page.
        if _target_key is not None:
            _pos_js = {str(k): [float(v[0]), float(v[1])] for k, v in _target_pos.items()}
            _pos_embed = f"window._inputGraphPositions = {json.dumps(_pos_js)};\n"
        else:
            _pos_embed = "window._inputGraphPositions = null;\n"
        input_graph_script = (
            f"window._inputGraphKey = {json.dumps(_target_key)};\n"
            f"{_pos_embed}"
            f"window._targetLayoutSource = {json.dumps(source_hint)};\n"
            f"{_render_snippet}\n"
            # Re-layout button — runs FA2 (Barnes-Hut for large graphs) on the
            # target Sigma instance. Disabled if graphology-library is missing.
            "var _rlBtn = document.getElementById('relayout-target');\n"
            "var _rlStatus = document.getElementById('relayout-status');\n"
            "if (_rlBtn) {\n"
            "  _rlBtn.addEventListener('click', function() {\n"
            "    if (!window._targetGraph || !window._targetSigma) return;\n"
            "    if (!window.graphologyLibrary || !window.graphologyLibrary.layoutForceAtlas2) {\n"
            "      _rlStatus.textContent = 'graphology-library not loaded';\n"
            "      return;\n"
            "    }\n"
            "    _rlBtn.disabled = true; _rlStatus.textContent = 'laying out…';\n"
            "    setTimeout(function() {\n"
            "      var t0 = performance.now();\n"
            "      var n = window._targetGraph.order;\n"
            "      var bh = n >= 1000;\n"
            "      window.graphologyLibrary.layoutForceAtlas2.assign(\n"
            "        window._targetGraph,\n"
            "        {iterations: n >= 5000 ? 50 : 100,\n"
            "         settings: {barnesHutOptimize: bh, scalingRatio: 10,\n"
            "                    gravity: 1, strongGravityMode: false,\n"
            "                    slowDown: 1, adjustSizes: false}}\n"
            "      );\n"
            "      window._targetSigma.refresh();\n"
            "      var dt = (performance.now() - t0) / 1000;\n"
            "      _rlStatus.textContent = 'done in ' + dt.toFixed(2) + 's';\n"
            "      _rlBtn.disabled = false;\n"
            "    }, 10);\n"
            "  });\n"
            "}\n"
        )

    # Build SSE script — connects to /stream with the same query params
    sse_script = """
      var qs = window.location.search;
      _es = new EventSource('/stream' + qs);
      var tbody = document.getElementById('timeline-table');
      var evCount = 0;
      var thresholdMs = __THRESHOLD__;
      var MAX_ROWS = 2000;
      var pendingBatch = [];
      var rafScheduled = false;

      // Step-graph viewer state
      var allEvents = [];                  // every event the timeline saw
      var graphEventIdx = [];              // indices (into allEvents) of events with graph_key
      var stepSnapshots = {};              // canonical_key -> {nodes, edges, loops}
      var stepCursor = -1;                 // position within graphEventIdx
      var stepSigma = null;                // current Sigma instance
      var playTimer = null;

      function _buildStepGraph(snap) {
        var g = new graphology.Graph({multi: true, allowSelfLoops: true});
        var size = _sigmaNodeSize(snap.nodes.length);
        snap.nodes.forEach(function(id) {
          var loopCount = 0;
          if (snap.loops) {
            for (var i = 0; i < snap.loops.length; i++) {
              if (snap.loops[i][0] === id) { loopCount = snap.loops[i][1] || 1; break; }
            }
          }
          g.addNode(id, {
            label: loopCount > 0 ? (String(id) + ' (loop×' + loopCount + ')') : String(id),
            size: size,
            color: loopCount > 0 ? '#e65100' : '#4f8ef7',
          });
        });
        var eid = 0;
        snap.edges.forEach(function(e) {
          var mult = e[2] || 1;
          var thickness = 1 + 0.5 * (mult - 1);
          g.addEdgeWithKey('e' + (eid++), e[0], e[1],
            {size: thickness, color: mult > 1 ? '#c62828' : '#b0b0b0'});
        });
        if (snap.loops) {
          snap.loops.forEach(function(l) {
            g.addEdgeWithKey('l' + (eid++), l[0], l[0],
              {size: 1, color: '#e65100'});
          });
        }
        return g;
      }

      function _assignStepPositions(g, key) {
        // If this snapshot matches the input graph, use the server-computed
        // layout so the visual stays coherent with the target panel.
        if (key && window._inputGraphKey && key === window._inputGraphKey &&
            window._inputGraphPositions) {
          var positioned = 0;
          g.forEachNode(function(node) {
            var p = window._inputGraphPositions[String(node)];
            if (p) {
              g.setNodeAttribute(node, 'x', p[0]);
              g.setNodeAttribute(node, 'y', p[1]);
              positioned += 1;
            }
          });
          if (positioned === g.order) return;
        }
        // Otherwise: random seed + short FA2 pass for readability.
        if (window.graphologyLibrary && window.graphologyLibrary.layout &&
            window.graphologyLibrary.layout.random) {
          window.graphologyLibrary.layout.random.assign(g);
        } else {
          g.forEachNode(function(node) {
            g.setNodeAttribute(node, 'x', Math.random());
            g.setNodeAttribute(node, 'y', Math.random());
          });
        }
        if (window.graphologyLibrary && window.graphologyLibrary.layoutForceAtlas2) {
          var n = g.order;
          // Cap FA2 work on large intermediate snapshots — a few iterations
          // produce a readable layout; deeper settling costs seconds.
          var iters = n >= 2000 ? 20 : (n >= 500 ? 40 : 80);
          window.graphologyLibrary.layoutForceAtlas2.assign(g, {
            iterations: iters,
            settings: {barnesHutOptimize: n >= 1000, scalingRatio: 10, gravity: 1},
          });
        }
      }

      function _snapSummary(snap) {
        // Returns (node_count, edge_count_including_mults, loop_count).
        var nc = (snap.nodes || []).length;
        var ec = 0;
        (snap.edges || []).forEach(function(e) {
          ec += e.length > 2 ? e[2] : 1;
        });
        var lc = 0;
        (snap.loops || []).forEach(function(l) {
          lc += l.length > 1 ? l[1] : 1;
        });
        return {n: nc, e: ec, l: lc};
      }

      function renderStep(idx) {
        if (graphEventIdx.length === 0) {
          document.getElementById('step-label').textContent = '(no graph events yet)';
          return;
        }
        if (idx < 0) idx = 0;
        if (idx >= graphEventIdx.length) idx = graphEventIdx.length - 1;
        stepCursor = idx;
        var evIdx = graphEventIdx[idx];
        var ev = allEvents[evIdx];
        var snap = stepSnapshots[ev.graph_key];
        var label =
          'event #' + ev.index + ' · ' + ev.event_type + ' · ' + ev.module +
          ' · ' + (idx + 1) + '/' + graphEventIdx.length;
        if (snap) {
          var sum = _snapSummary(snap);
          var keyPrefix = ev.graph_key ? ev.graph_key.substring(0, 10) : '';
          label += ' · ' + sum.n + 'n ' + sum.e + 'e';
          if (sum.l > 0) label += ' ' + sum.l + 'loops';
          if (keyPrefix) label += ' · hash ' + keyPrefix;
        }
        document.getElementById('step-label').textContent = label;
        var container = document.getElementById('step-graph');
        if (!snap) {
          if (stepSigma) { try { stepSigma.kill(); } catch (e) {} stepSigma = null; }
          container.innerHTML =
            '<div class="meta" style="padding:12px">snapshot not yet received</div>';
          return;
        }
        var g = _buildStepGraph(snap);
        _assignStepPositions(g, ev.graph_key);
        if (stepSigma) { try { stepSigma.kill(); } catch (e) {} stepSigma = null; }
        container.innerHTML = '';
        stepSigma = new Sigma(g, container, _sigmaSettings(g.order));
        window._stepSigmaInstance = stepSigma;
        // Highlights are driven exclusively by toggled cards in the
        // Contributing Graphs panel — the Step Graph no longer
        // auto-highlights on navigation. Re-apply whatever is
        // currently toggled so the input panel stays in sync.
        _composeAndApplyHighlight();
      }

      function stepPrev() { renderStep(stepCursor - 1); }
      function stepNext() { renderStep(stepCursor + 1); }
      function stepFirst() { renderStep(0); }
      function stepLast() { renderStep(graphEventIdx.length - 1); }
      function togglePlay() {
        var btn = document.getElementById('step-play');
        if (playTimer) {
          clearInterval(playTimer);
          playTimer = null;
          btn.textContent = '\u25B6\uFE0E';  // ▶︎
          btn.title = 'Auto-advance through steps.';
          return;
        }
        var intervalMs = parseInt(document.getElementById('step-interval').value) || 500;
        btn.textContent = '\u23F8\uFE0E';  // ⏸︎
        btn.title = 'Pause auto-advance.';
        playTimer = setInterval(function() {
          if (stepCursor + 1 >= graphEventIdx.length) {
            clearInterval(playTimer);
            playTimer = null;
            btn.textContent = '\u25B6\uFE0E';  // ▶︎
            btn.title = 'Auto-advance through steps.';
            return;
          }
          renderStep(stepCursor + 1);
        }, intervalMs);
      }

      function selectTimelineRow(evIndex) {
        var ev = allEvents[evIndex];
        if (!ev) return;
        var targetEvIdx = evIndex;
        if (!ev.graph_key) {
          // Snap to nearest previous event with a graph.
          for (var i = evIndex - 1; i >= 0; i--) {
            if (allEvents[i] && allEvents[i].graph_key) { targetEvIdx = i; break; }
          }
        }
        var pos = graphEventIdx.indexOf(targetEvIdx);
        if (pos >= 0) renderStep(pos);
      }

      document.getElementById('step-first').addEventListener('click', stepFirst);
      document.getElementById('step-prev').addEventListener('click', stepPrev);
      document.getElementById('step-next').addEventListener('click', stepNext);
      document.getElementById('step-last').addEventListener('click', stepLast);
      document.getElementById('step-play').addEventListener('click', togglePlay);
      document.getElementById('step-relayout').addEventListener('click', function() {
        // Force FA2 re-run on the currently-displayed step graph.
        if (!window._stepSigmaInstance) return;
        var g = window._stepSigmaInstance.getGraph();
        if (!g || !window.graphologyLibrary ||
            !window.graphologyLibrary.layoutForceAtlas2) return;
        // Reseed random positions so FA2 has non-degenerate starting
        // state, then run a fresh pass.
        g.forEachNode(function(n) {
          g.setNodeAttribute(n, 'x', Math.random());
          g.setNodeAttribute(n, 'y', Math.random());
        });
        var n = g.order;
        var iters = n >= 2000 ? 40 : (n >= 500 ? 80 : 150);
        try {
          window.graphologyLibrary.layoutForceAtlas2.assign(g, {
            iterations: iters,
            settings: {barnesHutOptimize: n >= 1000, scalingRatio: 10, gravity: 1},
          });
          window._stepSigmaInstance.refresh();
        } catch (e) { console.error('step re-layout failed', e); }
      });
      document.getElementById('step-refresh').addEventListener('click', function() {
        // Destroy + recreate the step renderer. Useful after the
        // browser ran out of WebGL contexts (after rendering many
        // other Sigma instances in the Contributing Graphs panel)
        // and the step graph appears blank or frozen.
        if (stepCursor >= 0 && stepCursor < graphEventIdx.length) {
          renderStep(stepCursor);
        }
      });
      document.getElementById('refresh-target').addEventListener('click', function() {
        // Destroy + recreate the input-graph renderer. Same use case
        // as step-refresh: recover from WebGL exhaustion. Also drops
        // the cached "original colors" since the new instance starts
        // fresh — without this, the next highlight would diff against
        // the old (killed) renderer's colors.
        window._targetOriginalColors = null;
        if (typeof window._renderInputGraph === 'function') {
          try {
            window._renderInputGraph();
            // Re-apply any active highlights to the fresh renderer.
            _composeAndApplyHighlight();
          } catch (e) {
            console.error('refresh-target failed', e);
          }
        }
      });
      var _clearHL = document.getElementById('clear-highlights');
      if (_clearHL) {
        _clearHL.addEventListener('click', _clearAllProvenanceHighlights);
      }

      function flushBatch() {
        rafScheduled = false;
        if (pendingBatch.length === 0) return;

        var frag = document.createDocumentFragment();
        var batch = pendingBatch;
        pendingBatch = [];

        for (var b = 0; b < batch.length; b++) {
          var ev = batch[b];
          allEvents[ev.index] = ev;
          if (ev.graph_key) {
            graphEventIdx.push(ev.index);
          }
          var row = document.createElement('tr');
          row.id = 'ev-' + ev.index;
          row.style.cursor = 'pointer';
          row.setAttribute('data-ev-index', ev.index);

          var durText = '';
          var highlight = false;
          var arrow = '';
          if (ev.gap > 0 && ev.index > 0) {
            durText = ev.gap.toFixed(3) + 's';
            if (ev.gap * 1000 >= thresholdMs) {
              highlight = true;
              arrow = ev.gap >= 1
                ? ' <span style="color:#e65100">&larr; ' + ev.gap.toFixed(1) + 's</span>'
                : ' <span style="color:#e65100">&larr; ' + (ev.gap*1000).toFixed(0) + 'ms</span>';
            }
          }

          if (highlight) row.style.background = '#fff3e0';
          var indent = '';
          for (var d = 0; d < ev.depth; d++) indent += '&nbsp;&nbsp;';
          var graphDot = ev.graph_key
            ? '<span title="' + ev.graph_key.substring(0, 16) + '" style="color:#1565c0">&#x25CF;</span>'
            : '';
          row.innerHTML =
            '<td>' + ev.timestamp + '</td>' +
            '<td>' + durText + '</td>' +
            '<td>' + ev.depth + '</td>' +
            '<td><span class="badge" style="background:' + ev.color + '">' + ev.event_type + '</span></td>' +
            '<td>' + ev.module + '</td>' +
            '<td>' + indent + ev.message + arrow + '</td>' +
            '<td style="text-align:center">' + graphDot + '</td>';
          frag.appendChild(row);
        }

        // Trim old rows if over limit
        while (tbody.rows.length + frag.childNodes.length - 1 > MAX_ROWS && tbody.rows.length > 1) {
          tbody.deleteRow(1);  // keep header
        }

        tbody.appendChild(frag);
        evCount += batch.length;
        document.getElementById('event-count').textContent = '(' + evCount + ' events)';
        document.getElementById('step-graph-count').textContent =
          graphEventIdx.length + ' graph step' + (graphEventIdx.length === 1 ? '' : 's');
        if (stepCursor < 0 && graphEventIdx.length > 0) {
          renderStep(0);
        }

        var scroll = document.getElementById('timeline-scroll');
        scroll.scrollTop = scroll.scrollHeight;
      }

      // Delegate clicks on timeline rows to the step viewer.
      document.getElementById('timeline-table').addEventListener('click', function(e) {
        var row = e.target.closest('tr[data-ev-index]');
        if (!row) return;
        selectTimelineRow(parseInt(row.getAttribute('data-ev-index')));
      });

      // Highlight palette for target-graph provenance overlay.
      // Multiple instances cycle through these colors.
      var _PROV_PALETTE = [
        '#ff5722', '#43a047', '#1e88e5', '#fdd835',
        '#8e24aa', '#00897b', '#d81b60', '#3949ab'
      ];

      // Set of canonical keys whose provenance is currently active
      // (toggled on by clicking the card or badge). Multiple cards
      // can be active simultaneously — each contributes its instances
      // to the highlight, with palette colors cycling across the
      // composed list.
      window._activeProvenanceKeys = window._activeProvenanceKeys || new Set();
      // Maps canonical_key → display label (card name) for active
      // sub-graphs. Populated when cards render so the input-graph
      // label area can show "name [hash]" for each active toggle.
      window._provenanceLabels = window._provenanceLabels || {};

      function _shortKey(k) {
        return (k && k.length > 8) ? k.substring(0, 8) : (k || '');
      }

      function _labelForKey(k) {
        var name = window._provenanceLabels[k];
        if (name) return name;
        // Fall back to canonical hash digest (8 chars) when we don't
        // have a card name registered.
        return _shortKey(k);
      }

      function _renderActiveSubgraphLabels() {
        var box = document.getElementById('active-subgraph-labels');
        if (!box) return;
        var keys = Array.from(window._activeProvenanceKeys);
        if (keys.length === 0) {
          box.style.display = 'none';
          box.innerHTML = '';
          return;
        }
        box.style.display = '';
        // Walk active keys in insertion order; assign palette colors
        // cumulatively to match the composed provenance order in
        // _composeAndApplyHighlight.
        var html = '<span style="font-size:10px;color:#666;margin-right:6px;">Active subgraphs:</span>';
        var paletteIdx = 0;
        keys.forEach(function(k) {
          var snap = stepSnapshots[k];
          var instCount = (snap && snap.provenance) ? snap.provenance.length : 1;
          var color = _PROV_PALETTE[paletteIdx % _PROV_PALETTE.length];
          paletteIdx += instCount;  // step palette by # instances of this card
          var label = _labelForKey(k);
          html +=
            '<span class="active-subgraph-chip" data-prov-chip-key="' + k +
            '" style="background:' + color +
            '" title="Click to remove this sub-graph from the highlight set">' +
              label +
              '<span class="x">\u2715</span>' +
            '</span>';
        });
        box.innerHTML = html;
        // Wire chip clicks to toggle off.
        box.querySelectorAll('[data-prov-chip-key]').forEach(function(el) {
          el.addEventListener('click', function() {
            _toggleProvenanceKey(el.getAttribute('data-prov-chip-key'));
          });
        });
      }

      function _composeAndApplyHighlight() {
        // Walk active keys in the order they were toggled (Set iter
        // order is insertion order in modern browsers) and concat
        // each snapshot's provenance instances.
        var combined = [];
        window._activeProvenanceKeys.forEach(function(k) {
          var snap = stepSnapshots[k];
          if (snap && snap.provenance) {
            snap.provenance.forEach(function(p) { combined.push(p); });
          }
        });
        if (combined.length === 0) {
          _clearTargetHighlight(true);
        } else {
          _highlightTargetGraph(combined);
        }
      }

      function _updateClearHighlightsBtn() {
        var btn = document.getElementById('clear-highlights');
        if (!btn) return;
        btn.style.display = window._activeProvenanceKeys.size > 0 ? '' : 'none';
      }

      function _clearAllProvenanceHighlights() {
        if (window._activeProvenanceKeys.size === 0) return;
        var keys = Array.from(window._activeProvenanceKeys);
        window._activeProvenanceKeys.clear();
        keys.forEach(function(key) {
          document.querySelectorAll('[data-prov-key="' + key + '"]')
            .forEach(function(el) { el.classList.remove('prov-active'); });
          document.querySelectorAll('[data-prov-badge-key="' + key + '"]')
            .forEach(function(el) { el.classList.remove('active'); });
        });
        _composeAndApplyHighlight();
        _updateClearHighlightsBtn();
        _renderActiveSubgraphLabels();
      }

      function _toggleProvenanceKey(key) {
        if (!key) return;
        var snap = stepSnapshots[key];
        if (!snap || !snap.provenance) return;
        if (window._activeProvenanceKeys.has(key)) {
          window._activeProvenanceKeys.delete(key);
        } else {
          window._activeProvenanceKeys.add(key);
        }
        // Update visual state on every card with this key.
        document.querySelectorAll('[data-prov-key="' + key + '"]')
          .forEach(function(el) {
            if (window._activeProvenanceKeys.has(key)) {
              el.classList.add('prov-active');
            } else {
              el.classList.remove('prov-active');
            }
          });
        document.querySelectorAll('[data-prov-badge-key="' + key + '"]')
          .forEach(function(el) {
            if (window._activeProvenanceKeys.has(key)) {
              el.classList.add('active');
            } else {
              el.classList.remove('active');
            }
          });
        _composeAndApplyHighlight();
        _updateClearHighlightsBtn();
        _renderActiveSubgraphLabels();
      }

      function _highlightTargetGraph(provenanceList) {
        if (!window._targetGraph || !window._targetSigma) return;
        // First clear any prior highlight (restores original colors).
        _clearTargetHighlight(false);
        if (!provenanceList || !provenanceList.length) {
          try { window._targetSigma.refresh(); } catch (e) {}
          return;
        }
        var g = window._targetGraph;
        // Save original colors once so we can revert on clear.
        if (!window._targetOriginalColors) {
          window._targetOriginalColors = {nodes: {}, edges: {}};
          g.forEachNode(function(n, attrs) {
            window._targetOriginalColors.nodes[String(n)] = attrs.color;
          });
          g.forEachEdge(function(e, attrs) {
            window._targetOriginalColors.edges[e] = attrs.color;
          });
        }
        var nodeHL = {};
        var edgeHL = {};
        provenanceList.forEach(function(prov, i) {
          var color = _PROV_PALETTE[i % _PROV_PALETTE.length];
          (prov.target_nodes || []).forEach(function(n) {
            nodeHL[String(n)] = color;
          });
          (prov.target_edges || []).forEach(function(e) {
            edgeHL[String(e[0]) + '|' + String(e[1])] = color;
            edgeHL[String(e[1]) + '|' + String(e[0])] = color;
          });
        });
        g.forEachNode(function(n) {
          var c = nodeHL[String(n)];
          if (c) g.setNodeAttribute(n, 'color', c);
        });
        g.forEachEdge(function(e, attrs, source, target) {
          var c = edgeHL[String(source) + '|' + String(target)];
          if (c) {
            g.setEdgeAttribute(e, 'color', c);
            g.setEdgeAttribute(e, 'size', 2.5);
          }
        });
        try { window._targetSigma.refresh(); } catch (e) {}
      }

      function _clearTargetHighlight(refresh) {
        if (!window._targetGraph) return;
        var g = window._targetGraph;
        var orig = window._targetOriginalColors;
        if (orig) {
          g.forEachNode(function(n) {
            var c = orig.nodes[String(n)];
            if (c !== undefined) g.setNodeAttribute(n, 'color', c);
          });
          g.forEachEdge(function(e) {
            var c = orig.edges[e];
            if (c !== undefined) g.setEdgeAttribute(e, 'color', c);
            g.setEdgeAttribute(e, 'size', 1.0);
          });
        }
        if (refresh !== false && window._targetSigma) {
          try { window._targetSigma.refresh(); } catch (e) {}
        }
      }

      function _runFA2OnGraph(g) {
        if (!g || !window.graphologyLibrary ||
            !window.graphologyLibrary.layoutForceAtlas2) return;
        g.forEachNode(function(n) {
          g.setNodeAttribute(n, 'x', Math.random());
          g.setNodeAttribute(n, 'y', Math.random());
        });
        var n = g.order;
        var iters = n >= 2000 ? 40 : (n >= 500 ? 80 : 150);
        try {
          window.graphologyLibrary.layoutForceAtlas2.assign(g, {
            iterations: iters,
            settings: {barnesHutOptimize: n >= 1000, scalingRatio: 10, gravity: 1},
          });
        } catch (e) { console.error('FA2 failed', e); }
      }

      // Render a single minor card's Sigma graph and wire its
      // Re-layout button. Factored out so overflow cards can render
      // lazily on expand.
      // Returns the Sigma instance so the caller can kill() it on
      // collapse.
      function _renderMinorCard(m, divId, btnId) {
        if (!m.sigma_json) return null;
        try {
          var g = new graphology.Graph();
          g.import(JSON.parse(m.sigma_json));
          var container = document.getElementById(divId);
          if (!container) return null;
          container.innerHTML = '';
          var s = new Sigma(g, container, _sigmaSettings(g.order));
          window._minorSigmas.push(s);
          var btn = document.getElementById(btnId);
          if (btn) {
            btn.addEventListener('click', function(e) {
              e.stopPropagation();
              _runFA2OnGraph(g);
              try { s.refresh(); } catch (ex) {}
            });
          }
          return s;
        } catch (e) {
          console.error('minor-card render failed', e);
          return null;
        }
      }

      // Render the Contributing-Graphs section. First `initialVisible`
      // cards render eagerly with Sigma; the rest appear as collapsed
      // title rows and expand on click (lazy render). This caps FA2
      // load since browsers struggle to render more than ~5 Sigma
      // instances at once. Each card has an Expand/Collapse toggle
      // so users can swap which graphs are rendered without reload.
      function renderMinorsSection(list, containerId, emptyText, initialVisible) {
        if (initialVisible === undefined) initialVisible = 3;
        var container = document.getElementById(containerId);
        if (!list || list.length === 0) {
          container.innerHTML = '<div class="meta">' + emptyText + '</div>';
          return;
        }
        container.innerHTML = '';
        list.forEach(function(m, i) {
          var card = document.createElement('div');
          card.className = 'minor-card';
          var divId = containerId + '-card-' + i;
          var btnId = divId + '-relayout';
          var refreshBtnId = divId + '-refresh';
          var toggleBtnId = divId + '-toggle';
          var startExpanded = i < initialVisible;

          // Closure-local state tracked per card.
          var state = { expanded: false, sigma: null };

          function _updateButtons() {
            var slot = document.getElementById(divId);
            var relayoutBtn = document.getElementById(btnId);
            var refreshBtn = document.getElementById(refreshBtnId);
            var toggleBtn = document.getElementById(toggleBtnId);
            if (slot) slot.style.display = state.expanded ? '' : 'none';
            if (relayoutBtn) relayoutBtn.style.display = state.expanded ? '' : 'none';
            if (refreshBtn) refreshBtn.style.display = state.expanded ? '' : 'none';
            if (toggleBtn) {
              // ▲ collapse, ▼ expand.
              toggleBtn.textContent = state.expanded ? '\u25B2' : '\u25BC';
              toggleBtn.title = state.expanded
                ? 'Hide this graph to free render resources.'
                : 'Expand to render this graph.';
            }
          }

          function _expand() {
            if (state.expanded) return;
            state.expanded = true;
            // Make sure the slot is fresh — if a previous Sigma left
            // any DOM behind the new one would render on top of stale
            // canvases.
            var slot = document.getElementById(divId);
            if (slot) slot.innerHTML = '';
            state.sigma = _renderMinorCard(m, divId, btnId);
            _updateButtons();
            // Wire the per-card power-refresh button to a full
            // kill+rebuild of this card's Sigma instance — recovers
            // from WebGL glitches without re-running the engine.
            var refreshBtn = document.getElementById(refreshBtnId);
            if (refreshBtn) {
              refreshBtn.onclick = function() {
                if (state.sigma) {
                  try { state.sigma.kill(); } catch (e) {}
                  var idx = (window._minorSigmas || []).indexOf(state.sigma);
                  if (idx >= 0) window._minorSigmas.splice(idx, 1);
                  state.sigma = null;
                }
                var s2 = document.getElementById(divId);
                if (s2) {
                  while (s2.firstChild) s2.removeChild(s2.firstChild);
                }
                state.sigma = _renderMinorCard(m, divId, btnId);
              };
            }
          }

          function _collapse() {
            if (!state.expanded) return;
            state.expanded = false;
            if (state.sigma) {
              try { state.sigma.kill(); } catch (e) {}
              // Remove from the global list so re-layout-all doesn't
              // touch a killed instance.
              var idx = (window._minorSigmas || []).indexOf(state.sigma);
              if (idx >= 0) window._minorSigmas.splice(idx, 1);
              state.sigma = null;
            }
            // Hard reset the slot: drop the WebGL canvas + label DOM
            // so a future expand renders cleanly.
            var slot = document.getElementById(divId);
            if (slot) {
              while (slot.firstChild) slot.removeChild(slot.firstChild);
            }
            _updateButtons();
          }

          var provN = m.provenance_count || 0;
          var hasProv = provN > 0;
          if (hasProv) {
            card.classList.add('has-provenance');
            if (m.key) {
              card.setAttribute('data-prov-key', m.key);
              // Register the display label so the input-graph chip
              // area can show this card's name even when the card
              // itself is collapsed/scrolled out of view.
              window._provenanceLabels[m.key] = m.name || _shortKey(m.key);
            }
            // Reflect current toggle state if this key was already
            // active from a prior card render of the same key.
            if (m.key && window._activeProvenanceKeys
                && window._activeProvenanceKeys.has(m.key)) {
              card.classList.add('prov-active');
            }
          }
          var provBadge = '';
          if (hasProv) {
            var badgeActive = (m.key && window._activeProvenanceKeys
                && window._activeProvenanceKeys.has(m.key)) ? ' active' : '';
            provBadge =
              '<span class="prov-badge' + badgeActive +
              '" data-prov-badge-key="' + (m.key || '') +
              '" title="Click to toggle highlight of ' + provN +
              ' instance' + (provN === 1 ? '' : 's') +
              ' on the input graph">\u25C9 ' + provN + '</span>';
          }
          card.innerHTML =
            '<div class="minor-label" style="display:flex;justify-content:space-between;align-items:center;gap:6px;">' +
              '<span>' + m.name + ' (' + m.edges + ' edges)' + provBadge + '</span>' +
              '<span>' +
                '<button type="button" id="' + toggleBtnId + '" style="font-size:10px;padding:2px 6px;" title="Expand to render this graph.">\u25BC</button> ' +
                '<button type="button" id="' + btnId + '" style="font-size:10px;padding:2px 6px;display:none;" title="Re-run ForceAtlas2 on this graph card.">\u21BA</button> ' +
                '<button type="button" id="' + refreshBtnId + '" style="font-size:10px;padding:2px 6px;display:none;" title="Destroy and recreate this renderer (use after WebGL glitches).">\u23FB</button>' +
              '</span>' +
            '</div>' +
            '<div id="' + divId + '" class="small-graph" style="display:none"></div>';
          container.appendChild(card);

          // Click anywhere on the card (or specifically the badge) to
          // toggle the input-graph highlight for this sub-graph. The
          // expand/collapse and re-layout buttons stop propagation so
          // they don't double-toggle.
          (function(key) {
            if (!key || !hasProv) return;
            card.addEventListener('click', function(e) {
              // Don't toggle when the click bubbled from a button.
              if (e.target.closest('button')) return;
              _toggleProvenanceKey(key);
            });
          })(m.key);

          // Wire the toggle button to switch states.
          (function() {
            var toggleBtn = document.getElementById(toggleBtnId);
            if (toggleBtn) {
              toggleBtn.addEventListener('click', function() {
                if (state.expanded) _collapse(); else _expand();
              });
            }
          })();

          if (startExpanded) {
            setTimeout(_expand, 50);
          }
        });
      }

      _es.onmessage = function(msg) {
        var d = JSON.parse(msg.data);

        if (d.type === 'batch') {
          if (d.snapshots) {
            for (var k in d.snapshots) {
              if (d.snapshots.hasOwnProperty(k)) stepSnapshots[k] = d.snapshots[k];
            }
          }
          for (var i = 0; i < d.events.length; i++) {
            pendingBatch.push(d.events[i]);
          }
          if (!rafScheduled) {
            rafScheduled = true;
            requestAnimationFrame(flushBatch);
          }
        }

        else if (d.type === 'done') {
          stopEngine();
          document.getElementById('event-count').textContent = '(' + d.event_count + ' events)';
          document.getElementById('result-container').innerHTML = d.result_html;

          // Merge final snapshots-with-provenance back into the local
          // stepSnapshots cache. The provenance is added AFTER the
          // initial snapshot was streamed, so this catch-up is needed.
          if (d.snapshots_with_provenance) {
            for (var pk in d.snapshots_with_provenance) {
              if (d.snapshots_with_provenance.hasOwnProperty(pk)) {
                if (stepSnapshots[pk]) {
                  stepSnapshots[pk].provenance =
                    d.snapshots_with_provenance[pk].provenance;
                } else {
                  stepSnapshots[pk] = d.snapshots_with_provenance[pk];
                }
              }
            }
          }

          // Summary
          if (d.summary && d.summary.length > 0) {
            var sh = '<table class="summary"><tr><th>EventType</th><th>Count</th><th>Duration</th><th colspan="2">Share</th></tr>';
            d.summary.forEach(function(s) {
              var barW = Math.max(s.pct_num * 1.5, s.pct_num > 0 ? 2 : 0);
              sh += '<tr>'
                + '<td><span class="badge" style="background:' + s.color + '">' + s.event_type + '</span></td>'
                + '<td style="text-align:right">' + s.count.toLocaleString() + '</td>'
                + '<td style="text-align:right">' + s.duration + '</td>'
                + '<td style="text-align:right;width:50px">' + s.pct + '</td>'
                + '<td style="width:160px"><span class="pct-bar" style="width:' + barW + 'px;background:' + s.color + '"></span></td>'
                + '</tr>';
            });
            sh += '</table>';
            document.getElementById('summary-container').innerHTML = sh;
          }

          // Contributing graphs — split by provenance
          var mc = document.getElementById('minors-container');
          var lookupList = d.minors_lookup || d.minors || [];
          var synthList = d.minors_synthesized || [];
          mc.innerHTML =
            '<div class="minor-subsection">' +
              '<h3 style="font-size:12px;margin:0 0 6px 0;color:#2e7d32;">' +
                'From Lookup Table (' + lookupList.length + ')' +
              '</h3>' +
              '<div id="minors-lookup" class="minors-grid"></div>' +
            '</div>' +
            '<div class="minor-subsection" style="margin-top:12px">' +
              '<h3 style="font-size:12px;margin:0 0 6px 0;color:#1565c0;">' +
                'Synthesized During Run (' + synthList.length + ')' +
              '</h3>' +
              '<div id="minors-synthesized" class="minors-grid"></div>' +
            '</div>';
          // Lookup section: show first 3 cards expanded by default
          // (cheap, most-useful reference graphs).
          // Synthesized section: all collapsed by default — these are
          // intermediate multigraphs that can explode in count (Cm2
          // has 125+), so let the user opt-in per card.
          renderMinorsSection(lookupList, 'minors-lookup',
            'No rainbow table entries used.', 3);
          renderMinorsSection(synthList, 'minors-synthesized',
            'No graphs synthesized from scratch (all hits were in the table).',
            0);
        }

        else if (d.type === 'error') {
          stopEngine();
          document.getElementById('result-container').innerHTML =
            '<div class="error-banner">' + d.message + '</div>';
          document.getElementById('minors-container').innerHTML = '';
        }
      };

      _es.onerror = function() {
        stopEngine();
      };
    """.replace("__THRESHOLD__", str(threshold_ms))

    # Build page — use __PLACEHOLDER__ approach to avoid JSON braces in .format()
    page = HTML.format(
        **ctrl_vars,
        graph_desc=graph_desc, input_meta=input_meta,
        event_colors_json=json.dumps(EVENT_COLORS),
        input_graph_script="__INPUT_GRAPH_SCRIPT__",
        sse_script="__SSE_SCRIPT__",
    )
    page = page.replace("__INPUT_GRAPH_SCRIPT__", input_graph_script)
    page = page.replace("__SSE_SCRIPT__", sse_script)
    return Response(page, mimetype="text/html")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5002
    print(f"Tutte Engine Visualizer running at http://localhost:{port}/")
    print(f"  Examples:")
    print(f"    http://localhost:{port}/?atlas=18")
    print(f"    http://localhost:{port}/?atlas=150")
    print(f"    http://localhost:{port}/?edges=0-1,1-2,2-3,3-0,0-2,1-3")
    print(f"    http://localhost:{port}/?source=dwave&dwave_topo=zephyr&dwave_m=1&dwave_t=1")
    app.run(debug=False, port=port, threaded=True)
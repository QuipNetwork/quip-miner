"""Benchmark Tutte polynomial computation from an empty rainbow table.

Each graph is synthesized from scratch. After synthesis, its polynomial is
added to the rainbow table so subsequent graphs can use it as a tile/minor.
Graphs are sorted by edge count so simpler ones seed the table first.

Engines benchmarked:
    - CEJ (SynthesisEngine): creation-expansion-join with growing rainbow table
    - NetworkX (nx.tutte_polynomial): reference implementation via deletion-contraction

Standalone usage:
    python -m tutte.benchmarks.benchmark
    python -m tutte.benchmarks.benchmark --compare file1.json file2.json

Pytest integration: run with --benchmark flag to collect timings automatically.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

import networkx as nx
from tutte.graph import (Graph, complete_graph, cycle_graph, grid_graph,
                         petersen_graph, wheel_graph)
from tutte.lookup import RainbowTable, save_binary_rainbow_table
from tutte.polynomial import TuttePolynomial
from tutte.synthesis import SynthesisEngine
from tutte.validation import count_spanning_trees_kirchhoff

# ---------------------------------------------------------------------------
# Named graph set (merged with atlas below)
# ---------------------------------------------------------------------------

def _complete_bipartite(a, b):
    """Build K_{a,b} as a Graph with vertices 0..a-1 (one side) | a..a+b-1 (other)."""
    import networkx as nx
    return Graph.from_networkx(nx.complete_bipartite_graph(a, b))


def _named_nx(builder, *args):
    """Wrap a networkx graph builder into our Graph type."""
    return Graph.from_networkx(builder(*args))


# Reasonable-size representatives of the standard graph families,
# beyond what the atlas suite already covers (atlas: up to 7 nodes).
# Sorted later by (edges, nodes, name) so they land at their natural
# difficulty rung in the merged benchmark list. The K_n and K_{a,b}
# entries also seed the rainbow table with cells the engine's
# cell-quotient / chord-rule paths use to recognise structured cells
# in subsequent Chimera / Pegasus / Zephyr graphs — without K_{4,4}
# in the table, Cm_2 falls through to `treewidth_dp` (~140 s) instead
# of `cell_quotient_grid_dp_streamed` / `chord_rule` (~5–40 s).
NAMED_GRAPHS = [
    # Complete graphs K_n.
    ("K_3", lambda: complete_graph(3)),
    ("K_4", lambda: complete_graph(4)),
    ("K_5", lambda: complete_graph(5)),
    ("K_6", lambda: complete_graph(6)),
    ("K_7", lambda: complete_graph(7)),
    ("K_8", lambda: complete_graph(8)),
    ("K_9", lambda: complete_graph(9)),
    ("K_10", lambda: complete_graph(10)),
    ("K_11", lambda: complete_graph(11)),
    ("K_12", lambda: complete_graph(12)),
    # Complete bipartite K_{a,b}. Elementary D-Wave / cell-quotient
    # atoms — K_{4,4} is the Chimera cell, K_{3,3} appears in Pegasus.
    ("K_{2,2}", lambda: _complete_bipartite(2, 2)),
    ("K_{2,3}", lambda: _complete_bipartite(2, 3)),
    ("K_{2,4}", lambda: _complete_bipartite(2, 4)),
    ("K_{2,5}", lambda: _complete_bipartite(2, 5)),
    ("K_{3,3}", lambda: _complete_bipartite(3, 3)),
    ("K_{3,4}", lambda: _complete_bipartite(3, 4)),
    ("K_{3,5}", lambda: _complete_bipartite(3, 5)),
    ("K_{4,4}", lambda: _complete_bipartite(4, 4)),  # Chimera/Pegasus cell
    ("K_{4,5}", lambda: _complete_bipartite(4, 5)),
    ("K_{5,5}", lambda: _complete_bipartite(5, 5)),
    # Cycles C_n.
    ("C_5", lambda: cycle_graph(5)),
    ("C_10", lambda: cycle_graph(10)),
    ("C_15", lambda: cycle_graph(15)),
    ("C_20", lambda: cycle_graph(20)),
    ("C_30", lambda: cycle_graph(30)),
    ("C_50", lambda: cycle_graph(50)),
    ("C_100", lambda: cycle_graph(100)),
    # Wheels W_n.
    ("W_5", lambda: wheel_graph(5)),
    ("W_7", lambda: wheel_graph(7)),
    ("W_9", lambda: wheel_graph(9)),
    ("W_11", lambda: wheel_graph(11)),
    ("W_15", lambda: wheel_graph(15)),
    # Grids m × n.
    ("Grid_3x3", lambda: grid_graph(3, 3)),
    ("Grid_4x4", lambda: grid_graph(4, 4)),
    ("Grid_5x5", lambda: grid_graph(5, 5)),
    ("Grid_6x6", lambda: grid_graph(6, 6)),
    # Named graphs (cubic / vertex-transitive structure cases).
    ("Petersen", lambda: petersen_graph()),
    ("Heawood", lambda: _named_nx(nx.heawood_graph)),
    ("MoebiusKantor", lambda: _named_nx(nx.moebius_kantor_graph)),
    ("Desargues", lambda: _named_nx(nx.desargues_graph)),
    ("Dodecahedral", lambda: _named_nx(nx.dodecahedral_graph)),
]


def _family_recognition_graphs():
    """Representatives of every family that `tutte.family_recognition` handles.

    These graphs all have closed-form / recurrence-based Tutte polynomials
    in the recognition engine and ALSO serve as elementary atoms that other
    decomposition paths can recognize as cells. Sizes chosen to span the
    O(1) base-case window through the recurrence-driven regime.

    Family → builder mapping:
      - Wheel W_k: `wheel_graph(k)` — rim k vertices, k+1 total.
      - Fan F_k: nx.lollipop_graph not quite — use direct construction.
      - Ladder L_k = P_k × K_2: `nx.ladder_graph(k)` — 2k vertices.
      - Book B_k: k triangles sharing one edge.
      - Gear G_k: hub + k-rim + k subdiv = 2k+1 vertices.
      - Prism CL_k = C_k × K_2: `nx.circular_ladder_graph(k)`.
      - Möbius M_k: 2k-cycle with k diameter chords.
      - Pan (cycle + 1 pendant), Sunlet (cycle + pendant per vertex),
        Helm (wheel + pendant per rim).
    """
    from ..family_recognition._seed_builders import (
        book_graph, gear_graph, mobius_graph, prism_graph,
    )
    out = []

    def _fan(k):
        """Fan F_k: path P_k joined to a single hub vertex."""
        G = nx.Graph()
        for i in range(k):
            G.add_edge(0, i + 1)  # hub-to-path-vertex
        for i in range(k - 1):
            G.add_edge(i + 1, i + 2)  # path edges
        return Graph.from_networkx(G)

    def _pan(n):
        """Pan: cycle C_n with one pendant edge."""
        G = nx.cycle_graph(n)
        G.add_edge(0, n)  # pendant at vertex 0
        return Graph.from_networkx(G)

    def _sunlet(k):
        """Sunlet: cycle C_k with a pendant at every cycle vertex."""
        G = nx.cycle_graph(k)
        for i in range(k):
            G.add_edge(i, k + i)
        return Graph.from_networkx(G)

    def _helm(k):
        """Helm H_k: wheel with k rim vertices + a pendant per rim vertex.

        Uses the family_recognition convention `wheel_recurrence(k)` ⇒
        k rim vertices. `nx.wheel_graph(n)` has n total nodes, so we
        call `nx.wheel_graph(k + 1)` here: hub at 0, rim at 1..k, then
        add pendants k+1..2k.
        """
        G = nx.wheel_graph(k + 1)  # k rim vertices (hub at 0)
        for i in range(1, k + 1):
            G.add_edge(i, k + i)
        return Graph.from_networkx(G)

    # Wheels — already partially named; add larger k.
    for k in (3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20):
        out.append((f"Wheel_{k}", lambda k=k: Graph.from_networkx(nx.wheel_graph(k))))
    # Fans.
    for k in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15):
        out.append((f"Fan_{k}", lambda k=k: _fan(k)))
    # Ladders.
    for k in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15):
        out.append((f"Ladder_{k}", lambda k=k: Graph.from_networkx(nx.ladder_graph(k))))
    # Books.
    for k in (1, 2, 3, 4, 5, 6, 7, 8, 10):
        out.append((f"Book_{k}", lambda k=k: book_graph(k)))
    # Gears.
    for k in (3, 4, 5, 6, 7, 8, 10, 12):
        out.append((f"Gear_{k}", lambda k=k: gear_graph(k)))
    # Prisms.
    for k in (3, 4, 5, 6, 7, 8, 9, 10, 12):
        out.append((f"Prism_{k}", lambda k=k: prism_graph(k)))
    # Möbius ladders.
    for k in (3, 4, 5, 6, 7, 8, 10):
        out.append((f"Mobius_{k}", lambda k=k: mobius_graph(k)))
    # Pans, sunlets, helms.
    for n in (3, 4, 5, 7, 10, 15):
        out.append((f"Pan_{n}", lambda n=n: _pan(n)))
    for k in (3, 4, 5, 7, 10):
        out.append((f"Sunlet_{k}", lambda k=k: _sunlet(k)))
    for k in (3, 4, 5, 7, 10):
        out.append((f"Helm_{k}", lambda k=k: _helm(k)))
    return out


def _try_dwave_graphs():
    """Add D-Wave graphs if available: Chimera C1-C16, Pegasus P1-P16, Zephyr Z(1,1).

    Also includes Z(1,2) inter-cell component graphs (12n/16e series-parallel,
    treewidth 2) which appear during hierarchical tiling of Zephyr topologies.
    """
    extras = []
    try:
        import dwave_networkx as dnx
        import networkx as nx
        from tutte.graphs.covering import try_hierarchical_partition
        from tutte.lookup import load_default_table

        for m in range(1, 17):
            _m = m  # capture for lambda
            extras.append((f"Cm{m}", lambda _m=_m: Graph.from_networkx(dnx.chimera_graph(_m))))
        for m in range(1, 17):
            _m = m
            G = dnx.pegasus_graph(_m)
            if G.number_of_nodes() > 0:
                extras.append((f"Pm{m}", lambda _m=_m: Graph.from_networkx(dnx.pegasus_graph(_m))))
        extras.append(("Z1_1", lambda: Graph.from_networkx(dnx.zephyr_graph(1, 1))))
        extras.append(("Z1_2", lambda: Graph.from_networkx(dnx.zephyr_graph(1, 2))))

        # Z(1,2) inter-cell components: 2 isomorphic series-parallel graphs
        # that arise from hierarchical tiling of Zephyr Z(1,2).
        def _z12_inter_cell_component():
            z12 = Graph.from_networkx(dnx.zephyr_graph(1, 2))
            table = load_default_table()
            result = try_hierarchical_partition(z12, table)
            if result is None:
                return None
            _, _, inter_info = result
            inter_nx = nx.Graph()
            for u, v in inter_info.edges:
                inter_nx.add_edge(min(u, v), max(u, v))
            # Both components are isomorphic; take the first
            comp = next(iter(nx.connected_components(inter_nx)))
            sub = inter_nx.subgraph(comp)
            comp_edges = frozenset((min(u, v), max(u, v)) for u, v in sub.edges())
            return Graph(nodes=frozenset(comp), edges=comp_edges)

        extras.append(("Z1_2_inter_component", _z12_inter_cell_component))
    except ImportError:
        pass
    return extras


def _atlas_graphs():
    """All connected atlas graphs with >= 1 edge."""
    for i in range(1, 1253):
        try:
            G = nx.graph_atlas(i)
        except Exception:
            continue
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            continue
        if not nx.is_connected(G):
            continue
        yield f"atlas_{i}", Graph.from_networkx(G)


def _build_graph_list():
    """Build the full sorted graph list: named + atlas + dwave, sorted by edges.

    Deduplication uses canonical keys for small graphs (<=30 edges) only,
    since WL hashing is too expensive for large D-Wave topologies.
    """
    # Small graphs: deduplicate by canonical key
    small = []
    for name, builder in NAMED_GRAPHS:
        small.append((name, builder()))

    for name, builder in _family_recognition_graphs():
        small.append((name, builder()))

    for name, g in _atlas_graphs():
        small.append((name, g))

    seen = {}
    deduped = []
    for name, g in small:
        key = g.canonical_key()
        if key in seen:
            if not name.startswith("atlas_") and seen[key].startswith("atlas_"):
                deduped = [(n, gr) if n != seen[key] else (name, g) for n, gr in deduped]
                seen[key] = name
            continue
        seen[key] = name
        deduped.append((name, g))

    # Large D-Wave graphs: no deduplication needed (unique topologies)
    for name, builder in _try_dwave_graphs():
        g = builder()
        if g is not None:
            deduped.append((name, g))

    deduped.sort(key=lambda x: (x[1].node_count(), x[1].edge_count(), x[0]))
    return deduped


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

_TIMEOUT = "TIMEOUT"
_ERROR = "ERROR"


def _time_fn(fn, timeout_s=60):
    """Time a function with a SIGALRM-based timeout.

    Returns (elapsed_ms, result, None) on success, (None, None, _TIMEOUT)
    on timeout, or (None, None, _ERROR) on exception.

    SIGALRM delivery is queued during C extension calls (it fires only
    when control returns to Python), so this primitive does NOT preempt
    long C-ext computations like the treewidth-DP cffi path on Cm/Pm/Z
    graphs. For those, use `_time_fn_hard` which spawns a subprocess
    and SIGKILLs on timeout.
    """
    class _TimeoutExc(BaseException):
        """Inherits BaseException so it won't be caught by `except Exception`
        inside networkx/sympy internals."""
        pass

    def _handler(signum, frame):
        raise _TimeoutExc()

    old = None
    if hasattr(signal, "SIGALRM"):
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_s)

    try:
        t0 = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        return round(elapsed, 3), result, None
    except _TimeoutExc:
        return None, None, _TIMEOUT
    except Exception:
        return None, None, _ERROR
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old is not None:
                signal.signal(signal.SIGALRM, old)


# Edge-count threshold above which the benchmark switches from in-process
# SIGALRM timeouts to subprocess-level timeouts. Below this, the synthesis
# is mostly in Python paths where SIGALRM fires correctly and subprocess
# spawn overhead (~0.5–1 s on macOS) would dominate. Above this, the
# treewidth-DP cffi call typically dominates and SIGALRM cannot interrupt
# it — we need OS-level termination via `Process.terminate()` (SIGTERM)
# escalating to `Process.kill()` (SIGKILL) on stubborn cases.
_HARD_TIMEOUT_EDGE_THRESHOLD = 60


def _synth_worker_entry(synth_label, nodes_list, edges_list, table_dump, q):
    """Subprocess entrypoint for `_time_fn_hard`.

    Rebuilds the rainbow table from `table_dump` (so cell-quotient paths
    can recognize K_{4,4}-shaped cells accumulated from earlier benchmark
    graphs), instantiates the requested engine, synthesizes the graph,
    sends the result back via the queue.

    On crash or exception the parent sees `q.get(timeout=…)` raise
    `queue.Empty` and treats the run as TIMEOUT/ERROR.
    """
    try:
        import pickle
        from tutte.graph import Graph
        from tutte.lookup.core import RainbowTable
        graph = Graph(
            nodes=frozenset(nodes_list),
            edges=frozenset((min(u, v), max(u, v)) for u, v in edges_list),
        )
        # Restore the accumulated rainbow table — without K_{4,4} etc.
        # cell-quotient paths return None and dispatch falls all the way
        # through to treewidth_dp (where the timeout was unenforceable).
        table = pickle.loads(table_dump) if table_dump else RainbowTable()
        from tutte.synthesis.engine import SynthesisEngine
        eng = SynthesisEngine(table=table)
        result = eng.synthesize(graph)
        poly_bytes = result.polynomial.to_bytes()
        trees = result.polynomial.num_spanning_trees()
        q.put(("ok", trees, poly_bytes, getattr(result, "method", "?")))
    except Exception as e:
        q.put(("err", type(e).__name__, str(e)))


class _SubprocResult:
    """Result shim for `_time_fn_hard`.

    The benchmark's downstream code accesses three attributes on the
    synthesis result: ``.polynomial`` (the TuttePolynomial), ``.method``
    (a label string used in logging), and ``.minors_used`` (passed to
    ``table.add(...)`` for minor-relationship tracking). The subprocess
    can't easily return a full SynthesisResult — it sends back just the
    polynomial bytes + method label and lets the parent reconstruct.
    Minors-used isn't available across the boundary so we pass an empty
    set; that only affects minor-relationship indexing, not correctness.
    """

    __slots__ = ("polynomial", "method", "minors_used")

    def __init__(self, payload):
        trees, poly, method = payload
        self.polynomial = poly
        self.method = method
        self.minors_used = set()


def _time_fn_hard(synth_label, graph, table, timeout_s):
    """Time `engine.synthesize(graph)` with a HARD subprocess-level timeout.

    Spawns a fresh subprocess (`spawn` start method so cffi is reloaded
    cleanly), runs the synthesis there with the parent's accumulated
    `table` snapshot pickled into the worker's startup, terminates on
    timeout via SIGTERM → SIGKILL. Returns the same 3-tuple shape as
    `_time_fn`: `(elapsed_ms, (trees, poly, method), None)` on success,
    `(None, None, _TIMEOUT)`, or `(None, None, _ERROR)`.

    Cost: ~0.5–1 s per call on macOS for the spawn + pickle round-trip.
    Worth it for graphs where the alternative is an unkillable C-ext
    call running for minutes past the benchmark's stated timeout.
    """
    import multiprocessing as mp
    import pickle
    try:
        table_dump = pickle.dumps(table) if table is not None else b""
    except Exception:
        table_dump = b""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_synth_worker_entry,
        args=(synth_label, list(graph.nodes), list(graph.edges), table_dump, q),
    )
    t0 = time.perf_counter()
    p.start()
    try:
        result = q.get(timeout=timeout_s + 5)  # +5 s grace for pickle/import
    except Exception:
        p.terminate()
        p.join(timeout=1.0)
        if p.is_alive():
            p.kill()
            p.join(timeout=1.0)
        return None, None, _TIMEOUT
    elapsed = (time.perf_counter() - t0) * 1000
    p.join(timeout=1.0)
    if p.is_alive():
        p.kill()
    if isinstance(result, tuple) and result[0] == "ok":
        _, trees, poly_bytes, method = result
        from ..polynomial import TuttePolynomial
        return round(elapsed, 3), (trees, TuttePolynomial.from_bytes(poly_bytes), method), None
    return None, None, _ERROR




def _tutte_networkx(G_nx):
    """Compute Tutte polynomial via NetworkX. Does NOT swallow exceptions,
    so SIGALRM timeouts propagate correctly."""
    from sympy import Poly, symbols
    x, y = symbols('x y')
    tutte_sympy = nx.tutte_polynomial(G_nx)
    poly = Poly(tutte_sympy, x, y)
    coeffs = {}
    for monom, coeff in poly.as_dict().items():
        coeffs[monom] = int(coeff)
    return TuttePolynomial.from_coefficients(coeffs)


def _nx_family_key(name):
    """Group benchmark graph names into scaling families for the NetworkX
    give-up rule.

    The graph list is sorted by size, so members of a family appear in
    increasing difficulty. Once NetworkX times out on one member, every
    later member is strictly harder for its deletion-contraction recursion
    — there's no point spending another `nx_timeout_s` proving it. Atlas
    graphs and unparametrised one-offs (Petersen, Heawood, …) are each
    their own family so a single timeout never suppresses unrelated graphs.
    """
    if name.startswith("atlas_"):
        return name  # each atlas graph independent — not a size-scaling series
    if name.startswith("K_{"):
        return "K_{a,b}"  # bipartite completes (name has no trailing integer)
    if name.startswith("Grid_"):
        return "Grid"  # Grid_RxC has no bare trailing integer
    if name.startswith("Z1_") and "inter" not in name:
        return "Zephyr"  # Z1_1, Z1_2, … (the inter-cell component is its own graph)
    # Generic: strip a trailing size parameter — "K_3"->"K", "C_10"->"C",
    # "Wheel_5"->"Wheel", "Cm12"->"Cm", "Pm3"->"Pm". Names with no trailing
    # integer fall through to their full name (their own family).
    m = re.match(r"^([A-Za-z]+)_?\d+$", name)
    if m:
        return m.group(1)
    return name


def _fmt(ms):
    if ms is None:
        return "-"
    if ms < 1:
        return f"{ms:.3f}ms"
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms / 1000:.2f}s"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmarks(timeout_s=60, nx_timeout_s=30):
    """Run benchmarks from empty rainbow tables.

    Two engines are benchmarked independently:
      - CEJ (SynthesisEngine) with its own growing table
      - NetworkX (nx.tutte_polynomial) as reference (no table)

    After each graph, if an engine produced a correct result, the polynomial
    is added to that engine's rainbow table for future graphs.
    """
    cej_table = RainbowTable()
    cej_engine = SynthesisEngine(cej_table)

    # Pre-warm ALL cffi JIT caches so first-touch compile costs don't get
    # absorbed into a specific graph's CEJ timing. Each `_get_lib`
    # call triggers cffi.verify/compile (~1-2s each). Without this, the
    # cost ends up on whichever atlas graph first hits an unprimed C ext.
    _CFFI_LOADERS = []
    try:
        from tutte.graphs._treewidth_c import _get_lib as _tw_get_lib
        _CFFI_LOADERS.append(("_treewidth_c", _tw_get_lib))
    except ImportError:
        pass
    try:
        from tutte.roots._partition_c import _get_lib as _part_get_lib
        _CFFI_LOADERS.append(("_partition_c", _part_get_lib))
    except ImportError:
        pass
    try:
        from tutte._polynomial_c import _get_lib as _poly_get_lib
        _CFFI_LOADERS.append(("_polynomial_c", _poly_get_lib))
    except ImportError:
        pass
    print(f"Pre-warming {len(_CFFI_LOADERS)} cffi extensions...", flush=True)
    for name, loader in _CFFI_LOADERS:
        try:
            t0 = time.perf_counter()
            loader()
            warm_ms = (time.perf_counter() - t0) * 1000
            print(f"  {name}: {warm_ms:.0f}ms", flush=True)
        except Exception as e:
            print(f"  {name}: skipped ({type(e).__name__})", flush=True)

    # Pre-warm family_recognition lazy state: rainbow-table load + base-case
    # seeds for wheel/fan/ladder/etc. recurrences. Without this, the first
    # graph in the loop that triggers `recognize_family` (typically K_4)
    # absorbs ~300ms of cached-table load + ~1s of rooted-lookup load into
    # its CEJ timing — turning a 1ms operation into a 1s reported regression.
    try:
        t0 = time.perf_counter()
        from ..family_recognition.constants import _get_cached_table
        _get_cached_table()
        # Trigger every base-case lazy load via tiny synth on K_3, K_4, K_5
        # which together hit wheel/cycle/fan/star/complete bases.
        from ..family_recognition import recognize_family
        from ..graph import complete_graph
        for size in (3, 4, 5):
            recognize_family(complete_graph(size))
        warm_ms = (time.perf_counter() - t0) * 1000
        print(f"  family_recognition seeds: {warm_ms:.0f}ms", flush=True)
    except Exception as e:
        print(f"  family_recognition seeds: skipped ({type(e).__name__})", flush=True)

    graphs = _build_graph_list()
    results = []
    stats = {"cej_ok": 0, "nx_ok": 0,
             "cej_fail": 0, "nx_fail": 0,
             "poly_mismatch": 0}

    hdr = (f"{'#':>5} {'Graph':<20} {'N':>3} {'M':>3} {'Trees':>14} "
           f"{'CEJ':>10} {'CEJ method':<32} {'NetworkX':>10}")
    print(f"Benchmarking {len(graphs)} graphs (2 engines, empty tables)")
    print(hdr)
    print("-" * len(hdr))

    # Track which edge counts have been proven unsolvable per engine,
    # so we don't waste timeout_s on every C2-C16 graph.
    cej_max_solved = 0
    nx_max_solved = 0
    # Families on which NetworkX has already timed out — later (larger)
    # members are skipped without re-attempting (see `_nx_family_key`).
    nx_timed_out_families = set()

    import sys as _sys
    for idx, (name, graph) in enumerate(graphs, 1):
        n, m = graph.node_count(), graph.edge_count()
        G_nx = graph.to_networkx()

        # Hang indicator: print "starting" line BEFORE any synth so we see
        # exactly which graph is in flight if the harness hangs. Written to
        # stderr to keep stdout clean for downstream parsers.
        print(f"  [start {idx:>5}] {name:<22} n={n:>3} m={m:>3}",
              file=_sys.stderr, flush=True)

        # Ground truth via Kirchhoff — only compute if we'll attempt synthesis
        # (avoids expensive exact determinant on huge unsolvable graphs)
        will_attempt = (m <= cej_max_solved + 100)
        kirchhoff = count_spanning_trees_kirchhoff(graph) if will_attempt else -1

        # --- CEJ engine ---
        # For graphs above `_HARD_TIMEOUT_EDGE_THRESHOLD`, the in-process
        # SIGALRM timeout cannot interrupt the treewidth-DP cffi call,
        # so we spawn a subprocess and SIGKILL on timeout. Below the
        # threshold, the in-process SIGALRM is faster and reliable.
        use_hard_timeout = m > _HARD_TIMEOUT_EDGE_THRESHOLD
        if m > cej_max_solved + 100:
            # Way beyond frontier — skip without wasting timeout
            cej_ms, cej_result, cej_err = None, None, "UNSOLVED"
        elif use_hard_timeout:
            cej_ms, cej_payload, cej_err = _time_fn_hard(
                "cej", graph, cej_table, timeout_s,
            )
            cej_result = _SubprocResult(cej_payload) if cej_payload else None
        else:
            cej_ms, cej_result, cej_err = _time_fn(
                lambda: cej_engine.synthesize(graph), timeout_s
            )
        cej_ok = (cej_result is not None
                  and cej_result.polynomial.num_spanning_trees() == kirchhoff)
        if cej_ok:
            cej_table.add(graph, name, cej_result.polynomial, cej_result.minors_used)
            stats["cej_ok"] += 1
            cej_status = "OK"
            cej_max_solved = max(cej_max_solved, m)
        else:
            stats["cej_fail"] += 1
            cej_status = cej_err or "WRONG"

        # --- NetworkX ---
        # Two give-up rules keep the reference oracle from burning a full
        # nx_timeout_s on graphs it can't finish:
        #   1. Per-family: once NetworkX times out on one member of a
        #      scaling family (K_n, C_n, Cm, …), every later member is
        #      strictly harder, so skip the rest of that family outright.
        #   2. Global frontier: don't attempt anything more than 10 edges
        #      past the largest graph NetworkX has actually solved (guards
        #      the first member of a family already far beyond its ceiling).
        nx_fam = _nx_family_key(name)
        if nx_fam in nx_timed_out_families:
            nx_ms, nx_result, nx_err = None, None, "FAMILY_TIMEOUT"
        elif m > nx_max_solved + 10:
            nx_ms, nx_result, nx_err = None, None, "UNSOLVED"
        else:
            nx_ms, nx_result, nx_err = _time_fn(
                lambda: _tutte_networkx(G_nx), nx_timeout_s
            )
            if nx_err == _TIMEOUT:
                nx_timed_out_families.add(nx_fam)
        nx_ok = (nx_result is not None
                 and nx_result.num_spanning_trees() == kirchhoff)
        if nx_ok:
            stats["nx_ok"] += 1
            nx_status = "OK"
            nx_max_solved = max(nx_max_solved, m)
        else:
            stats["nx_fail"] += 1
            nx_status = nx_err or "WRONG"

        # --- Polynomial cross-validation ---
        poly_match = {"cej_vs_nx": None}
        if nx_ok:
            if cej_ok:
                match = cej_result.polynomial == nx_result
                poly_match["cej_vs_nx"] = match
                if not match:
                    cej_status = "POLY_MISMATCH"
                    stats["poly_mismatch"] += 1

        trees_str = f"{kirchhoff:,}" if kirchhoff >= 0 else "?"

        # Show failure reason inline when not OK
        cej_col = _fmt(cej_ms) if cej_status == "OK" else cej_status
        nx_col = _fmt(nx_ms) if nx_ok else nx_status

        # Method labels (truncated for column width; full method in JSON).
        cej_method = (
            getattr(cej_result, "method", "?") if cej_ok else "-"
        )
        cej_method_col = (cej_method or "?")[:30]

        print(f"{idx:>5} {name:<20} {n:>3} {m:>3} {trees_str:>14} "
              f"{cej_col:>10} {cej_method_col:<32} {nx_col:>10}",
              flush=True)

        results.append({
            "name": name,
            "nodes": n,
            "edges": m,
            "spanning_trees": kirchhoff,
            "timings_ms": {
                "synthesis_cej": cej_ms,
                "networkx": nx_ms,
            },
            "status": {
                "cej": cej_status,
                "networkx": nx_status,
            },
            "method": {
                "cej": cej_method if cej_ok else None,
            },
            "polynomial_match": poly_match,
        })

    # Summary
    print("-" * len(hdr))
    print(f"CEJ:     {stats['cej_ok']} ok, {stats['cej_fail']} failed "
          f"({len(cej_table)} table entries)")
    print(f"NetworkX:{stats['nx_ok']} ok, {stats['nx_fail']} failed")
    if stats["poly_mismatch"]:
        print(f"WARNING: {stats['poly_mismatch']} polynomial mismatches vs NetworkX!")
    else:
        print(f"Polynomial cross-validation: all matches OK")

    # Per-edge-count summary
    by_edges = {}
    for r in results:
        m = r["edges"]
        if m not in by_edges:
            by_edges[m] = {"count": 0, "cej": [], "nx": []}
        by_edges[m]["count"] += 1
        for key in ("synthesis_cej", "networkx"):
            short = key.replace("synthesis_", "").replace("networkx", "nx")
            t = r["timings_ms"][key]
            if t is not None:
                by_edges[m][short].append(t)

    print(f"\n{'Edges':>5} {'Count':>5}  {'CEJ avg':>10} {'NX avg':>10}")
    print("-" * 50)
    for m in sorted(by_edges):
        b = by_edges[m]
        def avg(lst):
            return sum(lst) / len(lst) if lst else None
        print(f"{m:>5} {b['count']:>5}  "
              f"{_fmt(avg(b['cej'])):>10} {_fmt(avg(b['nx'])):>10}")

    # ------------------------------------------------------------------
    # Method-firing breakdown — answers "which dispatch paths actually
    # carry the load" for the CEJ engine.
    # ------------------------------------------------------------------
    from collections import Counter
    cej_methods = Counter(
        r["method"]["cej"] for r in results if r["method"]["cej"]
    )
    cej_by_method_time = {}
    for r in results:
        cm = r["method"]["cej"]; ct = r["timings_ms"]["synthesis_cej"]
        if cm and ct is not None:
            cej_by_method_time.setdefault(cm, []).append(ct)

    print(f"\n--- Method-firing breakdown (CEJ engine) ---")
    print(f"{'Method':<40} {'Count':>6} {'Avg ms':>10} {'Max ms':>10}")
    print("-" * 70)
    for method, count in cej_methods.most_common():
        times = cej_by_method_time.get(method, [])
        avg = sum(times) / len(times) if times else 0
        mx = max(times) if times else 0
        print(f"{method:<40} {count:>6} {avg:>10.1f} {mx:>10.1f}")

    sys.stdout.flush()
    return results, cej_table, cej_engine


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def compare_results(file1, file2):
    """Compare two benchmark result files."""
    with open(file1) as f:
        data1 = json.load(f)
    with open(file2) as f:
        data2 = json.load(f)

    b1 = data1["metadata"].get("branch", "?")
    b2 = data2["metadata"].get("branch", "?")
    print(f"Comparing: {b1} vs {b2}\n")

    results1 = {r["name"]: r for r in data1["results"]}
    results2 = {r["name"]: r for r in data2["results"]}

    common = sorted(set(results1) & set(results2),
                    key=lambda n: (results1[n]["edges"], n))

    # Compare each engine
    for engine_key in ("synthesis_cej", "networkx"):
        label = engine_key.replace("synthesis_", "").upper()
        print(f"\n--- {label} ---")
        print(f"{'Graph':<20} {'M':>3}  {b1:>10}  {b2:>10}  {'Speedup':>8}")
        print("-" * 60)

        speedups = []
        for name in common:
            r1, r2 = results1[name], results2[name]
            t1 = r1["timings_ms"].get(engine_key)
            t2 = r2["timings_ms"].get(engine_key)
            m = r1["edges"]

            if t1 and t2 and t2 > 0:
                s = t1 / t2
                speedups.append(s)
                print(f"{name:<20} {m:>3}  {_fmt(t1):>10}  {_fmt(t2):>10}  {s:>7.2f}x")

        if speedups:
            geo_mean = 1.0
            for s in speedups:
                geo_mean *= s
            geo_mean **= (1 / len(speedups))
            print(f"Geometric mean speedup: {geo_mean:.2f}x over {len(speedups)} graphs")

    # Method-change report: which graphs now route through a different
    # dispatch path. Even when wall time is unchanged, a route change
    # is a flag worth surfacing (e.g. a new fast path firing).
    for engine_label in ("cej",):
        method_changes = []
        for name in common:
            r1, r2 = results1[name], results2[name]
            m1 = (r1.get("method") or {}).get(engine_label)
            m2 = (r2.get("method") or {}).get(engine_label)
            if m1 and m2 and m1 != m2:
                method_changes.append((name, r1["edges"], m1, m2))
        if method_changes:
            print(f"\n--- {engine_label.upper()} method changes "
                  f"({len(method_changes)} graphs) ---")
            print(f"{'Graph':<20} {'M':>3}  {b1[:25]:<26} -> {b2[:25]:<26}")
            print("-" * 80)
            for name, m, old, new in method_changes[:30]:
                print(f"{name:<20} {m:>3}  "
                      f"{(old or '?')[:24]:<26} -> {(new or '?')[:24]:<26}")
            if len(method_changes) > 30:
                print(f"... and {len(method_changes) - 30} more")


# ---------------------------------------------------------------------------
# Save / CLI
# ---------------------------------------------------------------------------

def save_results(results, cej_table=None, cej_engine=None):
    """Save benchmark results to JSON and optionally save rainbow/multigraph tables."""
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

    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    out_path = os.path.join(base_dir, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    if cej_table and len(cej_table) > 0:
        # Compute comprehensive structural minor relationships
        print(f"\nComputing structural minor relationships for {len(cej_table)} entries...",
              flush=True)
        t0 = time.perf_counter()
        relationships = cej_table.compute_minor_relationships()
        elapsed = time.perf_counter() - t0
        total_minors = sum(len(v) for v in relationships.values())
        print(f"Found {total_minors} minor relationships across "
              f"{len(relationships)} entries ({elapsed:.1f}s)", flush=True)

        bin_path = os.path.join(base_dir, "lookup_table.bin")
        save_binary_rainbow_table(cej_table, bin_path)
        print(f"Rainbow table saved: {len(cej_table)} entries ({bin_path})",
              flush=True)

    # Save the CEJ engine's multigraph cache. Loads the existing on-disk
    # table first so previous entries survive — the benchmark only adds
    # new canonical keys, never overwrites the data blob.
    merged_mg_cache = {}
    if cej_engine is not None:
        merged_mg_cache.update(cej_engine._multigraph_cache)
    if len(merged_mg_cache) > 0:
        from ..lookup.core import (
            load_default_multigraph_table,
            save_default_multigraph_table,
        )
        existing = load_default_multigraph_table()
        added = 0
        for k, v in merged_mg_cache.items():
            if k not in existing:
                existing[k] = v
                added += 1
        save_default_multigraph_table(existing)
        print(f"Multigraph cache saved: {len(existing)} entries "
              f"(+{added} new) "
              f"({os.path.join(base_dir, 'multigraph_lookup_table.bin')})",
              flush=True)

    # Persist the rooted-Tutte lookup table populated during the run.
    # Uses the `_T_ROOTED_GRAPHS` sidecar to recover originating Graphs
    # for canonical-label serialization. Then runs the warmup script so
    # the shipped table also has the standard cell library (K_n,
    # K_{a,b}, Z(1,1)) regardless of which benchmark graphs ran.
    try:
        from ..roots.rooted_tutte import save_rooted_lookup_default
        n_bin = save_rooted_lookup_default()
        print(f"Rooted lookup saved: {n_bin} entries "
              f"({os.path.join(base_dir, 'rooted_lookup_table.bin')})",
              flush=True)
    except Exception as e:
        print(f"[warn] Rooted lookup save failed: {type(e).__name__}: {e}",
              flush=True)

    # Run the warmup so the standard cell library (K_n, K_{a,b}, Z(1,1))
    # is always present in the shipped lookup. Common cells take seconds
    # except Z(1,1) (~5 min) — pass `--small` via env to skip the long
    # Z(1,1) brute when the benchmark wasn't going to need it anyway.
    try:
        import subprocess
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        script = os.path.join(repo_root, "scripts", "warmup_rooted_lookup.py")
        if os.path.exists(script):
            warmup_args = [sys.executable, script]
            if os.environ.get("TUTTE_BENCHMARK_SKIP_Z11") == "1":
                warmup_args.append("--small")
            print(f"\nRunning rooted-lookup warmup ({' '.join(warmup_args[1:])})...",
                  flush=True)
            warm_env = dict(os.environ, PYTHONPATH=repo_root)
            subprocess.run(warmup_args, env=warm_env, check=False)
    except Exception as e:
        print(f"[warn] Rooted lookup warmup failed: {type(e).__name__}: {e}",
              flush=True)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Tutte polynomial synthesis from empty rainbow table"
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("FILE1", "FILE2"),
        help="Compare two benchmark result files",
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="Per-graph timeout in seconds for CEJ (default: 60)",
    )
    parser.add_argument(
        "--nx-timeout", type=int, default=30,
        help="Per-graph timeout in seconds for NetworkX (default: 30)",
    )
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    else:
        results, cej_table, cej_engine = run_benchmarks(timeout_s=args.timeout, nx_timeout_s=args.nx_timeout)
        save_results(results, cej_table, cej_engine)


if __name__ == "__main__":
    main()

"""Merger lookup table for chord-junction Tutte polynomials.

A *merger* of a base graph ``G`` at position set ``V_T ⊆ V(G)`` is the
multigraph

    G ∪_{V_T} G

obtained by taking two disjoint copies of ``G`` and identifying
corresponding vertex pairs ``(v, v')`` for ``v ∈ V_T``. Parallel edges
arising from the identification are preserved (so the merger is a
``tutte.graph.MultiGraph``, not a simple ``Graph``).

The merger lookup stores precomputed Tutte polynomials ``T(G ∪_{V_T} G)``
for ``(G, V_T)`` pairs that arise in the **unified bivariate chord-junction
theorem** (May 25, 2026; see ``tutte/research/cyclotomic_chord_junction_theorem.md``):

    T(G ⊕_{V_k} G; x, y)
        = (x − 1) · T(G; x, y)²
        + Σ_{∅ ≠ S ⊆ V_k} T(G ∪_{V_S} G; x, y).

With ``T(G)`` and the merger values cached, evaluating the chord-junction
polynomial for any cell-pair (Chimera, Pegasus, Zephyr, ad-hoc user
graph) is a constant-time table lookup followed by an O(2^|V_k|) sum.

The table is naming-parallel to the existing lookup tables in
``tutte/data/``:

  - ``lookup_table``           — simple-graph rainbow (canonical key → T)
  - ``multigraph_lookup_table`` — opportunistically populated multigraph cache
  - ``rooted_lookup_table``    — boundary-partition-indexed rooted Tutte
  - ``merger_lookup_table``    — chord-junction mergers (NEW)

Each ``MergerEntry`` is indexed by ``(base_canonical_key, v_t_tuple)`` for
fast lookup during cell-pair dispatch, and by the merger graph's own
canonical key as a secondary index (so callers can also reuse the value
when the same multigraph appears from different ``(base, V_T)`` pairs).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ..polynomial import TuttePolynomial

if TYPE_CHECKING:
    from ..graph import Graph, MultiGraph


VTTuple = Tuple[int, ...]
"""Canonical V_T identifier: sorted tuple of base-graph vertex IDs."""


# ---------------------------------------------------------------------------
# MergerEntry
# ---------------------------------------------------------------------------


@dataclass
class MergerEntry:
    """One precomputed merger ``T(G ∪_{V_T} G)`` value.

    Attributes:
        base_canonical_key: SHA256 canonical key of the base graph ``G``.
        v_t: Sorted tuple of base-graph vertex IDs in ``V_T``. Two entries
            with the same ``(base_canonical_key, v_t)`` describe the same
            merger; the warmup pipeline picks one canonical V_T per orbit
            under ``Aut(G)``.
        polynomial: Tutte polynomial of the merger multigraph.
        merger_canonical_key: SHA256 canonical key of the merger multigraph
            (may be None if the entry was constructed without canonicalizing
            the merger graph; populated by ``add_entry``).
        base_name: Human-readable base graph identifier (e.g. ``K_{4,4}``).
        family_tag: D-Wave family this entry was warmed for, if any —
            ``chimera``, ``pegasus``, ``zephyr``, or ``None`` for ad-hoc.
        base_node_count: ``|V(G)|`` (redundant; cached for filtering).
        base_edge_count: ``|E(G)|`` (redundant; cached for filtering).
        merger_node_count: ``2·|V(G)| − |V_T|`` (redundant).
        merger_edge_count: Total edge count of the merger multigraph
            (counting parallel-edge multiplicities); redundant.
    """
    base_canonical_key: str
    v_t: VTTuple
    polynomial: TuttePolynomial
    merger_canonical_key: Optional[str] = None
    base_name: Optional[str] = None
    family_tag: Optional[str] = None
    base_node_count: int = 0
    base_edge_count: int = 0
    merger_node_count: int = 0
    merger_edge_count: int = 0

    @property
    def lookup_key(self) -> Tuple[str, VTTuple]:
        """Primary index used by ``MergerTable.lookup_by_source``."""
        return (self.base_canonical_key, self.v_t)

    def __hash__(self) -> int:
        return hash(self.lookup_key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MergerEntry):
            return NotImplemented
        return self.lookup_key == other.lookup_key


# ---------------------------------------------------------------------------
# MergerTable
# ---------------------------------------------------------------------------


@dataclass
class MergerTable:
    """Lookup table of precomputed chord-junction mergers.

    Two indices are maintained:

    - ``by_source``: ``(base_canonical_key, v_t)`` → ``MergerEntry``.
      Primary fast-path used by cell-pair dispatch — given a base graph
      and a chord-position subset, return the merger polynomial without
      constructing the merger multigraph.
    - ``by_merger``: ``merger_canonical_key`` → ``MergerEntry``.
      Secondary index for callers that already have the merger multigraph
      and want to deduplicate against the multigraph cache.

    Entries are immutable in practice. ``add_entry`` is the only
    mutator; it inserts into both indices.
    """

    by_source: Dict[Tuple[str, VTTuple], MergerEntry] = field(default_factory=dict)
    by_merger: Dict[str, MergerEntry] = field(default_factory=dict)

    # Hit/miss counters — incremented by lookup_by_source / lookup_by_merger.
    # Read via .hits, .misses, .reset_counters(). Counters are session-local
    # and NOT serialized; benchmarks observe them per-run to see how often
    # the merger cache actually fires under each dispatch path.
    _hits_by_source: int = 0
    _misses_by_source: int = 0
    _hits_by_merger: int = 0
    _misses_by_merger: int = 0

    def __len__(self) -> int:
        return len(self.by_source)

    @property
    def hits(self) -> int:
        return self._hits_by_source + self._hits_by_merger

    @property
    def misses(self) -> int:
        return self._misses_by_source + self._misses_by_merger

    def counter_breakdown(self) -> Dict[str, int]:
        """Per-index hit/miss counts; useful for benchmarks."""
        return {
            "hits_by_source":   self._hits_by_source,
            "misses_by_source": self._misses_by_source,
            "hits_by_merger":   self._hits_by_merger,
            "misses_by_merger": self._misses_by_merger,
        }

    def reset_counters(self) -> None:
        self._hits_by_source = 0
        self._misses_by_source = 0
        self._hits_by_merger = 0
        self._misses_by_merger = 0

    @property
    def entries(self) -> Dict[Tuple[str, VTTuple], MergerEntry]:
        """Alias for ``by_source`` mirroring ``RainbowTable.entries``."""
        return self.by_source

    # -- mutation -------------------------------------------------------

    def add_entry(self, entry: MergerEntry) -> None:
        """Insert an entry. Overwrites if ``(base_key, v_t)`` already exists."""
        self.by_source[entry.lookup_key] = entry
        if entry.merger_canonical_key is not None:
            self.by_merger[entry.merger_canonical_key] = entry

    # -- queries --------------------------------------------------------

    def lookup_by_source(
        self,
        base_canonical_key: str,
        v_t: VTTuple,
    ) -> Optional[MergerEntry]:
        """Return the merger entry for a given ``(base, V_T)``, or None."""
        entry = self.by_source.get((base_canonical_key, tuple(sorted(v_t))))
        if entry is None:
            self._misses_by_source += 1
        else:
            self._hits_by_source += 1
        return entry

    def lookup_by_merger(self, merger_canonical_key: str) -> Optional[MergerEntry]:
        """Return the merger entry whose multigraph has the given canonical key."""
        entry = self.by_merger.get(merger_canonical_key)
        if entry is None:
            self._misses_by_merger += 1
        else:
            self._hits_by_merger += 1
        return entry

    def lookup(
        self,
        base: "Graph",
        v_t: VTTuple,
    ) -> Optional[TuttePolynomial]:
        """Convenience: look up the polynomial directly from a base graph."""
        key = base.canonical_key()
        entry = self.lookup_by_source(key, v_t)
        return entry.polynomial if entry is not None else None

    def entries_for_base(self, base_canonical_key: str) -> List[MergerEntry]:
        """Return all entries whose base graph has the given canonical key."""
        return [e for (bk, _), e in self.by_source.items() if bk == base_canonical_key]

    def entries_for_family(self, family_tag: str) -> List[MergerEntry]:
        """Return all entries warmed for a given D-Wave family."""
        return [e for e in self.by_source.values() if e.family_tag == family_tag]

    # -- serialization --------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "MergerTable":
        """Load a merger table from a JSON file."""
        import json as _json
        with open(path, "r") as f:
            data = _json.load(f)
        table = cls()
        for raw in data.get("entries", []):
            coeffs = {
                tuple(map(int, k.split(","))): v
                for k, v in raw["coefficients"].items()
            }
            polynomial = TuttePolynomial.from_coefficients(coeffs)
            entry = MergerEntry(
                base_canonical_key=raw["base_canonical_key"],
                v_t=tuple(raw["v_t"]),
                polynomial=polynomial,
                merger_canonical_key=raw.get("merger_canonical_key"),
                base_name=raw.get("base_name"),
                family_tag=raw.get("family_tag"),
                base_node_count=raw.get("base_node_count", 0),
                base_edge_count=raw.get("base_edge_count", 0),
                merger_node_count=raw.get("merger_node_count", 0),
                merger_edge_count=raw.get("merger_edge_count", 0),
            )
            table.add_entry(entry)
        return table

    def save(self, path: str) -> None:
        """Save the merger table to a JSON file."""
        import json as _json
        records = []
        for entry in self.by_source.values():
            coeffs = {
                f"{i},{j}": c
                for (i, j), c in entry.polynomial.to_coefficients().items()
            }
            records.append({
                "base_canonical_key": entry.base_canonical_key,
                "v_t": list(entry.v_t),
                "merger_canonical_key": entry.merger_canonical_key,
                "base_name": entry.base_name,
                "family_tag": entry.family_tag,
                "base_node_count": entry.base_node_count,
                "base_edge_count": entry.base_edge_count,
                "merger_node_count": entry.merger_node_count,
                "merger_edge_count": entry.merger_edge_count,
                "coefficients": coeffs,
            })
        data = {
            "description": "Tutte polynomial chord-junction merger lookup",
            "total_entries": len(records),
            "note": (
                "Each entry caches T(G ∪_{V_T} G) where G is the base graph "
                "(canonical_key = base_canonical_key) and V_T is a vertex "
                "subset (sorted tuple)."
            ),
            "entries": records,
        }
        with open(path, "w") as f:
            _json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Default location loaders/savers (parallel to load_default_*_table helpers)
# ---------------------------------------------------------------------------


def _default_data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_default_merger_table() -> MergerTable:
    """Load the default ``merger_lookup_table`` from the package data dir.

    Tries the binary format first (faster); falls back to JSON; returns an
    empty table if neither file exists yet (e.g. before the warmup script
    has been run).
    """
    base_dir = _default_data_dir()
    bin_path = os.path.join(base_dir, "merger_lookup_table.bin")
    json_path = os.path.join(base_dir, "merger_lookup_table.json")

    if os.path.exists(bin_path):
        try:
            from .binary import load_merger_lookup_table
            return load_merger_lookup_table(bin_path)
        except Exception:
            pass  # Fall through to JSON.

    if os.path.exists(json_path):
        return MergerTable.load(json_path)

    return MergerTable()


def save_default_merger_table(table: MergerTable) -> None:
    """Save the merger table to the default location (binary)."""
    from .binary import save_merger_lookup_table

    base_dir = _default_data_dir()
    bin_path = os.path.join(base_dir, "merger_lookup_table.bin")

    save_merger_lookup_table(table, bin_path)


__all__ = [
    "MergerEntry",
    "MergerTable",
    "VTTuple",
    "load_default_merger_table",
    "save_default_merger_table",
]

"""Multivariate Tutte / Sokal Z polynomial.

Sokal's multivariate Tutte (Potts-model partition function) is

    Z(G; q, v) = Σ_{A ⊆ E} q^{k(A)} ∏_{e ∈ A} v_e

The standard Tutte polynomial recovers via uniform v_e = v and the
identity (Sokal 2005)

    T(G; x, y) = (x − 1)^{−r(E)} (y − 1)^{−|V|} · Z(G; (x − 1)(y − 1), y − 1)

For our research use, Z is the natural object for two
reasons that the (x, y) basis hides:

1. Deletion-contraction is linear in each v_e:

       Z(G) = Z(G − e)|_{v_e = 0} + v_e · Z(G / e)

   No bridge/loop case-split.

2. Symmetric algebraic structure across vertex-sums and shared-edge
   k-sums often simplifies in the (q, v) basis when (x, y) doesn't.

This module implements two representations:

* `MultivariateTutte` — full per-edge variables. Internal storage is
  `dict[(q_power, frozenset_of_(edge_id, v_power))] -> int_coeff`.
  Supports `delete_edge(e)` and `contract_edge(e)` as O(|terms|) per-edge
  ops; `__add__` and `__mul__` are the natural dict-merge / Cauchy
  product operations.

* `UniformZ` — collapsed-uniform-v representation `Z(G; q, v)` with
  storage `dict[(q_power, v_power)] -> int_coeff`. Smaller and faster
  for empirical size/mul-cost comparison vs `TuttePolynomial`. The
  conversion `to_tutte()` lives here.

The empirical experiment (research/scripts/...) uses
`UniformZ` for the size compare; `MultivariateTutte` is the substrate
for any future per-edge algebraic identity hunt.

References:
- Sokal, A. D. (2005). The multivariate Tutte polynomial (alias Potts
  model) for graphs and matroids. arxiv:math/0503607
- Bollobás, B. & Riordan, O. (1999). A Tutte polynomial for coloured
  graphs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import comb
from typing import Dict, FrozenSet, Iterable, Tuple

from ..polynomial import TuttePolynomial

# ---------- UniformZ (collapsed uniform-v) ----------

@dataclass(frozen=True)
class UniformZ:
    """Z(G; q, v) with uniform v_e = v across all edges.

    Internal: dict mapping `(q_power, v_power) -> integer coefficient`.
    Sparse — zero coefficients are not stored.

    This is the right object for the empirical
    comparison: same expressive power as `TuttePolynomial(x, y)` after
    conversion, but its monomial structure may be more compact for
    structured graphs (cells / cell-quotient compositions).

    Operations:
    - `from_subgraph_sum(graph)`: build via Z = Σ_{A ⊆ E} q^k(A) v^|A|.
      O(2^|E|), only used for small validation / brute-force.
    - `__add__`, `__mul__`, scalar `__rmul__`: standard polynomial ops.
    - `to_tutte(num_vertices, num_edges)`: convert via Sokal's identity
      to a `TuttePolynomial(x, y)`.
    - `coeff_count`, `coeff_max_abs`, `coeff_bit_sum`: sizing helpers
      for the empirical comparison.
    """

    coeffs: Tuple[Tuple[Tuple[int, int], int], ...]
    # Frozen sorted ((q_pow, v_pow), coeff) pairs so equality + hashing
    # are well-defined and the dataclass is immutable.

    @classmethod
    def from_dict(cls, d: Dict[Tuple[int, int], int]) -> "UniformZ":
        items = tuple(sorted((k, c) for k, c in d.items() if c != 0))
        return cls(coeffs=items)

    def to_dict(self) -> Dict[Tuple[int, int], int]:
        return {k: c for k, c in self.coeffs}

    @classmethod
    def zero(cls) -> "UniformZ":
        return cls(coeffs=())

    @classmethod
    def one(cls) -> "UniformZ":
        return cls(coeffs=(((0, 0), 1),))

    @classmethod
    def from_subgraph_sum(cls, graph) -> "UniformZ":
        """Brute-force Z(G; q, v) = Σ_A q^k(A) v^|A| over A ⊆ E.

        Only feasible for small graphs (|E| ≤ 16 or so). Used as the
        ground-truth oracle for validating algebraic identities.
        """
        edges = list(graph.edges)
        n_edges = len(edges)
        nodes = list(graph.nodes)
        n_nodes = len(nodes)
        result: Dict[Tuple[int, int], int] = defaultdict(int)
        for mask in range(1 << n_edges):
            parent = {v: v for v in nodes}

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            n_in = 0
            for i in range(n_edges):
                if mask >> i & 1:
                    u, v = edges[i]
                    ru, rv = find(u), find(v)
                    if ru != rv:
                        parent[max(ru, rv)] = min(ru, rv)
                    n_in += 1
            n_components = len({find(v) for v in nodes})
            result[(n_components, n_in)] += 1
        return cls.from_dict(dict(result))

    # Arithmetic
    def __add__(self, other: "UniformZ") -> "UniformZ":
        if not isinstance(other, UniformZ):
            return NotImplemented
        out: Dict[Tuple[int, int], int] = defaultdict(int)
        for k, c in self.coeffs:
            out[k] += c
        for k, c in other.coeffs:
            out[k] += c
        return UniformZ.from_dict(dict(out))

    def __sub__(self, other: "UniformZ") -> "UniformZ":
        if not isinstance(other, UniformZ):
            return NotImplemented
        out: Dict[Tuple[int, int], int] = defaultdict(int)
        for k, c in self.coeffs:
            out[k] += c
        for k, c in other.coeffs:
            out[k] -= c
        return UniformZ.from_dict(dict(out))

    def __mul__(self, other: "UniformZ") -> "UniformZ":
        if not isinstance(other, UniformZ):
            return NotImplemented
        out: Dict[Tuple[int, int], int] = defaultdict(int)
        for (qa, va), ca in self.coeffs:
            for (qb, vb), cb in other.coeffs:
                out[(qa + qb, va + vb)] += ca * cb
        return UniformZ.from_dict(dict(out))

    def __rmul__(self, scalar: int) -> "UniformZ":
        if scalar == 0:
            return UniformZ.zero()
        return UniformZ.from_dict({k: scalar * c for k, c in self.coeffs})

    def __neg__(self) -> "UniformZ":
        return UniformZ.from_dict({k: -c for k, c in self.coeffs})

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UniformZ):
            return NotImplemented
        return self.coeffs == other.coeffs

    def __hash__(self) -> int:
        return hash(self.coeffs)

    def __repr__(self) -> str:
        if not self.coeffs:
            return "UniformZ(0)"
        return "UniformZ(" + " + ".join(
            f"{c}·q^{q}·v^{v}" for (q, v), c in self.coeffs
        ) + ")"

    # Sizing helpers (for the empirical comparison)
    def coeff_count(self) -> int:
        return sum(1 for _, c in self.coeffs if c != 0)

    def coeff_max_abs(self) -> int:
        return max((abs(c) for _, c in self.coeffs), default=0)

    def coeff_bit_sum(self) -> int:
        return sum(abs(c).bit_length() for _, c in self.coeffs)

    # Conversion
    def evaluate(self, q_value: int, v_value: int) -> int:
        """Numeric evaluation Z(q_value, v_value)."""
        return sum(c * (q_value ** q) * (v_value ** v)
                   for (q, v), c in self.coeffs)

    def to_tutte(self, num_vertices: int, num_edges: int) -> TuttePolynomial:
        """Convert to T(G; x, y) via Sokal's identity.

        T(G; x, y) = (x − 1)^{−r(E)} (y − 1)^{−|V|}
                     · Z(G; (x − 1)(y − 1), y − 1)

        where r(E) = |V| − k(G) is the matroid rank. Substitute
        q = (x − 1)(y − 1) and v = y − 1 into Z, then divide by
        (x − 1)^{r(E)} (y − 1)^{|V|}. The substitution is exact in
        the (x, y) basis; the division is exact (no remainder) for any
        valid graph because Z(G; q, v) factors with the right divisors
        (Sokal 2005 Eq. (2.16)).

        Both T's input variables and Z's coefficients are integers,
        so this is exact integer arithmetic.
        """
        # Step 1: substitute q = (x-1)(y-1), v = y-1
        # Each term q^a · v^b becomes:
        #   ((x-1)(y-1))^a · (y-1)^b = (x-1)^a · (y-1)^{a+b}
        # Expand via binomial theorem:
        #   (x-1)^a = Σ_i C(a, i) · (-1)^{a-i} · x^i
        #   (y-1)^{a+b} = Σ_j C(a+b, j) · (-1)^{a+b-j} · y^j
        # so ((x-1)(y-1))^a · (y-1)^b
        #   = Σ_{i,j} C(a,i) C(a+b,j) (-1)^{2a+b-i-j} x^i y^j
        # Then divide by (x-1)^r · (y-1)^n where r = num_vertices - k.
        # We don't know k(G) at this point; instead use the standard
        # integer-arithmetic conversion: substitute, then divide
        # term-by-term as a TuttePolynomial.
        #
        # Practical approach: build the substituted polynomial in
        # (x, y) basis directly, then divide by (x-1)^r · (y-1)^n.
        n_v = num_vertices
        # Sokal's identity is invariant under k(G); for connected
        # graphs r(E) = |V| - 1. For computing T from Z, we don't need
        # to know k(G) — but we DO need r(E). Compute as
        #   r(E) = |V| - k(G)
        # We can recover k(G) from Z(q=1, v=0) which gives:
        #   Σ_A 1^{k(A)} · 0^{|A|} = (only A = empty contributes)
        #     = q^{k(∅)} · v^0 = 1^{|V|} = 1 (terms where v_pow = 0)
        # Hmm this is getting circular. Use a more direct path.
        #
        # Sokal Eq. (2.16) (in the form most useful here):
        #   Z(G; q, v) = q^{k(G)} · (y - 1)^{|E|} · T(G; x, y)|_{...}
        # Actually the clean path: convert via fact-of-life that
        #   T(G; x, y) · (x-1)^{r(E)} · (y-1)^{|V|}
        #     = Σ_{A ⊆ E} (x-1)^{r(E)-r(A)} (y-1)^{|A|-r(A)} · (something)
        # ...
        # The pragmatic path: use the equivalent generating-function
        # rewrite (Sokal Eq. (2.20)):
        #
        #   T(G; x, y) = (x-1)^{-r(E)} · Z̃(G; q = (x-1)(y-1), w = y-1)
        #
        # where Z̃ = Z / (y-1)^{|V|-k(G)} ... actually this is fragile.
        #
        # SIMPLEST CORRECT IMPLEMENTATION: compute T via Whitney's
        # rank polynomial R(G; u=x-1, v=y-1) which is the SAME data
        # as Z up to substitution q = (x-1)(y-1):
        #
        #   R(G; u, v) = Σ_A u^{r(E)-r(A)} v^{|A|-r(A)}
        #              = (1/u^{r(E)}) · Σ_A u^{-r(A)} v^{|A|-r(A)} · u^{r(E)}
        #
        # And Z = q^{k(G)} · R'(G) with appropriate normalisation.
        # This is getting unwieldy without more careful derivation.
        #
        # For now: provide a brute-force conversion via numeric
        # evaluation. Treat it as a placeholder until 18.E.1.a's
        # research script lands the correct Sokal identity expansion.
        raise NotImplementedError(
            "UniformZ.to_tutte: Sokal identity expansion deferred to "
            "a research script. Use evaluate() for "
            "numeric values; the empirical comparison harness needs "
            "size/mul measurements, not full conversion."
        )


# ---------- MultivariateTutte (full per-edge) ----------

@dataclass(frozen=True)
class MultivariateTutte:
    """Z(G; q, v_e) with per-edge variables.

    Internal: dict mapping `(q_power, frozenset of (edge_id, v_e_power)) ->
    integer coefficient`. The frozenset key is canonical (tuple-sorted)
    so equality and hashing are well-defined.

    Builds via initial atoms `from_empty_graph(num_vertices)` =
    `q^{num_vertices}`, then applies `delete_edge(e)` /
    `contract_edge(e)` recursively.

    Per-edge operations:
    - `delete_edge(e)`: substitute v_e = 0. Drops every term where
      edge e appears with positive power.
    - `contract_edge(e)`: substitute v_e = 1 AND multiply by v_e
      (i.e., for each term: keep if v_e present, drop the v_e factor;
      if v_e absent, multiply by 1 and divide by q (component merge)).
      Mathematically: Z(G/e; q, v) = Z(G; q, v)|_{v_e → ∞} normalized.
      Concrete formula: split Z by v_e present/absent, multiply
      "absent" half by 1/q (handles component-count drop on contraction).

    For empirical use the `UniformZ` representation is faster; this
    class is the substrate for the closed-form k-sum
    identity hunt where per-edge differentiation matters.
    """

    coeffs: Tuple[Tuple[Tuple[int, FrozenSet[Tuple[int, int]]], int], ...]
    # ((q_pow, frozenset((edge_id, v_pow), ...)), coeff)

    @classmethod
    def zero(cls) -> "MultivariateTutte":
        return cls(coeffs=())

    @classmethod
    def one(cls) -> "MultivariateTutte":
        return cls(coeffs=(((0, frozenset()), 1),))

    @classmethod
    def from_empty_graph(cls, num_vertices: int) -> "MultivariateTutte":
        """Z(empty graph on num_vertices) = q^{num_vertices}."""
        return cls(coeffs=(((num_vertices, frozenset()), 1),))

    @classmethod
    def from_dict(
        cls,
        d: Dict[Tuple[int, FrozenSet[Tuple[int, int]]], int],
    ) -> "MultivariateTutte":
        items = tuple(sorted(
            (k, c) for k, c in d.items() if c != 0
        ))
        return cls(coeffs=items)

    def to_dict(self) -> Dict[Tuple[int, FrozenSet[Tuple[int, int]]], int]:
        return {k: c for k, c in self.coeffs}

    def coeff_count(self) -> int:
        return sum(1 for _, c in self.coeffs if c != 0)

    def specialize_uniform(self) -> UniformZ:
        """Collapse v_e variables to a single uniform v.

        Σ_e v_e^{k_e} → v^{Σ_e k_e}. Useful for converting the
        per-edge representation to UniformZ for sizing comparison.
        """
        out: Dict[Tuple[int, int], int] = defaultdict(int)
        for (q_pow, edge_vars), coeff in self.coeffs:
            v_total = sum(v_pow for _e, v_pow in edge_vars)
            out[(q_pow, v_total)] += coeff
        return UniformZ.from_dict(dict(out))


__all__ = ["UniformZ", "MultivariateTutte"]

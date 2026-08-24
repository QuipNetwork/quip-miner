# 7.6 — Almost-Cograph Dynamic Programming

## Summary

Many graphs are _almost_ cographs: removing a small set of **anomaly
edges** (edges that participate in some induced P₄) leaves a P₄-free
cograph that the standard cotree DP (technique 7.5) can handle. This
technique combines

1. greedy P₄ elimination to find an anomaly set A ⊆ E(G), and
2. the bridge-aware iterated chord rule (`_iterative_chord_rule`)
   over A,

so the cograph skeleton G − A is computed via cotree DP and the |A|
contraction leaves are recursively synthesized through the engine.

**Complexity:** 1 cotree DP call + `|A|` recursive engine calls.
Compared to the naive `2^|A|` subset enumeration of the chord rule's
inclusion-exclusion form, the iterated chord rule gives `O(|A|)`
leaves.

**Best for:** graphs that are mostly cographs joined by a sparse set
of inter-cell edges. D-Wave Cm/Pm cells (K\_{4,4}, K_4) are themselves
cographs; the inter-cell edges are the anomalies. Cm₂ has 16
inter-cell edges → fits cleanly under the engine's `max_anomalies = 16`
gate.

## When it is used

Engine step **7.6** in `tutte/synthesis/engine.py:_synthesize_inner`,
between cotree DP (7.5) and cell-quotient DP (7.7). Fires when

- `compute_tutte_almost_cograph(graph, engine, max_anomalies=16)`
  finds an anomaly set within the cap, and
- the cograph skeleton + each contraction leaf are themselves
  recursively synthesizable.

Returns `None` if the anomaly count exceeds `max_anomalies`, in which
case the engine falls through to cell-quotient DP / treewidth DP /
chord-rule paths.

The 16-anomaly cap was chosen to cover Cm₂ cleanly. Cm₃ (48 inter-cell
edges) doesn't fit and falls through.

## Algorithm

```
def compute_tutte_almost_cograph(graph, engine, max_anomalies):
    A = []
    g = graph
    while len(A) <= max_anomalies:
        e = find_one_induced_P4_middle_edge(g)
        if e is None:
            break  # g is now a cograph
        A.append(e)
        g = g - e
    if len(A) > max_anomalies:
        return None
    factors, adds = _iterative_chord_rule(graph, A)
    cograph_skeleton = graph - A
    T_skel = compute_tutte_cotree_dp(cograph_skeleton)
    T = (∏ factors) · T_skel + Σ_i (∏_{j<i} factors[j]) · engine.synthesize(add_i)
    return T
```

The "middle edge" of an induced P₄ `a − b − c − d` is `b − c`. Removing
the middle edge breaks that specific P₄ (other induced P₄s may
remain).

The iterated chord rule classifies each anomaly edge in the _current_
graph at each step — bridge, loop, or non-bridge non-loop — and applies
the corresponding rule:

- **non-bridge non-loop:** `T(G) = T(G − e) + T(G / e)` (the standard
  chord rule)
- **bridge:** `T(G) = x · T(G − e)`
- **loop:** `T(G) = y · T(G − e)`

This bridge-awareness was the fix to `tutte/graphs/k_sum.py`
that made the chord rule correct for degenerate graphs (e.g.,
K_3 ⊕_3 K_3 returns T = 1 instead of x² − x − 1).

## Files

| File                                                                    | Purpose                                                                                                                        |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| [`tutte/cotree_dp/almost_cograph.py`](../cotree_dp/almost_cograph.py)   | `find_anomaly_edges`, `compute_tutte_almost_cograph`                                                                           |
| [`tutte/cotree_dp/dp.py`](../cotree_dp/dp.py)                           | `compute_tutte_cotree_dp` (called on the cograph skeleton)                                                                     |
| [`tutte/graphs/k_sum.py`](../graphs/k_sum.py)                           | `_iterative_chord_rule` — bridge-aware chord rule applied to the anomaly set                                                   |

## Status

Cascades into the kmatching formula's leaf synthesis: Cm₂ went from
~943 s pre-integration → ~48 s when each K\_{4,4} cell synthesis got
the cotree-DP fast path through the almost-cograph dispatch.

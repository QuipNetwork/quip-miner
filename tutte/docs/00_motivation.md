# Tutte Polynomials as a Difficulty Mechanism

This document describes the motivation for using Tutte polynomials as a structural difficulty measure in the Quip Protocol's quantum proof-of-work system.

## 1. The Problem

The Quip Protocol uses Ising model optimization as its proof-of-work puzzle. For each block, a random Ising model is generated on a fixed processor topology (e.g., Zephyr Z(9,2) with 1,368 qubits and 7,692 couplers). Miners compete to find low-energy spin configurations that satisfy a difficulty threshold.

A miner's solutions must achieve energy below this threshold to constitute a valid proof. Setting the threshold correctly requires an accurate estimate of the inherent difficulty of each Ising instance:

- An overly lenient threshold results in trivially fast block production, undermining network security.
- An overly strict threshold renders blocks unsolvable, stalling the chain.

The current approach uses an empirical scaling formula:

```
Expected ground state energy ≈ -c × √(avg_degree) × N
```

Where:

- `N` — the number of qubits in the topology
- `c` — an empirical constant calibrated from hardware data (0.75 for Advantage2)
- `avg_degree` — the average vertex degree of the topology graph (2M/N, where M is the number of edges)

This formula provides a reasonable first-order estimate, but it treats all Ising instances on the same topology as equally difficult. In practice, the difficulty of an Ising instance depends on the structure of its **coupling graph** — the subgraph of the processor topology formed by the non-zero couplers (J ≠ 0) in that particular instance. Two instances on the same topology can have very different coupling graphs, and therefore very different difficulty, even when their edge counts are similar.

The Tutte polynomial addresses this limitation by providing a structural measure of graph complexity that accounts for the specific coupling graph of each instance.

## 2. Why Tutte Polynomials?

The Tutte polynomial `T(G; x, y)` is a bivariate polynomial that encodes structural information about a graph `G`. It is the most general graph invariant computable by deletion-contraction, and it specializes to several classical graph invariants:

| Evaluation | Invariant |
|---|---|
| `T(1, 1)` | Number of spanning trees |
| `T(2, 1)` | Number of spanning forests |
| `T(1, 2)` | Number of spanning connected subgraphs |
| `T(2, 0)` | Number of acyclic orientations |

These invariants collectively describe the connectivity, cyclicity, and combinatorial complexity of a graph within a single algebraic object.

### Structural Fingerprint

Two non-isomorphic graphs may share the same edge count, degree sequence, or even spectrum, yet they almost never share the same Tutte polynomial. `T(G)` therefore serves as a structural fingerprint that distinguishes graphs by their internal topology.

For Ising models, this is significant because the hardness of finding low-energy states depends on the structure of the coupling graph: the number and arrangement of cycles, the location of bottlenecks, and the graph's decomposition properties.

Consider two Ising instances on the same Zephyr topology with different coupling patterns:

- **Instance A** has many zero couplers, producing a coupling graph that is nearly a tree. Tree-structured problems are efficiently solvable by both quantum and classical methods.
- **Instance B** has a dense, highly connected coupling graph with many frustrated cycles. Such problems are computationally difficult for all known solvers.

Both instances have similar edge counts, so the empirical scaling formula assigns comparable difficulty. However, `T(G_A)` and `T(G_B)` are fundamentally different polynomials: `T(G_A)` has few terms with small coefficients, while `T(G_B)` has many terms with large coefficients. The Tutte polynomial reflects this structural distinction in a way that the energy-based formula does not.

### Difficulty Calibration

The Tutte polynomial provides a topology-aware difficulty measure. Rather than applying a uniform energy threshold for all instances on the same topology, `T(G)` enables calibration that accounts for the specific coupling graph of each instance.

Specific evaluations of `T(G)` provide difficulty-relevant parameters:

- `T(1,1)` — the spanning tree count quantifies how "tree-like" the coupling graph is. Higher counts indicate greater path diversity, which affects sampler mixing time.
- `T(2,1)` — the spanning forest count measures the graph's capacity for independent substructures, relevant to parallel tempering efficiency.

### Hardware-Agnostic Difficulty

Different quantum computers have different native topologies:

- **D-Wave** — Zephyr, Pegasus, Chimera (sparse, structured lattices)
- **IBM** — heavy-hex lattice (planar, low connectivity)
- **IonQ / Quantinuum** — all-to-all or near-all-to-all connectivity

When miners on different hardware propose solutions, each solution is defined on a different coupling graph — the subgraph of that hardware's native topology where the Ising couplers are non-zero. Comparing raw energies across topologies does not yield a meaningful comparison: an energy of -500 on a sparse tree-like graph is trivially achievable, while the same energy on a dense frustrated graph represents significant computational work.

The Tutte polynomial normalizes difficulty across topologies. Given two proposed solutions:

1. Miner A on D-Wave submits a solution on Zephyr subgraph `H₁`
2. Miner B on IBM submits a solution on heavy-hex subgraph `H₂`

The validator computes `T(H₁)` and `T(H₂)` and compares their structural complexity — spanning tree counts, coefficient distributions, polynomial degree — to determine which coupling graph was inherently harder to solve. This comparison is independent of the hardware used to produce the solution.

This enables a fair difficulty comparison even when:

- The coupling graphs have different node counts and edge counts
- The underlying hardware topologies are structurally different
- The solvers use different algorithmic approaches (quantum annealing, gate-based, or classical)

Without `T(G)`, the protocol would require separate difficulty curves calibrated per hardware family — an approach that is fragile and breaks whenever new hardware is introduced. With `T(G)`, difficulty is defined by the graph's combinatorial structure, not by the machine that solved it.

### Verification Asymmetry

Computing `T(G)` is #P-hard in the worst case — no known polynomial-time algorithm exists for arbitrary graphs. However, verifying properties derived from `T(G)` is efficient:

- `T(1,1)` can be checked against the Kirchhoff matrix-tree theorem (determinant of a Laplacian minor) in O(n³).
- Consistency with known graph properties (edge count, node count, connectivity) can be verified in time proportional to the polynomial's degree.
- Comparison against pre-computed tables of known topologies requires O(1) lookup time.

This asymmetry — expensive to compute, cheap to verify — is a desirable property for a proof-of-work mechanism.
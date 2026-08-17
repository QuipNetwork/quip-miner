//! Result revalidation for the coordinator's own miner-gating decision.
//!
//! This is a *derivative* check, not the chain consensus path: the pallet does
//! the authoritative byte-exact validation, so the coordinator is free to use
//! an optimized representation as long as it reaches the same accept/reject
//! decision. Energies are scored in place from wire spin bytes against a
//! position-resolved graph ([`ResolvedTopo`]) — no per-solution decode and no
//! node-id lookup. `quantum_validation::energy_of_solution` is retained only as
//! a debug golden check.

use quantum_validation::{calculate_diversity, select_diverse, MilliValue};
use quip_proto::v1::{ising_problem, IsingProblem, QualityGates, Solution};
use quip_protocol::wire::decode_i32_le;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Upper bound on solutions packed into a submitted proof. Must not exceed the
/// chain pallet's `QuantumPowMaxSolutions` (`BoundedVec` bound, currently 32):
/// a proof carrying more rows fails SCALE bounded-vector decode before the
/// extrinsic dispatches. The coordinator therefore submits only the
/// diverse-selected subset, and the pallet re-selects within it.
pub const MAX_PROOF_SOLUTIONS: usize = 32;

/// Outcome of coordinator-side result revalidation for miner gating.
#[derive(Debug, Clone, PartialEq)]
pub struct Validated {
    /// Best energy among gate-passing rows, or `i64::MAX` if none cleared.
    pub best_energy_milli: i64,
    /// Diversity (milli) of the diverse-selected subset.
    pub diversity_milli: u32,
    /// Count of unique energy-valid solutions after Z2 dedup.
    pub n_valid: u32,
    /// Whether `n_valid` and diversity clear the advertised gates.
    pub accepted: bool,
    /// The diverse-selected *gate-passing* solutions to submit now
    /// (≤ [`MAX_PROOF_SOLUTIONS`]). Empty when no rows cleared the energy gate.
    /// Packed into the immediate-submit proof instead of the raw result rows so
    /// submissions stay within the pallet's bound.
    pub selected_solutions: Vec<Solution>,
    /// Minimum energy over *all* shape-valid rows, ignoring the current gate.
    /// Unlike `best_energy_milli` (which is `i64::MAX` when nothing clears the
    /// gate) this reflects the true best the miner found, so the decay-ratchet
    /// stash can project when the easing gate will admit it.
    pub raw_best_energy_milli: i64,
    /// The deduped rows (≤ [`MAX_PROOF_SOLUTIONS`]) the stash retains for a
    /// future, eased-ceiling submission, chosen regardless of the current gate.
    /// Ordered by ascending energy and pruned so that every prefix of them
    /// still clears the chain's diversity gate.
    pub stash_solutions: Vec<Solution>,
}

/// Outcome of replaying the chain's proof gates over an already-scored row set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProofGates {
    /// Lowest energy among the rows the ceiling admits, or `i64::MAX` if none.
    pub best_energy_milli: i64,
    /// Diversity (milli) of the diverse selection over the admitted rows.
    pub diversity_milli: u32,
    /// Count of rows the ceiling admits.
    pub n_valid: u32,
    /// Whether the admitted rows clear all three chain gates.
    pub accepted: bool,
}

/// The chain's post-energy-filter procedure: select the most separated
/// `min(rows.len(), min_solutions)` rows, then score that selection. Returns
/// the selected indices into `rows` alongside the diversity in milli.
#[expect(
    clippy::indexing_slicing,
    reason = "indices come from select_diverse over the same slice"
)]
fn select_and_score(rows: &[&[i8]], min_solutions: u32) -> (Vec<usize>, u32) {
    let target = rows.len().min(min_solutions.max(1) as usize);
    let selected = select_diverse(rows, target).unwrap_or_else(|e| {
        // Unreachable in principle (every byte was validated during scoring);
        // log if the invariant ever breaks instead of silently accepting the
        // full set.
        tracing::warn!(error = ?e, "select_diverse failed on validated spins; selecting all");
        (0..rows.len()).collect()
    });
    let picked: Vec<&[i8]> = selected.iter().map(|&i| rows[i]).collect();
    let diversity = calculate_diversity(&picked).unwrap_or_else(|e| {
        tracing::warn!(error = ?e, "calculate_diversity failed on validated spins; treating as 0");
        0
    });
    (selected, diversity)
}

/// Replay the chain's proof gates over rows that already carry their scored
/// energy, so a proof held in the stash can be re-checked against the
/// difficulty live now rather than the one it was stashed under.
///
/// The energies are the ones this coordinator computed when the result was
/// validated; the pallet recomputes them from the same problem and reaches the
/// same values, so only the gate parameters have to be fresh. `gates`
/// `min_energy_milli` carries the chain's `max_energy_milli` ceiling.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "n_valid is a solution count; the pallet bound is 32"
)]
pub fn check_proof_gates(solutions: &[Solution], gates: &QualityGates) -> ProofGates {
    let admitted: Vec<&Solution> = solutions
        .iter()
        .filter(|s| s.energy_milli < gates.min_energy_milli)
        .collect();
    let Some(best_energy_milli) = admitted.iter().map(|s| s.energy_milli).min() else {
        return ProofGates {
            best_energy_milli: i64::MAX,
            diversity_milli: 0,
            n_valid: 0,
            accepted: false,
        };
    };
    let rows: Vec<&[i8]> = admitted
        .iter()
        .map(|s| bytemuck::cast_slice::<u8, i8>(&s.spins_bytes))
        .collect();
    let n_valid = rows.len() as u32;
    let (_, diversity_milli) = select_and_score(&rows, gates.min_solutions);
    ProofGates {
        best_energy_milli,
        diversity_milli,
        n_valid,
        accepted: n_valid >= gates.min_solutions && diversity_milli >= gates.min_diversity_milli,
    }
}

/// Build validation gates from the session difficulty target (`SetTarget`), or
/// a permissive default when no target has been advertised.
#[must_use]
pub fn gates_from_target(target: Option<&quip_proto::v1::SetTarget>) -> QualityGates {
    target.map_or(
        QualityGates {
            min_energy_milli: i64::MAX,
            min_diversity_milli: 0,
            min_solutions: 0,
        },
        |t| QualityGates {
            min_energy_milli: t.max_energy_milli,
            min_diversity_milli: t.min_diversity_milli,
            min_solutions: t.min_solutions,
        },
    )
}

/// Coordinator scoring topology, resolved once when a topology is set. Edge
/// endpoints are stored as spin-vector **positions**, so per-solution energy
/// scoring is pure indexing — no node-id lookup. A run constant shared by
/// `Arc`; empty when no topology is set (inline jobs carry their own edges).
#[derive(Debug, Clone, Default)]
pub struct ResolvedTopo {
    edges_pos: Vec<(u32, u32)>,
}

impl ResolvedTopo {
    /// Resolve node-id edges into position-indexed edges once, via a
    /// `node id -> position` map. A well-formed topology references only its
    /// own nodes, so every endpoint resolves; an unknown endpoint (a malformed
    /// topology) maps to position 0 rather than being dropped, which would
    /// misalign edges with the `j` coupling array. It is surfaced elsewhere.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "topology node count is bounded far below u32::MAX"
    )]
    pub fn new(nodes: &[u32], edges_ids: &[(u32, u32)]) -> Self {
        let pos: HashMap<u32, u32> = nodes
            .iter()
            .enumerate()
            .map(|(i, &n)| (n, i as u32))
            .collect();
        let edges_pos = edges_ids
            .iter()
            .map(|&(u, v)| {
                (
                    pos.get(&u).copied().unwrap_or(0),
                    pos.get(&v).copied().unwrap_or(0),
                )
            })
            .collect();
        Self { edges_pos }
    }
}

/// Map one wire spin byte to its `±1` value. `0x01 -> +1`, `0xFF -> -1`; any
/// other byte is invalid (mirrors `wire::decode_spins`).
#[inline]
fn byte_spin(b: u8) -> Option<i64> {
    match b {
        0x01 => Some(1),
        0xFF => Some(-1),
        _ => None,
    }
}

/// Canonical form of a spin row for dedup: a solution `s` and its global flip
/// `-s` map to one key by forcing the first spin to `+1`.
///
/// The protocol's notion of solution *uniqueness* is flip-invariant by design:
/// consensus `symmetric_hamming` scores `s` and `-s` as distance 0, so they are
/// never "diverse" from each other and must count once. This mirrors the v0.2
/// reference (`shared/quantum_proof_of_work.py` `_unique_rows`, which negates
/// each row whose anchor spin is `-1`). Note this is *not* about energy: at
/// `h != 0` the field term flips sign under `s -> -s`, so `E(-s) != E(s)` and
/// twin pairs that are both energy-valid are vanishingly rare — canonicalization
/// is a near-no-op there. It bites at `h = 0` (`parity_shared`), where flip-
/// symmetric ground states genuinely appear and would otherwise double-count.
fn canonicalize_spins(spins: &[i8]) -> Vec<i8> {
    if spins.first() == Some(&-1) {
        spins.iter().map(|&s| -s).collect()
    } else {
        spins.to_vec()
    }
}

/// Energy of one solution read directly from its wire spin bytes against
/// position-resolved edges. No per-solution allocation and no node lookup.
/// Returns `None` on a wrong-length or invalid-byte solution (mirrors the
/// consensus shape/spin checks). Golden-checked against
/// `quantum_validation::energy_of_solution` in debug builds.
fn energy_in_place(
    spins: &[u8],
    h_milli: &[MilliValue],
    edges_pos: &[(u32, u32)],
    j_milli: &[MilliValue],
) -> Option<i64> {
    if spins.len() != h_milli.len() {
        return None;
    }
    let mut energy: i64 = 0;
    for (&field, &sb) in h_milli.iter().zip(spins) {
        energy += i64::from(field) * byte_spin(sb)?;
    }
    for (&(u, v), &coupling) in edges_pos.iter().zip(j_milli) {
        let su = byte_spin(*spins.get(u as usize)?)?;
        let sv = byte_spin(*spins.get(v as usize)?)?;
        energy += i64::from(coupling) * su * sv;
    }
    Some(energy)
}

/// Recompute energies, apply quality gates, return the validation summary.
///
/// A solution is energy-valid when `energy < gates.min_energy_milli` (strict);
/// the wire gate's `min_energy_milli` carries the chain's `max_energy_milli`
/// ceiling (see `derive_pow_job`). The set is accepted when
/// `n_valid >= min_solutions` and `diversity_milli >= min_diversity_milli`.
///
/// `topo` is the run-resolved position-indexed graph, used for topology-hash
/// jobs; inline `EdgeList` jobs carry their own (already position-indexed)
/// edges.
#[expect(
    clippy::indexing_slicing,
    reason = "indices drawn from enumerate/len of the same scored buffers"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "n_valid is solution count; pallet bound is 32, well below u32::MAX"
)]
pub fn validate_result(
    problem: &IsingProblem,
    solutions: &[Solution],
    gates: &QualityGates,
    topo: &ResolvedTopo,
) -> Validated {
    // The problem's h/j are coordinator-authored (encoded with `encode_i32_le`
    // when the job was built). A decode failure here therefore means the
    // coordinator's own encoding is corrupt — surface it and reject, rather
    // than silently scoring against a truncated (empty) problem, which would
    // drop coupling/field terms and yield a plausible-but-wrong accept/reject.
    let (h_milli, j_milli) = match (
        decode_i32_le(&problem.h_milli_le32),
        decode_i32_le(&problem.j_milli_le32),
    ) {
        (Ok(h), Ok(j)) => (h, j),
        (h, j) => {
            tracing::error!(
                h_err = %crate::logging::display_option(h.err()),
                j_err = %crate::logging::display_option(j.err()),
                "validate_result: malformed problem wire bytes (not i32-aligned); rejecting result"
            );
            return Validated {
                best_energy_milli: i64::MAX,
                diversity_milli: 0,
                n_valid: 0,
                accepted: false,
                selected_solutions: Vec::new(),
                raw_best_energy_milli: i64::MAX,
                stash_solutions: Vec::new(),
            };
        }
    };

    // Position-resolved edges: inline jobs carry their own (endpoints already
    // spin indices); topology-hash jobs borrow the run constant.
    let inline_edges: Vec<(u32, u32)>;
    let edges_pos: &[(u32, u32)] = match &problem.graph {
        Some(ising_problem::Graph::Edges(e)) => {
            inline_edges = e.u.iter().zip(&e.v).map(|(&u, &v)| (u, v)).collect();
            &inline_edges
        }
        Some(ising_problem::Graph::TopologyHash(_)) => &topo.edges_pos,
        None => &[],
    };

    // Energy gate: score each solution straight from its wire bytes, in
    // parallel. `filter_map` + `unzip` preserve order and drop invalid rows.
    // Survivors keep a borrow of their spin bytes (no copy) for the diversity
    // pass below.
    let (byte_rows, energies): (Vec<&[u8]>, Vec<i64>) = solutions
        .par_iter()
        .filter_map(|sol| {
            let e = energy_in_place(&sol.spins_bytes, &h_milli, edges_pos, &j_milli)?;
            Some((sol.spins_bytes.as_slice(), e))
        })
        .unzip();

    debug_assert_energies_match(&byte_rows, &h_milli, edges_pos, &j_milli, &energies);

    let energy_valid_indices: Vec<usize> = energies
        .iter()
        .enumerate()
        .filter_map(|(i, &e)| (e < gates.min_energy_milli).then_some(i))
        .collect();

    // Dedup before counting/diversity: exact duplicates and Z2-flip twins
    // (a solution and its global spin flip) are the same physical state and
    // must not be double counted or inflate the diversity score — most
    // visible at h=0 (parity_shared), where flip-symmetric ground states are
    // common. Sequential (not rayon): this is a cheap post-filter over the
    // already-scored valid indices, not the scoring loop. Keeps the first
    // occurrence, so it stays deterministic given the fixed solution order.
    let mut seen_canonical: HashSet<Vec<i8>> = HashSet::new();
    let unique_valid_indices: Vec<usize> = energy_valid_indices
        .into_iter()
        .filter(|&i| {
            let spins = bytemuck::cast_slice::<u8, i8>(byte_rows[i]);
            seen_canonical.insert(canonicalize_spins(spins))
        })
        .collect();
    let n_valid = unique_valid_indices.len() as u32;

    let (best_energy_milli, diversity_milli, selected_solutions) =
        if unique_valid_indices.is_empty() {
            (i64::MAX, 0u32, Vec::new())
        } else {
            // Diversity reads the valid spin vectors as `&[i8]` reinterpreted from
            // their wire bytes (0x01/0xFF -> +1/-1) — a zero-copy view, sound
            // because every byte was validated during scoring.
            let energy_valid: Vec<&[i8]> = unique_valid_indices
                .iter()
                .map(|&i| bytemuck::cast_slice::<u8, i8>(byte_rows[i]))
                .collect();
            let (selected, diversity) = select_and_score(&energy_valid, gates.min_solutions);
            let best = selected
                .iter()
                .map(|&i| energies[unique_valid_indices[i]])
                .min()
                .unwrap_or(i64::MAX);
            // Materialize the diverse subset (capped at the pallet bound) as the
            // exact rows to submit, so proofs never exceed MAX_PROOF_SOLUTIONS.
            let selected_solutions: Vec<Solution> = selected
                .iter()
                .take(MAX_PROOF_SOLUTIONS)
                .map(|&i| {
                    let row = unique_valid_indices[i];
                    Solution {
                        spins_bytes: byte_rows[row].to_vec(),
                        energy_milli: energies[row],
                    }
                })
                .collect();
            (best, diversity, selected_solutions)
        };

    let accepted = n_valid >= gates.min_solutions && diversity_milli >= gates.min_diversity_milli;

    // Gate-agnostic view for the decay-ratchet stash: the true best energy the
    // miner found, and the rows that stay submittable as the difficulty eases.
    // Both ignore the current (harder) gate, which nothing here has to clear.
    let raw_best_energy_milli = energies.iter().copied().min().unwrap_or(i64::MAX);
    let stash_solutions = stash_rows(&byte_rows, &energies, gates);

    Validated {
        best_energy_milli,
        diversity_milli,
        n_valid,
        accepted,
        selected_solutions,
        raw_best_energy_milli,
        stash_solutions,
    }
}

/// Unique candidate rows examined while filling the stash. Each candidate costs
/// a diversity pass over the rows already held, so an unbounded scan of a large
/// result would dominate validation; the chain accepts at most
/// [`MAX_PROOF_SOLUTIONS`] rows in a proof regardless.
const STASH_SCAN_LIMIT: usize = 4 * MAX_PROOF_SOLUTIONS;

/// Pick the rows the decay-ratchet stash holds for a later, eased-ceiling
/// submission.
///
/// The chain filters a submitted proof to `energy < max_energy_milli` at the
/// inclusion block and scores diversity over only the survivors, so the subset
/// it judges is always a lowest-energy prefix of what was submitted. Rows are
/// therefore admitted in ascending energy order only while each resulting
/// prefix still clears `min_diversity_milli` under the chain's own selection; a
/// row sitting too close to one already held is skipped in favour of a more
/// distant, higher-energy row. Taking the globally lowest energies instead
/// packs the prefix with near-twins of the optimum and collapses its diversity.
///
/// The single best row is always kept, because the decay projection rides on
/// it. Scoring it alone yields 0 diversity, but a one-row prefix can never
/// reach the diversity gate anyway: the chain rejects a difficulty carrying
/// `min_diversity_milli > 0` with `min_solutions < 2`.
#[expect(
    clippy::indexing_slicing,
    reason = "indices drawn from enumerate/len of the same scored buffers"
)]
fn stash_rows(byte_rows: &[&[u8]], energies: &[i64], gates: &QualityGates) -> Vec<Solution> {
    let mut order: Vec<usize> = (0..byte_rows.len()).collect();
    order.sort_by_key(|&i| energies[i]);

    let mut seen: HashSet<Vec<i8>> = HashSet::new();
    let mut examined = 0usize;
    let mut chosen: Vec<usize> = Vec::new();
    for i in order {
        if chosen.len() == MAX_PROOF_SOLUTIONS || examined == STASH_SCAN_LIMIT {
            break;
        }
        let spins = bytemuck::cast_slice::<u8, i8>(byte_rows[i]);
        if !seen.insert(canonicalize_spins(spins)) {
            continue;
        }
        examined += 1;
        chosen.push(i);
        if chosen.len() < 2 {
            continue;
        }
        let prefix: Vec<&[i8]> = chosen
            .iter()
            .map(|&k| bytemuck::cast_slice::<u8, i8>(byte_rows[k]))
            .collect();
        let (_, diversity) = select_and_score(&prefix, gates.min_solutions);
        if diversity < gates.min_diversity_milli {
            let _ = chosen.pop();
        }
    }

    chosen
        .into_iter()
        .map(|i| Solution {
            spins_bytes: byte_rows[i].to_vec(),
            energy_milli: energies[i],
        })
        .collect()
}

/// Strict less-than tie-break: first accepted holds on equal energy.
#[must_use]
pub fn beats_current(candidate_milli: i64, current_best_milli: Option<i64>) -> bool {
    match current_best_milli {
        None => true,
        Some(cur) => candidate_milli < cur,
    }
}

/// Debug-only golden check: the in-place scorer must equal the consensus
/// `energy_of_solution`. An identity node map makes the consensus scorer treat
/// edge endpoints as the same positions `energy_in_place` used, so the two
/// agree for both inline and topology-hash graphs.
#[cfg(debug_assertions)]
fn debug_assert_energies_match(
    byte_rows: &[&[u8]],
    h_milli: &[MilliValue],
    edges_pos: &[(u32, u32)],
    j_milli: &[MilliValue],
    energies: &[i64],
) {
    use quantum_validation::energy_of_solution;
    use quip_protocol::wire::decode_spins;

    // `energy_of_solution` resolves every edge endpoint by a linear scan of
    // `nodes` (`position_of_node`), so this golden re-check costs
    // O(edges * nodes) per solution. That is a cheap sanity net on small
    // problems but pathological on large topologies — a 4577-node Advantage2
    // graph is ~20 billion unoptimized ops per result, which stalls a debug
    // `drive` run indefinitely (quip-w5p.14). Skip it past a modest size; the
    // release path never runs it at all, and small-problem coverage still
    // catches an `energy_in_place` divergence (the formula is size-invariant).
    const MAX_GOLDEN_EDGE_NODE_PRODUCT: usize = 1_000_000;
    if edges_pos.len().saturating_mul(h_milli.len()) > MAX_GOLDEN_EDGE_NODE_PRODUCT {
        return;
    }

    let nodes: Vec<u32> = (0..h_milli.len())
        .map(|i| u32::try_from(i).unwrap_or(u32::MAX))
        .collect();
    for (bytes, &e) in byte_rows.iter().zip(energies) {
        let Ok(spins) = decode_spins(bytes) else {
            continue;
        };
        if let Ok(g) = energy_of_solution(&spins, h_milli, edges_pos, j_milli, &nodes) {
            debug_assert_eq!(g, e, "energy_in_place {e} diverged from consensus {g}");
        }
    }
}

#[cfg(not(debug_assertions))]
fn debug_assert_energies_match(
    _byte_rows: &[&[u8]],
    _h_milli: &[MilliValue],
    _edges_pos: &[(u32, u32)],
    _j_milli: &[MilliValue],
    _energies: &[i64],
) {
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_protocol::wire::encode_i32_le;
    use quip_protocol::wire::encode_spins;

    #[test]
    fn golden_energy_and_strict_tiebreak() {
        // E = 1*1 + (-0.5)*(-1) + 2.0*1*(-1) = 1 + 0.5 - 2 = -0.5 → -500 milli
        let problem = IsingProblem {
            graph: Some(ising_problem::Graph::Edges(quip_proto::v1::EdgeList {
                u: vec![0],
                v: vec![1],
            })),
            h_milli_le32: encode_i32_le(&[1000, -500]),
            j_milli_le32: encode_i32_le(&[2000]),
            num_reads: 1,
            num_sweeps: 0,
            anneal_time_us: 0,
        };
        let sol = Solution {
            spins_bytes: encode_spins(&[1, -1]),
            energy_milli: -500,
        };
        let gates = QualityGates {
            min_energy_milli: 0, // accept anything < 0
            min_diversity_milli: 0,
            min_solutions: 1,
        };
        let v = validate_result(&problem, &[sol], &gates, &ResolvedTopo::default());
        assert_eq!(v.best_energy_milli, -500);
        assert!(v.accepted);
        assert_eq!(v.n_valid, 1);

        assert!(beats_current(-501, Some(-500)));
        assert!(!beats_current(-500, Some(-500))); // strict <
        assert!(!beats_current(-499, Some(-500)));
        assert!(beats_current(-1, None));
    }

    #[test]
    fn stash_view_retains_raw_best_when_gate_not_cleared() {
        // E([1,-1]) = 1000 - (-500) - 2000 = -500 milli. Gate demands < -1000,
        // so it does NOT clear. F7: the gate-passing best is i64::MAX, but the
        // raw best (-500) and the candidate rows survive so the decay-ratchet
        // stash can retain what will win once the gate eases. F1: the submit
        // subset is capped at MAX_PROOF_SOLUTIONS.
        let problem = IsingProblem {
            graph: Some(ising_problem::Graph::Edges(quip_proto::v1::EdgeList {
                u: vec![0],
                v: vec![1],
            })),
            h_milli_le32: encode_i32_le(&[1000, -500]),
            j_milli_le32: encode_i32_le(&[2000]),
            num_reads: 1,
            num_sweeps: 0,
            anneal_time_us: 0,
        };
        let sol = Solution {
            spins_bytes: encode_spins(&[1, -1]),
            energy_milli: -500,
        };
        let gates = QualityGates {
            min_energy_milli: -1000,
            min_diversity_milli: 0,
            min_solutions: 1,
        };
        let v = validate_result(&problem, &[sol], &gates, &ResolvedTopo::default());
        assert!(!v.accepted);
        assert_eq!(v.n_valid, 0);
        assert_eq!(v.best_energy_milli, i64::MAX); // nothing cleared the gate
        assert_eq!(v.raw_best_energy_milli, -500); // true best, for the stash
        assert_eq!(v.stash_solutions.len(), 1); // candidate retained
        assert!(v.selected_solutions.is_empty()); // nothing to submit now
        assert!(v.stash_solutions.len() <= MAX_PROOF_SOLUTIONS);
    }

    /// Regression for quip-w5p.14: a large topology must not stall the debug
    /// golden re-check. `energy_of_solution` resolves edges by a linear
    /// `position_of_node` scan (O(edges*nodes) per solution); un-capped, this
    /// test hangs for minutes in a debug build. The size cap in
    /// `debug_assert_energies_match` skips it, so `validate_result` returns fast.
    #[test]
    fn large_topology_does_not_stall_debug_golden_check() {
        let num_nodes = 5000usize;
        let num_edges = 30_000usize; // edges*nodes = 150M, far above the 1M cap
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "test fixture: i % 3 is 0..=2"
        )]
        let fields: Vec<i32> = (0..num_nodes)
            .map(|idx| ((idx % 3) as i32 - 1) * 1000)
            .collect();
        let mut edge_u = Vec::with_capacity(num_edges);
        let mut edge_v = Vec::with_capacity(num_edges);
        let mut couplings = Vec::with_capacity(num_edges);
        for edge_idx in 0..num_edges {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "test fixture: node indices < 5000 fit u32"
            )]
            {
                edge_u.push((edge_idx % num_nodes) as u32);
                edge_v.push(((edge_idx * 7 + 1) % num_nodes) as u32);
            }
            couplings.push(if edge_idx % 2 == 0 { 1000 } else { -1000 });
        }
        let problem = IsingProblem {
            graph: Some(ising_problem::Graph::Edges(quip_proto::v1::EdgeList {
                u: edge_u,
                v: edge_v,
            })),
            h_milli_le32: encode_i32_le(&fields),
            j_milli_le32: encode_i32_le(&couplings),
            num_reads: 0,
            num_sweeps: 0,
            anneal_time_us: 0,
        };
        let spins: Vec<i8> = (0..num_nodes)
            .map(|idx| if idx % 2 == 0 { 1 } else { -1 })
            .collect();
        let sols: Vec<Solution> = (0..4)
            .map(|_| Solution {
                spins_bytes: encode_spins(&spins),
                energy_milli: 0,
            })
            .collect();
        let gates = QualityGates {
            min_energy_milli: i64::MAX, // accept all
            min_diversity_milli: 0,
            min_solutions: 0,
        };
        let vd = validate_result(&problem, &sols, &gates, &ResolvedTopo::default());
        // Four identical solutions dedup (Z2-canonical) to one unique valid row;
        // the point is that this returned at all.
        assert_eq!(vd.n_valid, 1);
    }

    #[test]
    fn rejects_when_below_min_solutions() {
        let problem = IsingProblem {
            graph: None,
            h_milli_le32: encode_i32_le(&[1000]),
            j_milli_le32: vec![],
            num_reads: 1,
            num_sweeps: 0,
            anneal_time_us: 0,
        };
        let sol = Solution {
            spins_bytes: encode_spins(&[1]),
            energy_milli: 1000,
        };
        let gates = QualityGates {
            min_energy_milli: 0, // 1000 is not < 0
            min_diversity_milli: 0,
            min_solutions: 1,
        };
        let v = validate_result(&problem, &[sol], &gates, &ResolvedTopo::default());
        assert!(!v.accepted);
        assert_eq!(v.n_valid, 0);
    }

    #[test]
    fn topology_edges_use_node_ids_not_spin_indices() {
        // nodes [10, 20]; edge (10, 20); h/j aligned to node positions.
        // spins [1, -1] → E = 1000*1 + (-500)*(-1) + 2000*1*(-1) = -500
        let problem = IsingProblem {
            graph: Some(ising_problem::Graph::TopologyHash(vec![0u8; 32])),
            h_milli_le32: encode_i32_le(&[1000, -500]),
            j_milli_le32: encode_i32_le(&[2000]),
            num_reads: 1,
            num_sweeps: 0,
            anneal_time_us: 0,
        };
        let sol = Solution {
            spins_bytes: encode_spins(&[1, -1]),
            energy_milli: -500,
        };
        let gates = QualityGates {
            min_energy_milli: 0,
            min_diversity_milli: 0,
            min_solutions: 1,
        };
        let topo = ResolvedTopo::new(&[10, 20], &[(10, 20)]);
        let v = validate_result(&problem, &[sol], &gates, &topo);
        assert_eq!(v.best_energy_milli, -500);
        assert!(v.accepted);
    }

    #[test]
    fn diversity_uses_quantum_validation_rounding() {
        // Two fully opposite solutions on 3 spins: hamming=3, symmetric=0
        // under flip → diversity 0. Two solutions that differ in 1 of 2:
        // dist=1, pairs=1, n=2 → round(1000*1/(1*2)) = 500.
        let problem = IsingProblem {
            graph: None,
            h_milli_le32: encode_i32_le(&[0, 0]),
            j_milli_le32: vec![],
            num_reads: 2,
            num_sweeps: 0,
            anneal_time_us: 0,
        };
        let sols = [
            Solution {
                spins_bytes: encode_spins(&[1, 1]),
                energy_milli: 0,
            },
            Solution {
                spins_bytes: encode_spins(&[1, -1]),
                energy_milli: 0,
            },
        ];
        let gates = QualityGates {
            min_energy_milli: 1, // 0 < 1 → both valid
            min_diversity_milli: 0,
            min_solutions: 2,
        };
        let v = validate_result(&problem, &sols, &gates, &ResolvedTopo::default());
        assert_eq!(v.n_valid, 2);
        assert_eq!(v.diversity_milli, 500);
        assert!(v.accepted);
    }

    #[test]
    fn dedup_collapses_exact_duplicate_solutions() {
        let problem = IsingProblem {
            graph: None,
            h_milli_le32: encode_i32_le(&[0, 0]),
            j_milli_le32: vec![],
            num_reads: 2,
            num_sweeps: 0,
            anneal_time_us: 0,
        };
        let sols = [
            Solution {
                spins_bytes: encode_spins(&[1, -1]),
                energy_milli: 0,
            },
            Solution {
                spins_bytes: encode_spins(&[1, -1]),
                energy_milli: 0,
            },
        ];
        let gates = QualityGates {
            min_energy_milli: 1, // 0 < 1 → both energy-valid
            min_diversity_milli: 0,
            min_solutions: 1,
        };
        let v = validate_result(&problem, &sols, &gates, &ResolvedTopo::default());
        assert_eq!(v.n_valid, 1);
    }

    #[test]
    fn dedup_collapses_z2_flip_twin_at_h_zero() {
        // h = 0: a solution and its global flip are energy-equal and are the
        // same physical state, so they must collapse to one.
        let problem = IsingProblem {
            graph: None,
            h_milli_le32: encode_i32_le(&[0, 0, 0]),
            j_milli_le32: vec![],
            num_reads: 2,
            num_sweeps: 0,
            anneal_time_us: 0,
        };
        let sols = [
            Solution {
                spins_bytes: encode_spins(&[1, -1, 1]),
                energy_milli: 0,
            },
            Solution {
                spins_bytes: encode_spins(&[-1, 1, -1]),
                energy_milli: 0,
            },
        ];
        let gates = QualityGates {
            min_energy_milli: 1,
            min_diversity_milli: 0,
            min_solutions: 1,
        };
        let v = validate_result(&problem, &sols, &gates, &ResolvedTopo::default());
        assert_eq!(v.n_valid, 1);
    }

    /// Spins that carry no field, so they set distance without moving energy.
    const SHAPE_SPINS: usize = 32;
    /// Spins carrying powers-of-two fields, which set a row's energy level.
    const LEVEL_SPINS: u32 = 4;
    /// Chain-static gates used by the clustered-optimum fixture below.
    const FIXTURE_MIN_SOLUTIONS: u32 = 3;
    const FIXTURE_MIN_DIVERSITY: u32 = 300;

    fn level_fixture_problem() -> IsingProblem {
        let mut h = vec![0i32; SHAPE_SPINS];
        h.extend_from_slice(&[1000, 2000, 4000, 8000]);
        IsingProblem {
            graph: None,
            h_milli_le32: encode_i32_le(&h),
            j_milli_le32: vec![],
            num_reads: 0,
            num_sweeps: 0,
            anneal_time_us: 0,
        }
    }

    /// Row `shape` of the order-32 Sylvester-Hadamard matrix, whose distinct
    /// rows sit at symmetric Hamming distance 16 — the maximum for 32 spins.
    /// `flips` negates that many leading spins, which parks a row next to its
    /// base instead. The trailing level spins give the row energy
    /// `15000 - 2000 * level`.
    fn level_row(shape: usize, flips: usize, level: u32) -> Solution {
        let mut spins: Vec<i8> = (0..SHAPE_SPINS)
            .map(|col| {
                if (shape & col).count_ones().is_multiple_of(2) {
                    1
                } else {
                    -1
                }
            })
            .collect();
        for s in spins.iter_mut().take(flips) {
            *s = -*s;
        }
        for bit in 0..LEVEL_SPINS {
            spins.push(if (level >> bit) & 1 == 1 { -1 } else { 1 });
        }
        Solution {
            spins_bytes: encode_spins(&spins),
            energy_milli: 15_000 - 2_000 * i64::from(level),
        }
    }

    /// Four near-identical rows holding the four lowest energies, with five
    /// mutually distant rows above them. This is the shape that made the chain
    /// reject a proof the coordinator had cleared locally: the whole set is
    /// diverse, every low-energy prefix of it is not.
    fn clustered_optimum_rows() -> Vec<Solution> {
        vec![
            level_row(0, 0, 15),
            level_row(0, 1, 14),
            level_row(0, 2, 13),
            level_row(0, 3, 12),
            level_row(1, 0, 5),
            level_row(2, 0, 4),
            level_row(4, 0, 3),
            level_row(8, 0, 2),
            level_row(16, 0, 1),
        ]
    }

    /// The chain's own procedure written out against the consensus primitives:
    /// keep the rows under `ceiling`, select `min(count, min_solutions)` of them
    /// by maximum separation, then score that selection. Returns
    /// `(valid_solution_count, diversity_milli)` — the two values the pallet
    /// gates on. This is the oracle the coordinator's row choice must satisfy.
    #[expect(
        clippy::indexing_slicing,
        reason = "indices come from select_diverse over the same slice"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "fixture row counts are single digits"
    )]
    fn pallet_gates(rows: &[Solution], ceiling: i64, min_solutions: u32) -> (u32, u32) {
        let valid: Vec<&[i8]> = rows
            .iter()
            .filter(|s| s.energy_milli < ceiling)
            .map(|s| bytemuck::cast_slice::<u8, i8>(&s.spins_bytes))
            .collect();
        if valid.is_empty() {
            return (0, 0);
        }
        let target = valid.len().min(min_solutions.max(1) as usize);
        let selected = select_diverse(&valid, target).expect("fixture spins are valid");
        let picked: Vec<&[i8]> = selected.iter().map(|&i| valid[i]).collect();
        let diversity = calculate_diversity(&picked).expect("fixture spins are valid");
        (valid.len() as u32, diversity)
    }

    /// Every ceiling that admits a different subset of `rows`: one just below
    /// and one just above each row energy.
    fn ceiling_sweep(rows: &[Solution]) -> Vec<i64> {
        let mut ceilings: Vec<i64> = rows
            .iter()
            .flat_map(|s| [s.energy_milli, s.energy_milli + 1])
            .collect();
        ceilings.sort_unstable();
        ceilings.dedup();
        ceilings
    }

    /// The gates the fixture rows are stashed under: a ceiling far below every
    /// row, so nothing is submittable yet and the whole result goes to the
    /// decay-ratchet stash.
    fn fixture_gates() -> QualityGates {
        QualityGates {
            min_energy_milli: -100_000,
            min_diversity_milli: FIXTURE_MIN_DIVERSITY,
            min_solutions: FIXTURE_MIN_SOLUTIONS,
        }
    }

    #[test]
    fn stashed_rows_keep_every_low_energy_prefix_diverse() {
        let problem = level_fixture_problem();
        let rows = clustered_optimum_rows();
        let gates = fixture_gates();

        // The failure shape: sorting purely by energy puts the four clustered
        // rows first, so a ceiling admitting only those collapses diversity
        // even though the full set scores well above the gate.
        let mut energy_sorted = rows.clone();
        energy_sorted.sort_by_key(|s| s.energy_milli);
        let (all_valid, all_diversity) =
            pallet_gates(&energy_sorted, i64::MAX, gates.min_solutions);
        assert_eq!(all_valid, 9);
        assert!(
            all_diversity >= gates.min_diversity_milli,
            "fixture must clear the gate over the whole set, scored {all_diversity}"
        );
        let energy_sorted_failures: Vec<(i64, u32)> = ceiling_sweep(&energy_sorted)
            .into_iter()
            .filter_map(|ceiling| {
                let (n_valid, diversity) =
                    pallet_gates(&energy_sorted, ceiling, gates.min_solutions);
                (n_valid >= gates.min_solutions && diversity < gates.min_diversity_milli)
                    .then_some((ceiling, diversity))
            })
            .collect();
        assert!(
            !energy_sorted_failures.is_empty(),
            "fixture must reproduce the rejection: pure energy order should fail some ceiling"
        );

        // The stash rows must clear the same sweep at every ceiling that admits
        // enough rows to reach the diversity gate at all.
        let v = validate_result(&problem, &rows, &gates, &ResolvedTopo::default());
        assert!(!v.accepted, "fixture stashes rather than submits");
        for ceiling in ceiling_sweep(&v.stash_solutions) {
            let (n_valid, diversity) =
                pallet_gates(&v.stash_solutions, ceiling, gates.min_solutions);
            if n_valid >= gates.min_solutions {
                assert!(
                    diversity >= gates.min_diversity_milli,
                    "ceiling {ceiling} admits {n_valid} stashed rows scoring {diversity}, \
                     below the {} gate",
                    gates.min_diversity_milli
                );
            }
        }
    }

    #[test]
    fn stash_keeps_the_best_row_and_drops_only_its_near_twins() {
        let problem = level_fixture_problem();
        let rows = clustered_optimum_rows();
        let v = validate_result(&problem, &rows, &fixture_gates(), &ResolvedTopo::default());

        assert_eq!(v.raw_best_energy_milli, -15_000);
        assert_eq!(
            v.stash_solutions.first().map(|s| s.energy_milli),
            Some(-15_000),
            "the decay projection rides on the best row, which must always be stashed"
        );
        // The three near-twins of the best row are the only rows dropped.
        let kept: Vec<i64> = v.stash_solutions.iter().map(|s| s.energy_milli).collect();
        assert_eq!(kept, vec![-15_000, 5_000, 7_000, 9_000, 11_000, 13_000]);
    }

    #[test]
    fn stash_falls_back_to_energy_order_when_no_diversity_is_demanded() {
        let problem = level_fixture_problem();
        let rows = clustered_optimum_rows();
        let gates = QualityGates {
            min_energy_milli: -100_000,
            min_diversity_milli: 0,
            min_solutions: FIXTURE_MIN_SOLUTIONS,
        };
        let v = validate_result(&problem, &rows, &gates, &ResolvedTopo::default());
        let kept: Vec<i64> = v.stash_solutions.iter().map(|s| s.energy_milli).collect();
        assert_eq!(
            kept,
            vec![-15_000, -13_000, -11_000, -9_000, 5_000, 7_000, 9_000, 11_000, 13_000],
            "with no diversity gate every deduped row is worth holding"
        );
    }

    #[test]
    fn proof_gate_check_tracks_the_live_ceiling() {
        let problem = level_fixture_problem();
        let rows = clustered_optimum_rows();
        let stashed = validate_result(&problem, &rows, &fixture_gates(), &ResolvedTopo::default())
            .stash_solutions;

        // Too early: the ceiling admits only the best row, short of the
        // solution count the chain demands.
        let early = check_proof_gates(
            &stashed,
            &QualityGates {
                min_energy_milli: 0,
                min_diversity_milli: FIXTURE_MIN_DIVERSITY,
                min_solutions: FIXTURE_MIN_SOLUTIONS,
            },
        );
        assert_eq!(early.n_valid, 1);
        assert!(!early.accepted);

        // Eased far enough to admit the whole stash: every gate clears.
        let ready = check_proof_gates(
            &stashed,
            &QualityGates {
                min_energy_milli: 20_000,
                min_diversity_milli: FIXTURE_MIN_DIVERSITY,
                min_solutions: FIXTURE_MIN_SOLUTIONS,
            },
        );
        assert_eq!(ready.n_valid, 6);
        assert_eq!(ready.best_energy_milli, -15_000);
        assert!(ready.accepted);

        // The check must agree with the chain's own procedure at every ceiling.
        for ceiling in ceiling_sweep(&stashed) {
            let gates = QualityGates {
                min_energy_milli: ceiling,
                min_diversity_milli: FIXTURE_MIN_DIVERSITY,
                min_solutions: FIXTURE_MIN_SOLUTIONS,
            };
            let (n_valid, diversity) = pallet_gates(&stashed, ceiling, gates.min_solutions);
            let got = check_proof_gates(&stashed, &gates);
            assert_eq!(got.n_valid, n_valid, "ceiling {ceiling}");
            assert_eq!(got.diversity_milli, diversity, "ceiling {ceiling}");
        }
    }

    #[test]
    fn proof_gate_check_rejects_a_diversity_collapse() {
        // The rows a pure energy sort would have stashed, re-checked at the
        // ceiling that admits exactly the clustered prefix: the count gate
        // clears and the diversity gate does not — the chain's
        // `InsufficientDiversity` rejection.
        let mut energy_sorted = clustered_optimum_rows();
        energy_sorted.sort_by_key(|s| s.energy_milli);
        let got = check_proof_gates(
            &energy_sorted,
            &QualityGates {
                min_energy_milli: -8_000,
                min_diversity_milli: FIXTURE_MIN_DIVERSITY,
                min_solutions: FIXTURE_MIN_SOLUTIONS,
            },
        );
        assert_eq!(got.n_valid, 4);
        assert!(got.diversity_milli < FIXTURE_MIN_DIVERSITY);
        assert!(!got.accepted);
    }

    #[test]
    fn dedup_does_not_collapse_genuinely_distinct_solutions() {
        let problem = IsingProblem {
            graph: None,
            h_milli_le32: encode_i32_le(&[0, 0]),
            j_milli_le32: vec![],
            num_reads: 2,
            num_sweeps: 0,
            anneal_time_us: 0,
        };
        let sols = [
            Solution {
                spins_bytes: encode_spins(&[1, 1]),
                energy_milli: 0,
            },
            Solution {
                spins_bytes: encode_spins(&[1, -1]),
                energy_milli: 0,
            },
        ];
        let gates = QualityGates {
            min_energy_milli: 1,
            min_diversity_milli: 0,
            min_solutions: 2,
        };
        let v = validate_result(&problem, &sols, &gates, &ResolvedTopo::default());
        assert_eq!(v.n_valid, 2);
    }
}

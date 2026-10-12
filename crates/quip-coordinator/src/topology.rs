//! Session-cached topology and the chain-canonical topology hash.
//!
//! The hash matches `pallet_quantum_pow::topology::hash_topology`:
//! `blake2_256(SCALE.encode((sorted_nodes, canonical_sorted_edges,
//! allowed_h.canonical_bytes(), allowed_j.canonical_bytes(),
//! allowed_spin.canonical_bytes())))`.

use parity_scale_codec::Encode;
use quantum_validation::{AllowedValueSpec, MilliValue};
use sp_core::hashing::blake2_256;

/// Default binary spin set (`±MILLI_SCALE`) used when a caller has no spin spec.
pub const DEFAULT_SPIN_SET: [MilliValue; 2] = [-1000, 1000];

/// A hardware (or logical) graph identified by a deterministic hash.
#[derive(Debug, Clone)]
pub struct Topology {
    /// 32-byte chain-canonical topology hash.
    pub hash: Vec<u8>,
    /// Node ids in the graph (order preserved for position mapping).
    pub nodes: Vec<u32>,
    /// Parallel edge endpoint arrays `(u, v)`.
    pub edges: (Vec<u32>, Vec<u32>),
    /// Allowed h-field values (milli), advertised so the miner can pick its
    /// adapt difficulty band.
    pub allowed_h: Vec<i32>,
    /// Allowed J-coupling values (milli). A lease miner draws couplings from this set.
    pub allowed_j: Vec<i32>,
}

impl Topology {
    /// Build a topology using the given allowed-value *sets* (Set-variant specs).
    #[must_use]
    pub fn from_nodes_edges(
        nodes: Vec<u32>,
        edges: Vec<(u32, u32)>,
        allowed_h: &[MilliValue],
        allowed_j: &[MilliValue],
        allowed_spin: &[MilliValue],
    ) -> Self {
        let hash = topology_hash_sets(&nodes, &edges, allowed_h, allowed_j, allowed_spin).to_vec();
        let (u, v): (Vec<u32>, Vec<u32>) = edges.into_iter().unzip();
        Self {
            hash,
            nodes,
            edges: (u, v),
            allowed_h: allowed_h.to_vec(),
            allowed_j: allowed_j.to_vec(),
        }
    }

    /// Rebuild edges as `(u, v)` pairs from the parallel endpoint arrays.
    #[must_use]
    pub fn edge_pairs(&self) -> Vec<(u32, u32)> {
        self.edges
            .0
            .iter()
            .zip(&self.edges.1)
            .map(|(&u, &v)| (u, v))
            .collect()
    }

    /// Convert to the wire `Topology` message sent at session handshake.
    #[must_use]
    pub fn to_proto(&self) -> quip_proto::v1::Topology {
        quip_proto::v1::Topology {
            hash: self.hash.clone(),
            nodes: self.nodes.clone(),
            edges: Some(quip_proto::v1::EdgeList {
                u: self.edges.0.clone(),
                v: self.edges.1.clone(),
            }),
            allowed_h_milli: self.allowed_h.clone(),
            allowed_j_milli: self.allowed_j.clone(),
        }
    }
}

/// D-Wave's graph-only fingerprint. This is not a chain topology hash.
#[must_use]
pub(crate) fn native_topology_hash(nodes: &[u32], edges: &[(u32, u32)]) -> Option<[u8; 32]> {
    let n_nodes = u32::try_from(nodes.len()).ok()?;
    let n_edges = u32::try_from(edges.len()).ok()?;
    let mut nodes = nodes.to_vec();
    nodes.sort_unstable();
    let mut edges: Vec<(u32, u32)> = edges.iter().map(|&(u, v)| (u.min(v), u.max(v))).collect();
    edges.sort_unstable();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&n_nodes.to_le_bytes());
    for node in nodes {
        bytes.extend_from_slice(&node.to_le_bytes());
    }
    bytes.extend_from_slice(&n_edges.to_le_bytes());
    for (u, v) in edges {
        bytes.extend_from_slice(&u.to_le_bytes());
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Some(blake2_256(&bytes))
}

/// Canonical chain topology hash for full `AllowedValueSpec` inputs.
///
/// Byte-identical to `pallet_quantum_pow::topology::hash_topology`.
#[must_use]
pub fn topology_hash(
    nodes: &[u32],
    edges: &[(u32, u32)],
    allowed_h: &AllowedValueSpec<&[MilliValue]>,
    allowed_j: &AllowedValueSpec<&[MilliValue]>,
    allowed_spin: &AllowedValueSpec<&[MilliValue]>,
) -> [u8; 32] {
    let mut canonical_nodes = nodes.to_vec();
    canonical_nodes.sort_unstable();

    let mut canonical_edges: Vec<(u32, u32)> = edges
        .iter()
        .map(|&(u, v)| if u <= v { (u, v) } else { (v, u) })
        .collect();
    canonical_edges.sort_unstable();

    blake2_256(
        &(
            canonical_nodes,
            canonical_edges,
            allowed_h.canonical_bytes(),
            allowed_j.canonical_bytes(),
            allowed_spin.canonical_bytes(),
        )
            .encode(),
    )
}

/// Convenience wrapper when all three specs are discrete sets.
#[must_use]
pub fn topology_hash_sets(
    nodes: &[u32],
    edges: &[(u32, u32)],
    allowed_h: &[MilliValue],
    allowed_j: &[MilliValue],
    allowed_spin: &[MilliValue],
) -> [u8; 32] {
    topology_hash(
        nodes,
        edges,
        &AllowedValueSpec::Set(allowed_h),
        &AllowedValueSpec::Set(allowed_j),
        &AllowedValueSpec::Set(allowed_spin),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned against the pallet construction: sorted nodes, min/max edges,
    /// set-canonical specs, SCALE-encoded tuple, `blake2_256`.
    #[test]
    fn golden_topology_hash_known_fixture() {
        let nodes = [0u32, 1, 2];
        let edges = [(0u32, 1), (1, 2)];
        let h: &[i32] = &[-1000, 0, 1000];
        let j: &[i32] = &[-1000, 1000];
        let spin: &[i32] = &[-1000, 1000];

        let got = topology_hash_sets(&nodes, &edges, h, j, spin);
        // Precomputed offline (Python blake2b-256 of the SCALE payload).
        let expected =
            hex_literal("56dff5ca824517ba7f0593ec12a5dd102eb0a14f775b35c415467e0e0d6e19c2");
        assert_eq!(got, expected);
    }

    #[test]
    fn hash_is_order_independent_for_nodes_and_edges() {
        let h: &[i32] = &[-1000, 0, 1000];
        let j: &[i32] = &[-1000, 1000];
        let spin: &[i32] = &DEFAULT_SPIN_SET;
        let a = topology_hash_sets(&[2, 0, 1], &[(1, 0), (2, 1)], h, j, spin);
        let b = topology_hash_sets(&[0, 1, 2], &[(0, 1), (1, 2)], h, j, spin);
        assert_eq!(a, b);
    }

    #[test]
    fn hash_is_order_independent_for_set_elements() {
        let nodes = [0u32, 1];
        let edges = [(0u32, 1)];
        let spin: &[i32] = &DEFAULT_SPIN_SET;
        let a = topology_hash_sets(&nodes, &edges, &[1000, -1000, 0], &[-1000, 1000], spin);
        let b = topology_hash_sets(&nodes, &edges, &[-1000, 0, 1000], &[1000, -1000], spin);
        assert_eq!(a, b);
    }

    #[test]
    fn hash_changes_when_specs_differ() {
        let nodes = [0u32, 1];
        let edges = [(0u32, 1)];
        let spin: &[i32] = &DEFAULT_SPIN_SET;
        let a = topology_hash_sets(&nodes, &edges, &[-1000, 0, 1000], &[-1000, 1000], spin);
        let b = topology_hash_sets(&nodes, &edges, &[-1000, 1000], &[-1000, 1000], spin);
        assert_ne!(a, b);
    }

    #[test]
    fn native_fingerprint_matches_dwave_bytes_and_differs_from_chain_hash() {
        let expected =
            hex_literal("02bf488906736b21dc5c3257ea04da8ff3592e0c81e0be5179135c72424cc8d4");
        assert_eq!(
            native_topology_hash(&[0, 1, 2], &[(0, 1), (1, 2)]),
            Some(expected)
        );
        assert_eq!(
            native_topology_hash(&[2, 0, 1], &[(2, 1), (1, 0)]),
            Some(expected)
        );
        assert_ne!(
            expected,
            topology_hash_sets(
                &[0, 1, 2],
                &[(0, 1), (1, 2)],
                &[-1000, 0, 1000],
                &[-1000, 1000],
                &[-1000, 1000],
            )
        );
        assert_ne!(
            native_topology_hash(&[0, 1, 2], &[(0, 1), (1, 2), (1, 2)]),
            Some(expected)
        );
    }

    fn hex_literal(s: &str) -> [u8; 32] {
        let mut out = [0u8; 32];
        for i in 0..32 {
            #[expect(
                clippy::indexing_slicing,
                reason = "fixed 32-byte output; s is a 64-char hex fixture"
            )]
            {
                out[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).expect("hex");
            }
        }
        out
    }
}

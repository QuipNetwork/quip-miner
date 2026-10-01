//! Root-origin chain seeding: the two `Sudo`-wrapped calls that make a fresh
//! chain mineable.
//!
//! A chain with no `QuantumPow.DefaultTopology` accepts no proofs, so a fresh
//! network needs one root call to register a topology and a second to set its
//! difficulty. `register_topology` writes `DefaultTopology` and the
//! `MineableTopologies` whitelist entry itself when no default exists yet, so
//! those two calls are the whole sequence.
//!
//! This module encodes the calls and drives the two-call seed sequence.

use super::real::{Confirmation, RealChainClient, SignedCallOutcome};
use super::scale_types::{
    DifficultyConfig, MinerKind, QUANTUM_POW_PALLET_INDEX, REGISTER_TOPOLOGY_CALL_INDEX,
    SET_DIFFICULTY_CALL_INDEX, SUDO_CALL_INDEX, SUDO_PALLET_INDEX,
};
use super::ChainError;
use crate::drive::TopologySpec;
use crate::topology::topology_hash_sets;
use parity_scale_codec::Encode;
use quantum_validation::AllowedValueSpec;

/// The graph and allowed-value sets a `register_topology` call carries.
///
/// Every allowed-value argument uses the `Set` variant. The other variants
/// (`IntegerRange`, `ContinuousRange`) describe sampling ranges that no
/// hardware topology has needed, so the seed path does not offer them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeedTopology {
    /// Graph node ids.
    pub nodes: Vec<u32>,
    /// Graph edges as endpoint pairs.
    pub edges: Vec<(u32, u32)>,
    /// Allowed per-node h field values, in milli units.
    pub allowed_h_milli: Vec<i32>,
    /// Allowed per-edge j coupling values, in milli units.
    pub allowed_j_milli: Vec<i32>,
    /// Allowed solution spin values, in milli units.
    pub allowed_spin_milli: Vec<i32>,
}

impl SeedTopology {
    /// The chain-canonical topology hash this graph registers under.
    ///
    /// Computed locally so the caller can confirm what the chain stored
    /// instead of trusting the submit to have encoded correctly.
    #[must_use]
    pub fn topology_hash(&self) -> [u8; 32] {
        topology_hash_sets(
            &self.nodes,
            &self.edges,
            &self.allowed_h_milli,
            &self.allowed_j_milli,
            &self.allowed_spin_milli,
        )
    }
}

/// Encode `Sudo.sudo(QuantumPow.register_topology(..))`.
///
/// `Box<RuntimeCall>` encodes as the `RuntimeCall` itself, so the wrapped call
/// is the sudo prefix followed by the inner call bytes with no length prefix
/// between them.
#[must_use]
pub fn encode_register_topology(topology: &SeedTopology) -> Vec<u8> {
    let mut call = vec![
        SUDO_PALLET_INDEX,
        SUDO_CALL_INDEX,
        QUANTUM_POW_PALLET_INDEX,
        REGISTER_TOPOLOGY_CALL_INDEX,
    ];
    (
        topology.nodes.clone(),
        topology.edges.clone(),
        AllowedValueSpec::Set(topology.allowed_h_milli.clone()),
        AllowedValueSpec::Set(topology.allowed_j_milli.clone()),
        AllowedValueSpec::Set(topology.allowed_spin_milli.clone()),
    )
        .encode_to(&mut call);
    call
}

/// Encode `Sudo.sudo(QuantumPow.set_difficulty(topology_hash, difficulty))`.
#[must_use]
pub fn encode_set_difficulty(topology_hash: [u8; 32], difficulty: &DifficultyConfig) -> Vec<u8> {
    let mut call = vec![
        SUDO_PALLET_INDEX,
        SUDO_CALL_INDEX,
        QUANTUM_POW_PALLET_INDEX,
        SET_DIFFICULTY_CALL_INDEX,
    ];
    call.extend_from_slice(&topology_hash);
    difficulty.encode_to(&mut call);
    call
}

/// The difficulty a fresh chain starts at. These are the values the public
/// testnet was seeded with. They are deliberately loose: a chain with no
/// proof history has no difficulty curve to sit on, and a CPU miner has to be
/// able to land the first proof.
pub const DEFAULT_SEED_DIFFICULTY: DifficultyConfig = DifficultyConfig {
    min_solutions: 5,
    max_energy_milli: -2_500_000,
    min_diversity_milli: 200,
};

impl SeedTopology {
    /// Build a seed topology from a parsed topology spec.
    #[must_use]
    pub fn from_spec(spec: &TopologySpec) -> Self {
        Self {
            nodes: spec.topology.nodes.clone(),
            edges: spec.topology.edge_pairs(),
            allowed_h_milli: spec.allowed_h_milli.clone(),
            allowed_j_milli: spec.allowed_j_milli.clone(),
            allowed_spin_milli: spec.allowed_spin_milli.clone(),
        }
    }
}

/// Everything `seed_chain` needs.
#[derive(Clone, Debug)]
pub struct SeedParams {
    /// Validator RPC endpoint.
    pub validator: String,
    /// Sudo signer material, in any form `load_hybrid_pair` accepts.
    pub sudo_key: String,
    /// The topology to register as the chain default.
    pub topology: SeedTopology,
    /// The difficulty to set for that topology.
    pub difficulty: DifficultyConfig,
}

/// What `seed_chain` did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeedReport {
    /// The registered topology hash, confirmed by reading it back.
    pub topology_hash: [u8; 32],
    /// Node count in the registered graph.
    pub nodes: usize,
    /// Edge count in the registered graph.
    pub edges: usize,
    /// Block that included `register_topology`.
    pub register_block: String,
    /// Block that included `set_difficulty`.
    pub difficulty_block: String,
}

/// Register `params.topology` as the chain default and set its difficulty.
///
/// Refuses to run when `DefaultTopology` is already set. The pallet writes that
/// value only when it is empty (`register_topology`), and a second registration
/// would land a topology that no miner selects, so a chain that is already
/// seeded has to be wiped rather than re-seeded.
///
/// # Errors
/// Returns an error when the validator is unreachable, when a topology is
/// already registered, when either call fails to dispatch, or when the hash the
/// chain stored does not match the hash computed locally.
pub async fn seed_chain(params: SeedParams) -> Result<SeedReport, ChainError> {
    let client = RealChainClient::new(
        vec![params.validator.clone()],
        params.sudo_key.clone(),
        MinerKind::Cpu,
    );

    if let Some(existing) = client.default_topology().await? {
        return Err(ChainError::Submit(format!(
            "DefaultTopology is already set to 0x{}; a chain can only be seeded \
             once, so wipe the chain data and restart the validator first",
            hex_lower(&existing)
        )));
    }

    let expected = params.topology.topology_hash();
    let register_block = dispatch(
        &client,
        &encode_register_topology(&params.topology),
        "register_topology",
        Confirmation::DefaultTopology,
    )
    .await?;

    // Confirm the chain stored the hash computed locally. This checks the whole
    // encoding chain in one comparison: argument order, bounded-vector
    // mirroring, allowed-value variant bytes, and canonical ordering.
    let stored = client.default_topology().await?.ok_or_else(|| {
        ChainError::Submit(
            "register_topology was included but DefaultTopology is still unset; \
             the call failed inner validation"
                .into(),
        )
    })?;
    if stored != expected {
        return Err(ChainError::Submit(format!(
            "chain registered topology 0x{} but this build computed 0x{}; \
             the call encoding and the hash function disagree",
            hex_lower(&stored),
            hex_lower(&expected)
        )));
    }

    let difficulty_block = dispatch(
        &client,
        &encode_set_difficulty(expected, &params.difficulty),
        "set_difficulty",
        Confirmation::Difficulty {
            topology_hash: expected,
        },
    )
    .await?;

    Ok(SeedReport {
        topology_hash: expected,
        nodes: params.topology.nodes.len(),
        edges: params.topology.edges.len(),
        register_block,
        difficulty_block,
    })
}

/// Submit one call and return the block that included it.
async fn dispatch(
    client: &RealChainClient,
    call: &[u8],
    what: &str,
    confirmation: Confirmation,
) -> Result<String, ChainError> {
    tracing::info!(call = what, "submitting sudo call");
    let (outcome, _) = client.submit_signed_call(call, confirmation).await?;
    match outcome {
        SignedCallOutcome::Success { block, .. } => {
            tracing::info!(call = what, block = %block, "sudo call included");
            Ok(block)
        }
        SignedCallOutcome::DispatchFailed { error, .. } => Err(ChainError::Submit(format!(
            "{what} was included but the dispatch failed: {error}"
        ))),
        SignedCallOutcome::Invalid { message } => Err(ChainError::Submit(format!(
            "{what} was rejected by the transaction pool: {message}"
        ))),
        SignedCallOutcome::Dropped { message } => Err(ChainError::Submit(format!(
            "{what} was dropped before inclusion: {message}"
        ))),
    }
}

/// Lower-case hex without a prefix, for error text.
fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> SeedTopology {
        SeedTopology {
            nodes: vec![0, 1, 2],
            edges: vec![(0, 1), (1, 2)],
            allowed_h_milli: vec![-1000, 0, 1000],
            allowed_j_milli: vec![-1000, 1000],
            allowed_spin_milli: vec![-1000, 1000],
        }
    }

    #[test]
    fn register_topology_call_carries_the_sudo_and_inner_prefixes() {
        let call = encode_register_topology(&tiny());
        #[expect(
            clippy::indexing_slicing,
            reason = "a well-formed call always starts with the four prefix bytes"
        )]
        {
            assert_eq!(call[0], SUDO_PALLET_INDEX);
            assert_eq!(call[1], SUDO_CALL_INDEX);
            assert_eq!(call[2], QUANTUM_POW_PALLET_INDEX);
            assert_eq!(call[3], REGISTER_TOPOLOGY_CALL_INDEX);
        }
    }

    #[test]
    fn register_topology_arguments_encode_as_the_pallet_tuple() {
        let t = tiny();
        let call = encode_register_topology(&t);
        let expected = (
            t.nodes.clone(),
            t.edges.clone(),
            AllowedValueSpec::Set(t.allowed_h_milli.clone()),
            AllowedValueSpec::Set(t.allowed_j_milli.clone()),
            AllowedValueSpec::Set(t.allowed_spin_milli),
        )
            .encode();
        #[expect(
            clippy::indexing_slicing,
            reason = "the prefix is four bytes; the remainder is the argument block"
        )]
        {
            assert_eq!(&call[4..], &expected[..]);
        }
    }

    #[test]
    fn allowed_value_sets_encode_as_the_set_variant() {
        let call = encode_register_topology(&tiny());
        // nodes: Compact(3) = 0x0c, then 3 * u32 = 12 bytes -> 13 bytes.
        // edges: Compact(2) = 0x08, then 2 * (u32,u32) = 16 bytes -> 17 bytes.
        let h_start = 4 + 13 + 17;
        #[expect(
            clippy::indexing_slicing,
            reason = "h_start is the computed offset of the first allowed-value spec"
        )]
        {
            assert_eq!(call[h_start], 0, "AllowedValueSpec::Set is variant 0");
            assert_eq!(call[h_start + 1], 0x0c, "three h values, Compact(3)");
        }
    }

    #[test]
    fn set_difficulty_call_carries_the_hash_then_the_config() {
        let hash = [7u8; 32];
        let difficulty = DifficultyConfig {
            min_solutions: 5,
            max_energy_milli: -2_500_000,
            min_diversity_milli: 200,
        };
        let call = encode_set_difficulty(hash, &difficulty);
        #[expect(
            clippy::indexing_slicing,
            reason = "a well-formed call is prefix plus 32-byte hash plus config"
        )]
        {
            assert_eq!(call[0], SUDO_PALLET_INDEX);
            assert_eq!(call[1], SUDO_CALL_INDEX);
            assert_eq!(call[2], QUANTUM_POW_PALLET_INDEX);
            assert_eq!(call[3], SET_DIFFICULTY_CALL_INDEX);
            assert_eq!(&call[4..36], &hash[..], "H256 encodes as 32 raw bytes");
            assert_eq!(&call[36..], &difficulty.encode()[..]);
        }
    }

    #[test]
    fn a_seed_topology_hashes_to_the_canonical_chain_hash() {
        let t = tiny();
        assert_eq!(
            t.topology_hash(),
            t.topology_hash(),
            "hash is deterministic"
        );
        let mut reordered = tiny();
        reordered.nodes = vec![2, 0, 1];
        reordered.edges = vec![(1, 2), (0, 1)];
        assert_eq!(
            t.topology_hash(),
            reordered.topology_hash(),
            "the chain hash canonicalizes node and edge order"
        );
    }

    #[test]
    fn the_default_difficulty_matches_the_values_the_testnet_was_seeded_with() {
        assert_eq!(DEFAULT_SEED_DIFFICULTY.min_solutions, 5);
        assert_eq!(DEFAULT_SEED_DIFFICULTY.max_energy_milli, -2_500_000);
        assert_eq!(DEFAULT_SEED_DIFFICULTY.min_diversity_milli, 200);
    }

    #[test]
    fn a_topology_spec_converts_to_a_seed_topology() {
        let text = crate::presets::preset_spec("smoke").unwrap();
        let spec = crate::drive::parse_topology_spec(text).unwrap();
        let seed = SeedTopology::from_spec(&spec);
        assert_eq!(seed.nodes, spec.topology.nodes);
        assert_eq!(seed.edges, spec.topology.edge_pairs());
        assert_eq!(seed.allowed_h_milli, spec.allowed_h_milli);
        assert_eq!(seed.allowed_spin_milli, spec.allowed_spin_milli);
    }

    #[test]
    fn the_seed_topology_hash_matches_the_spec_topology_hash() {
        // parse_topology_spec builds Topology::hash through the same canonical
        // hash function. If these ever disagree, the confirmation read in
        // seed_chain would compare against the wrong value.
        let text = crate::presets::preset_spec("advantage2-system1").unwrap();
        let spec = crate::drive::parse_topology_spec(text).unwrap();
        let seed = SeedTopology::from_spec(&spec);
        assert_eq!(seed.topology_hash().to_vec(), spec.topology.hash);
    }
}

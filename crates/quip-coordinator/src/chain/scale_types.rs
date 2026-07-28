//! Minimal SCALE-compatible types mirrored from pallet `types.rs`.
//!
//! Prefer these over path-depending the FRAME pallets (which drag
//! polkadot-sdk git + full FRAME). Layout must stay byte-identical to
//! `pallet_quantum_pow` / `pallet_quantum_compute_mempool`.

// SCALE `Encode` for multi-field enum variants emits `index as u8` in a
// derived impl; the lint span points at the variant but attributes on the
// enum do not cover the sibling impl, so suppress at module scope.
#![expect(
    clippy::cast_possible_truncation,
    reason = "SCALE Encode writes enum discriminants as u8; pallet enums are tiny"
)]

use parity_scale_codec::{Decode, Encode};
use quantum_validation::AllowedValueSpec;
use sp_core::{H256, U256};

/// On-wire difficulty config (matches pallet `DifficultyConfig`).
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct DifficultyConfig {
    /// Minimum valid solutions required.
    pub min_solutions: u32,
    /// Energy ceiling in milli units (solutions must be strictly below).
    pub max_energy_milli: i64,
    /// Minimum diversity of the solution set in milli units.
    pub min_diversity_milli: u32,
}

/// Per-topology curve override (matches pallet `CurveC`): per-mille c-triple
/// stored under `TopologyCurveC[topology_hash]`. Field order is the pallet's.
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct CurveCScale {
    /// Easy-regime curve c (per-mille).
    pub easy_milli: u32,
    /// Knee-regime curve c (per-mille).
    pub knee_milli: u32,
    /// Hard-regime curve c (per-mille).
    pub hard_milli: u32,
}

/// Runtime-API mining snapshot (matches pallet `MiningSnapshot`).
///
/// Note: no `block_number` field — callers fetch the header separately.
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct MiningSnapshotScale {
    /// Hash of the block that contained the last winning proof.
    pub last_proof_block_hash: H256,
    /// Current (possibly decayed) difficulty gates.
    pub difficulty: DifficultyConfig,
    /// Topology identity hash.
    pub topology_hash: H256,
    /// Topology node ids.
    pub nodes: Vec<u32>,
    /// Topology undirected edges as `(u, v)` node-id pairs.
    pub edges: Vec<(u32, u32)>,
    /// Allowed linear-field values.
    pub allowed_h_values: AllowedValueSpec<Vec<i32>>,
    /// Allowed coupling values.
    pub allowed_j_values: AllowedValueSpec<Vec<i32>>,
    /// Allowed spin values.
    pub allowed_spin_values: AllowedValueSpec<Vec<i32>>,
}

/// Proof payload for `QuantumPow.submit_proof` (pallet index 10, call index 4).
///
/// Energies / diversity are **not** sent — the chain recomputes them.
/// `solutions` is a list of bit-packed spin vectors. `device_access_time_us`
/// is miner-reported compute time in microseconds (self-reported
/// observability, unverifiable; `0` = unreported).
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct QuantumProof {
    /// Topology the proof was mined for.
    pub topology_hash: H256,
    /// Derived `PoW` nonce.
    pub nonce: U256,
    /// 32-byte salt used in nonce derivation.
    pub salt: [u8; 32],
    /// Bit-packed spin vectors.
    pub solutions: Vec<Vec<u8>>,
    /// Miner-reported device access time in microseconds (`0` = unreported).
    pub device_access_time_us: u64,
}

/// Ising params nested in a mempool `JobOrder`.
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct IsingParams {
    /// Topology node ids.
    pub nodes: Vec<u32>,
    /// Topology undirected edges as `(u, v)` node-id pairs.
    pub edges: Vec<(u32, u32)>,
    /// Per-node linear fields (aligned with `nodes`).
    pub h_values: Vec<i32>,
    /// Per-edge couplings (aligned with `edges`).
    pub j_values: Vec<i32>,
    /// Optional energy gate.
    pub min_energy_milli: Option<i64>,
    /// Optional diversity gate.
    pub min_diversity_milli: Option<u32>,
    /// Optional minimum solution count.
    pub min_solutions: Option<u32>,
}

/// Order timing (`BlockNumber` = `u32` on the default runtime).
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct OrderTiming {
    /// Blocks until the order expires.
    pub deadline_blocks: u32,
    /// Blocks to wait after first solution before closing.
    pub block_wait: u32,
}

/// Minimal status enum (SCALE tag order must match the pallet).
#[derive(Clone, Copy, Debug, Encode, Decode, PartialEq, Eq)]
pub enum OrderStatus {
    /// Order is open for solutions.
    Opened,
    /// Order expired without closing.
    Expired,
    /// Order closed (rewarded / settled).
    Closed,
}

/// Reward resolution enum (SCALE tag order must match the pallet).
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub enum RewardResolution {
    /// Winner-take-all on best energy.
    SingleBest,
    /// Weighted split across top-`n` solutions.
    TopNWeighted {
        /// Number of top solutions that share the reward.
        n: u32,
    },
    /// Equal split across top-`n` solutions.
    TopNEqual {
        /// Number of top solutions that share the reward.
        n: u32,
    },
}

/// Job mode enum. Bid carries optional account / miner-type filters.
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub enum JobMode {
    /// Any registered miner may answer.
    Open,
    /// Restricted bid with optional miner and type filters.
    Bid {
        /// Allowed miner account ids (`AccountId32` raw bytes).
        miners: Option<Vec<[u8; 32]>>,
        /// Allowed miner-type tags.
        miner_types: Option<Vec<u8>>,
    },
}

/// Result delivery enum.
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub enum ResultDelivery {
    /// Results stay on-chain only.
    OnChainOnly,
    /// Push results to a callback endpoint.
    Callback {
        /// Callback endpoint bytes.
        endpoint: Vec<u8>,
    },
    /// Callback plus poll-for-result path.
    CallbackWithPoll {
        /// Callback endpoint bytes.
        endpoint: Vec<u8>,
    },
}

/// Full on-chain `JobOrder` (`AccountId32` / `Balance=u128` / `BlockNumber=u32`).
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct JobOrderScale {
    /// Spec / topology identity.
    pub spec_id: H256,
    /// Proposer account (`AccountId32` raw bytes).
    pub proposer: [u8; 32],
    /// Nested Ising problem statement.
    pub ising_params: IsingParams,
    /// Reward amount in plancks.
    pub reward: u128,
    /// Open vs bid mode.
    pub mode: JobMode,
    /// How the reward is split among winners.
    pub resolution: RewardResolution,
    /// Deadline and wait timing.
    pub timing: OrderTiming,
    /// How results are delivered.
    pub delivery: ResultDelivery,
    /// Current order status.
    pub status: OrderStatus,
    /// Block number when the order was created.
    pub created_at: u32,
    /// Block of the first accepted solution, if any.
    pub first_solution_at: Option<u32>,
    /// Number of solutions accepted so far.
    pub solution_count: u32,
}

/// Pallet index of `QuantumPow` in the runtime construct.
pub const QUANTUM_POW_PALLET_INDEX: u8 = 10;
/// Call index of `submit_proof` within `QuantumPow`.
pub const SUBMIT_PROOF_CALL_INDEX: u8 = 4;

/// SCALE-encode the `QuantumPow.submit_proof(proof)` call body.
#[must_use]
pub fn encode_submit_proof_call(proof: &QuantumProof) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(QUANTUM_POW_PALLET_INDEX);
    out.push(SUBMIT_PROOF_CALL_INDEX);
    // Call args: single composite field `proof`.
    out.extend(proof.encode());
    out
}

/// Require an `AllowedValueSpec::Set` and return its (non-empty) values.
///
/// The coordinator draws `PoW` models with [`draw_ising_milli`] over a discrete
/// allowed set. The chain's `generate_ising_model` samples `IntegerRange` /
/// `ContinuousRange` specs with a *different* RNG consumption (see
/// `quantum_validation::AllowedValueSpec::sample`), so a coordinator-side
/// expansion of a range into a `Set` would neither reproduce the chain's model
/// nor match its topology hash. Rather than mine unverifiable jobs, reject
/// non-`Set` specs (and empty sets) here so the failure surfaces at snapshot
/// decode with a clear message.
///
/// [`draw_ising_milli`]: quip_protocol::chacha8::draw_ising_milli
///
/// # Errors
/// Returns an error for `IntegerRange` / `ContinuousRange` specs and for an
/// empty `Set`.
pub fn require_set_values(spec: &AllowedValueSpec<Vec<i32>>) -> Result<Vec<i32>, String> {
    match spec {
        AllowedValueSpec::Set(v) if !v.is_empty() => Ok(v.clone()),
        AllowedValueSpec::Set(_) => {
            Err("allowed-value Set is empty; a PoW model cannot be drawn".to_string())
        }
        AllowedValueSpec::IntegerRange { .. } | AllowedValueSpec::ContinuousRange { .. } => Err(
            "coordinator requires AllowedValueSpec::Set; range specs (IntegerRange/\
             ContinuousRange) sample differently on-chain and are not supported"
                .to_string(),
        ),
    }
}

/// On-chain winning block (matches pallet `QBlock<AccountId32, u128, u32>`,
/// `types.rs:205`). Field order is the pallet's; layout must stay
/// byte-identical.
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct QBlockScale {
    /// Winning miner account (`AccountId32` raw bytes).
    pub miner: [u8; 32],
    /// 32-byte salt used in nonce derivation.
    pub salt: [u8; 32],
    /// Achieved winning energy in milli units (most negative = hardest).
    pub energy_milli: i64,
    /// Block reward in plancks.
    pub reward: u128,
    /// Substrate block number the proof was accepted at.
    pub submitted_at: u32,
    /// Active difficulty gates the proof cleared.
    pub difficulty: DifficultyConfig,
    /// Hash of the block holding the previous winning proof (nonce input).
    pub last_proof_block_hash: H256,
    /// Topology the proof was mined against.
    pub topology_hash: H256,
    /// Miner-reported compute time in microseconds (`0` = unreported).
    pub device_access_time_us: u64,
}

/// Runtime-API view augmenting [`QBlockScale`] with its derived nonce
/// (matches pallet `QBlockWithNonce`, `types.rs:236`).
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct QBlockWithNonceScale {
    /// The winning block. (Field name mirrors the pallet's `solution`.)
    pub solution: QBlockScale,
    /// Derived `PoW` nonce as a `U256` (big-endian of the BLAKE3 digest).
    pub nonce: U256,
}

/// Registered topology definition (matches pallet
/// `TopologyMeta<Nodes, Edges, AllowedValues, BlockNumber>`, `types.rs:79`).
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct TopologyMetaScale {
    /// Topology node ids.
    pub nodes: Vec<u32>,
    /// Topology undirected edges as `(u, v)` node-id pairs.
    pub edges: Vec<(u32, u32)>,
    /// Allowed linear-field values.
    pub allowed_h_values: AllowedValueSpec<Vec<i32>>,
    /// Allowed coupling values.
    pub allowed_j_values: AllowedValueSpec<Vec<i32>>,
    /// Allowed spin values.
    pub allowed_spin_values: AllowedValueSpec<Vec<i32>>,
    /// Block number the topology was registered at.
    pub registered_at: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qblock_with_nonce_scale_roundtrip() {
        let q = QBlockScale {
            miner: [0x11; 32],
            salt: [0x22; 32],
            energy_milli: -14_520_000,
            reward: 1_000_000_000_000,
            submitted_at: 4242,
            difficulty: DifficultyConfig {
                min_solutions: 5,
                max_energy_milli: -14_490_000,
                min_diversity_milli: 200,
            },
            last_proof_block_hash: H256::from([3u8; 32]),
            topology_hash: H256::from([0x6e; 32]),
            device_access_time_us: 987_654,
        };
        let qn = QBlockWithNonceScale {
            solution: q,
            nonce: U256::from_big_endian(&[0xab; 32]),
        };
        // Runtime API returns Option<QBlockWithNonce>.
        let encoded = Some(qn.clone()).encode();
        let decoded: Option<QBlockWithNonceScale> =
            Decode::decode(&mut &encoded[..]).expect("decode");
        assert_eq!(decoded, Some(qn));
    }

    #[test]
    fn topology_meta_scale_roundtrip() {
        let m = TopologyMetaScale {
            nodes: vec![0, 1, 2, 3],
            edges: vec![(0, 1), (1, 2), (2, 3), (0, 3)],
            allowed_h_values: AllowedValueSpec::Set(vec![-1000, 0, 1000]),
            allowed_j_values: AllowedValueSpec::Set(vec![-1000, 1000]),
            allowed_spin_values: AllowedValueSpec::Set(vec![-1000, 1000]),
            registered_at: 7,
        };
        let encoded = Some(m.clone()).encode();
        let decoded: Option<TopologyMetaScale> =
            Decode::decode(&mut &encoded[..]).expect("decode");
        assert_eq!(decoded, Some(m));
    }

    #[test]
    fn qblock_with_nonce_decode_rejects_truncated() {
        // Untrusted chain bytes: a chopped payload must Err, never panic.
        let qn = QBlockWithNonceScale {
            solution: QBlockScale {
                miner: [1; 32],
                salt: [2; 32],
                energy_milli: -1,
                reward: 0,
                submitted_at: 0,
                difficulty: DifficultyConfig {
                    min_solutions: 1,
                    max_energy_milli: -1,
                    min_diversity_milli: 0,
                },
                last_proof_block_hash: H256::zero(),
                topology_hash: H256::zero(),
                device_access_time_us: 0,
            },
            nonce: U256::zero(),
        };
        let enc = qn.encode();
        #[expect(
            clippy::indexing_slicing,
            reason = "enc is a full encode; len-4 in bounds"
        )]
        let truncated = &enc[..enc.len() - 4];
        assert!(QBlockWithNonceScale::decode(&mut &truncated[..]).is_err());
    }

    #[test]
    fn require_set_values_accepts_nonempty_set() {
        let spec = AllowedValueSpec::Set(vec![-1000, 0, 1000]);
        assert_eq!(require_set_values(&spec).unwrap(), vec![-1000, 0, 1000]);
    }

    #[test]
    fn require_set_values_rejects_empty_set() {
        let spec = AllowedValueSpec::Set(Vec::new());
        assert!(require_set_values(&spec).is_err());
    }

    #[test]
    fn require_set_values_rejects_ranges() {
        // Range specs sample differently on-chain; the coordinator must not
        // silently expand them into a Set (would diverge from consensus).
        assert!(require_set_values(&AllowedValueSpec::IntegerRange { min: -2, max: 2 }).is_err());
        assert!(require_set_values(&AllowedValueSpec::ContinuousRange {
            min: -1000,
            max: 1000
        })
        .is_err());
    }

    #[test]
    fn quantum_proof_scale_roundtrip() {
        let proof = QuantumProof {
            topology_hash: H256::from([0xab; 32]),
            nonce: U256::from(0x1234_5678u64),
            salt: [0xcd; 32],
            solutions: vec![vec![0b1010_0101], vec![0x00, 0xff]],
            device_access_time_us: 987_654,
        };
        let encoded = proof.encode();
        let decoded = QuantumProof::decode(&mut &encoded[..]).expect("decode");
        assert_eq!(decoded, proof);
    }

    #[test]
    fn mining_snapshot_scale_roundtrip() {
        let snap = MiningSnapshotScale {
            last_proof_block_hash: H256::from([1u8; 32]),
            difficulty: DifficultyConfig {
                min_solutions: 5,
                max_energy_milli: -1_200_000,
                min_diversity_milli: 200,
            },
            topology_hash: H256::from([2u8; 32]),
            nodes: vec![0, 1, 2],
            edges: vec![(0, 1), (1, 2)],
            allowed_h_values: AllowedValueSpec::Set(vec![-1000, 0, 1000]),
            allowed_j_values: AllowedValueSpec::Set(vec![-1000, 1000]),
            allowed_spin_values: AllowedValueSpec::Set(vec![-1000, 1000]),
        };
        let encoded = Some(snap.clone()).encode();
        let decoded: Option<MiningSnapshotScale> =
            Decode::decode(&mut &encoded[..]).expect("decode");
        assert_eq!(decoded, Some(snap));
    }

    #[test]
    fn submit_proof_call_starts_with_pallet_and_call_index() {
        let proof = QuantumProof {
            topology_hash: H256::repeat_byte(0),
            nonce: U256::zero(),
            salt: [0u8; 32],
            solutions: vec![vec![0]],
            device_access_time_us: 0,
        };
        let call = encode_submit_proof_call(&proof);
        #[expect(
            clippy::indexing_slicing,
            reason = "call always starts with pallet+call index bytes"
        )]
        {
            assert_eq!(call[0], QUANTUM_POW_PALLET_INDEX);
            assert_eq!(call[1], SUBMIT_PROOF_CALL_INDEX);
            // Remainder is the proof encoding.
            assert_eq!(&call[2..], &proof.encode());
        }
    }

    #[test]
    fn job_order_ising_params_roundtrip() {
        let params = IsingParams {
            nodes: vec![0, 1],
            edges: vec![(0, 1)],
            h_values: vec![0, 0],
            j_values: vec![-1000],
            min_energy_milli: Some(-500),
            min_diversity_milli: Some(100),
            min_solutions: Some(2),
        };
        let order = JobOrderScale {
            spec_id: H256::repeat_byte(9),
            proposer: [0x11; 32],
            ising_params: params,
            reward: 1_000_000_000_000,
            mode: JobMode::Open,
            resolution: RewardResolution::SingleBest,
            timing: OrderTiming {
                deadline_blocks: 100,
                block_wait: 10,
            },
            delivery: ResultDelivery::OnChainOnly,
            status: OrderStatus::Opened,
            created_at: 42,
            first_solution_at: None,
            solution_count: 0,
        };
        let enc = order.encode();
        let dec = JobOrderScale::decode(&mut &enc[..]).expect("decode");
        assert_eq!(dec, order);
    }

    fn valid_job_order_bytes() -> Vec<u8> {
        JobOrderScale {
            spec_id: H256::repeat_byte(9),
            proposer: [0x11; 32],
            ising_params: IsingParams {
                nodes: vec![0, 1],
                edges: vec![(0, 1)],
                h_values: vec![0, 0],
                j_values: vec![-1000],
                min_energy_milli: Some(-500),
                min_diversity_milli: Some(100),
                min_solutions: Some(2),
            },
            reward: 1_000_000_000_000,
            mode: JobMode::Open,
            resolution: RewardResolution::SingleBest,
            timing: OrderTiming {
                deadline_blocks: 100,
                block_wait: 10,
            },
            delivery: ResultDelivery::OnChainOnly,
            status: OrderStatus::Opened,
            created_at: 42,
            first_solution_at: None,
            solution_count: 0,
        }
        .encode()
    }

    // The JobProposed decode (real.rs) parses untrusted on-chain bytes; it must
    // reject malformed input with an error, never panic. Only a live #[ignore]
    // devnet test exercised this before — these are the offline guards.

    #[test]
    fn job_order_decode_rejects_truncated_bytes() {
        let enc = valid_job_order_bytes();
        // Chop the tail: the later fields (timing/status/counts) can no longer
        // be decoded.
        #[expect(
            clippy::indexing_slicing,
            reason = "enc is a full JobOrderScale encode; len-4 is in bounds"
        )]
        let truncated = &enc[..enc.len() - 4];
        assert!(JobOrderScale::decode(&mut &truncated[..]).is_err());
    }

    #[test]
    fn job_order_decode_rejects_empty() {
        assert!(JobOrderScale::decode(&mut &[][..]).is_err());
    }

    #[test]
    fn job_order_decode_rejects_bad_enum_tag() {
        // `mode: JobMode` is a 3-field composite; corrupt the SCALE variant tag
        // of the first enum (`mode`, after the fixed-size spec_id/proposer and
        // the variable ising_params). A tag past the variant count must error,
        // not panic. Scan for the first byte whose flip yields an out-of-range
        // enum tag by fuzzing each position and asserting no decode panics.
        let enc = valid_job_order_bytes();
        for i in 0..enc.len() {
            let mut bad = enc.clone();
            #[expect(
                clippy::indexing_slicing,
                reason = "i iterates 0..enc.len(); in-bounds by construction"
            )]
            {
                bad[i] = 0xFF; // 0xFF is out of range for every enum in the struct
            }
            // Must return Ok or Err, never panic (the point of the guard).
            let _ = JobOrderScale::decode(&mut &bad[..]);
        }
    }
}

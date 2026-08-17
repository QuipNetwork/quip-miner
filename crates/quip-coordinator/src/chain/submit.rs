//! Extrinsic submit + hybrid sign (CONFIRM) and groundable receipt classification.

/// Action the fire-loop should take after a submit attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitAction {
    /// Extrinsic accepted (or no error string to classify).
    Success,
    /// Transient / retriable pallet rejection (energy, proof limit).
    Retry,
    /// Round-stale rejection; abandon the current generation.
    StopRoundStale,
    /// Fatal rejection or unknown error; stop submitting this proof.
    StopFatal,
}

/// What a submit attempt produced: the action the fire-loop takes, plus the
/// chain detail the dashboard reports on the submission row.
///
/// `extrinsic_hash` is known as soon as the extrinsic is signed, so it is set on
/// every path that reached the node. `block_hash` and `block_number` are set only
/// on [`SubmitAction::Success`], which the pallet confirmation defines as a win:
/// `QuantumPow.QBlocks[block_number].miner` equals the signing account.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubmitReceipt {
    /// Action the fire-loop should take.
    pub action: SubmitAction,
    /// Hash of the signed extrinsic, once it is signed.
    pub extrinsic_hash: Option<[u8; 32]>,
    /// Hash of the winning block. `None` when the submission did not win.
    pub block_hash: Option<[u8; 32]>,
    /// Height of the winning block. `None` when the submission did not win.
    pub block_number: Option<u64>,
}

impl SubmitReceipt {
    /// A receipt that carries an action and no chain detail.
    #[must_use]
    pub const fn action_only(action: SubmitAction) -> Self {
        Self {
            action,
            extrinsic_hash: None,
            block_hash: None,
            block_number: None,
        }
    }
}

/// A validated proof ready for chain submission.
#[derive(Debug, Clone)]
pub struct Proof {
    /// Wire job id this proof answers.
    pub job_id: Vec<u8>,
    /// Best solution energy in milli units.
    pub best_energy_milli: i64,
    /// Diversity of the solution set in milli units.
    pub diversity_milli: u32,
    /// Count of solutions that passed local gates.
    pub n_valid: u32,
    /// Valid solutions (spins + reported energies).
    pub solutions: Vec<quip_proto::v1::Solution>,
    /// `true` when this is a `PoW` proof (not a mempool order).
    pub is_pow: bool,
    /// Mempool order id bytes; empty for pure `PoW`.
    pub order_id: Vec<u8>,
    /// Producer generation this proof belongs to.
    pub generation: u64,
    /// `PoW` salt (32 bytes). Required for live `submit_proof`; empty for
    /// `FakeChain` paths that only record the proof.
    pub salt: Vec<u8>,
    /// Miner-reported compute time in microseconds, from the result's
    /// `SamplerMeta.device_access_time_us`. Plumbed into the on-chain
    /// `QuantumProof`; `0` = unreported.
    pub device_access_time_us: u64,
}

/// Outcome of a `MinerRegistry.set_descriptor` submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorOutcome {
    /// The descriptor landed (create or replace).
    Filed,
    /// The pallet rejected the payload. Do not retry.
    Rejected,
}

const DESCRIPTOR_REJECT: [&str; 20] = [
    "EmptyNodeId",
    "EmptyNodeName",
    "EmptyPublicHost",
    "EmptyRpcEndpoint",
    "EmptyMinerLabel",
    "EmptyMinerBackend",
    "EmptyMinerDeviceId",
    "EmptyOsSystem",
    "EmptyCpuBrand",
    "EmptyCpuArch",
    "EmptyGpuVendor",
    "EmptyGpuName",
    "InvalidGpuUtilization",
    "EmptyPythonVersion",
    "EmptyQuipVersion",
    "EmptyDockerImage",
    "NoMiners",
    "InvalidPort",
    "InsufficientBalance",
    "LiquidityRestrictions",
];

/// Classify a `set_descriptor` pallet error. `None` is a successful dispatch.
/// Unknown strings stay `None` so the caller can treat them as transient.
#[must_use]
pub fn classify_descriptor(error: Option<&str>) -> Option<DescriptorOutcome> {
    let Some(e) = error else {
        return Some(DescriptorOutcome::Filed);
    };
    if DESCRIPTOR_REJECT.iter().any(|s| e.contains(s)) {
        return Some(DescriptorOutcome::Rejected);
    }
    None
}

/// Outcome of a `MinerRegistry.participate` submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticipationOutcome {
    /// The declaration landed.
    Declared,
    /// This account already declared this qblock. Treat as success.
    AlreadyDeclared,
    /// The candidate qblock moved under the call.
    StaleQBlock,
    /// No descriptor is stored for this account.
    DescriptorMissing,
}

/// Classify a `participate` pallet error. `None` is a successful dispatch.
/// Unknown strings stay `None` so the caller can treat them as transient.
#[must_use]
pub fn classify_participation(error: Option<&str>) -> Option<ParticipationOutcome> {
    let Some(e) = error else {
        return Some(ParticipationOutcome::Declared);
    };
    if e.contains("DuplicateParticipation") {
        return Some(ParticipationOutcome::AlreadyDeclared);
    }
    if e.contains("InvalidQBlockId") {
        return Some(ParticipationOutcome::StaleQBlock);
    }
    if e.contains("DescriptorRequired") {
        return Some(ParticipationOutcome::DescriptorMissing);
    }
    None
}

/// Outcome of a `QuantumPow.register_miner` submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistrationOutcome {
    /// The registration landed and the deposit is reserved.
    Registered,
    /// This account was already in `QuantumPow.Miners`. Treat as success.
    AlreadyRegistered,
}

/// Classify a `register_miner` pallet error. `None` is a successful dispatch.
///
/// Only the already-registered race is benign. Everything else — a deposit the
/// account cannot cover above all — stays `None` so the caller reports it and
/// retries rather than marking the miner ready to submit proofs it cannot land.
#[must_use]
pub fn classify_registration(error: Option<&str>) -> Option<RegistrationOutcome> {
    let Some(e) = error else {
        return Some(RegistrationOutcome::Registered);
    };
    if e.contains("MinerAlreadyRegistered") {
        return Some(RegistrationOutcome::AlreadyRegistered);
    }
    None
}

/// Classify a pallet/dispatch error string into a fire-loop action.
///
/// Mirrors `substrate/submitter.py:_classify_receipt`. Unknown → fail loud.
#[must_use]
pub fn classify_receipt(error: Option<&str>) -> SubmitAction {
    const RETRY: [&str; 2] = ["InsufficientEnergy", "ProofLimitReached"];
    const STALE: [&str; 3] = ["InvalidNonce", "TopologyNotRegistered", "InvalidTopology"];
    const FATAL: [&str; 3] = [
        "InsufficientSolutions",
        "InsufficientDiversity",
        "MinerNotRegistered",
    ];
    let Some(e) = error else {
        return SubmitAction::Success;
    };
    if RETRY.iter().any(|s| e.contains(s)) {
        return SubmitAction::Retry;
    }
    if STALE.iter().any(|s| e.contains(s)) {
        return SubmitAction::StopRoundStale;
    }
    if FATAL.iter().any(|s| e.contains(s)) {
        return SubmitAction::StopFatal;
    }
    SubmitAction::StopFatal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_pallet_errors() {
        assert!(matches!(classify_receipt(None), SubmitAction::Success));
        assert!(matches!(
            classify_receipt(Some("InsufficientEnergy")),
            SubmitAction::Retry
        ));
        assert!(matches!(
            classify_receipt(Some("ProofLimitReached")),
            SubmitAction::Retry
        ));
        assert!(matches!(
            classify_receipt(Some("InvalidNonce")),
            SubmitAction::StopRoundStale
        ));
        assert!(matches!(
            classify_receipt(Some("TopologyNotRegistered")),
            SubmitAction::StopRoundStale
        ));
        assert!(matches!(
            classify_receipt(Some("InsufficientSolutions")),
            SubmitAction::StopFatal
        ));
        assert!(matches!(
            classify_receipt(Some("SomethingUnknown")),
            SubmitAction::StopFatal
        ));
    }

    #[test]
    fn classifies_descriptor_errors() {
        assert_eq!(classify_descriptor(None), Some(DescriptorOutcome::Filed));
        assert_eq!(
            classify_descriptor(Some("MinerRegistry: EmptyNodeName")),
            Some(DescriptorOutcome::Rejected)
        );
        assert_eq!(
            classify_descriptor(Some("NoMiners")),
            Some(DescriptorOutcome::Rejected)
        );
        assert_eq!(
            classify_descriptor(Some("InsufficientBalance")),
            Some(DescriptorOutcome::Rejected)
        );
        assert_eq!(classify_descriptor(Some("SomethingUnknown")), None);
    }

    #[test]
    fn classifies_registration_errors() {
        assert_eq!(
            classify_registration(None),
            Some(RegistrationOutcome::Registered)
        );
        assert_eq!(
            classify_registration(Some("QuantumPow: MinerAlreadyRegistered")),
            Some(RegistrationOutcome::AlreadyRegistered)
        );
        // A deposit the account cannot reserve must not read as success.
        assert_eq!(classify_registration(Some("InsufficientBalance")), None);
        assert_eq!(classify_registration(Some("SomethingUnknown")), None);
    }

    #[test]
    fn classifies_participation_errors() {
        assert_eq!(
            classify_participation(None),
            Some(ParticipationOutcome::Declared)
        );
        assert_eq!(
            classify_participation(Some("MinerRegistry: DuplicateParticipation")),
            Some(ParticipationOutcome::AlreadyDeclared)
        );
        assert_eq!(
            classify_participation(Some("InvalidQBlockId")),
            Some(ParticipationOutcome::StaleQBlock)
        );
        assert_eq!(
            classify_participation(Some("DescriptorRequired")),
            Some(ParticipationOutcome::DescriptorMissing)
        );
        assert_eq!(classify_participation(Some("SomethingUnknown")), None);
    }

    /// A receipt with no chain detail is the shape every non-winning path
    /// returns. Keeping one constructor for it stops each call site inventing
    /// its own `None` triple.
    #[test]
    fn action_only_carries_no_chain_detail() {
        let r = SubmitReceipt::action_only(SubmitAction::Retry);
        assert_eq!(r.action, SubmitAction::Retry);
        assert_eq!(r.extrinsic_hash, None);
        assert_eq!(r.block_hash, None);
        assert_eq!(r.block_number, None);
    }

    #[test]
    fn a_winning_receipt_carries_the_extrinsic_and_the_block() {
        let r = SubmitReceipt {
            action: SubmitAction::Success,
            extrinsic_hash: Some([0xab; 32]),
            block_hash: Some([0xcd; 32]),
            block_number: Some(10_249),
        };
        assert_eq!(r.action, SubmitAction::Success);
        assert_eq!(r.block_number, Some(10_249));
    }
}

//! Optional `system_dryRun` diagnostics.
//!
//! Submit outcome comes from chain state, which says whether a proof landed but
//! not why it did not. A dry run returns `ApplyExtrinsicResult`, a small closed
//! type that decodes without runtime metadata, so it can name the failing
//! pallet error.
//!
//! `system_dryRun` is an unsafe RPC method. A node started with
//! `--rpc-methods=safe`, which is what this project deploys, denies it. This
//! module is therefore probed at runtime and contributes to log text only.
//!
//! Nothing in this module may influence a `SubmitAction` or any other
//! control-flow decision. It produces log text only.

use super::scale_types::{MINER_REGISTRY_PALLET_INDEX, QUANTUM_POW_PALLET_INDEX};

/// A dispatch error localised to one pallet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModuleError {
    /// Runtime index of the pallet that raised the error.
    pub pallet: u8,
    /// Pallet-local error index, little-endian in a four-byte field.
    pub error: [u8; 4],
}

/// SCALE variant index of `DispatchError::Module`.
const DISPATCH_ERROR_MODULE: u8 = 3;

/// Decode a `system_dryRun` result into a module error, when it is one.
///
/// The encoding is `Result<Result<(), DispatchError>, TransactionValidityError>`.
/// Returns `None` for a success, for a non-module error, and for any blob this
/// build does not recognise. Diagnostics must never fail loudly.
#[must_use]
pub fn decode_dispatch_error(bytes: &[u8]) -> Option<ModuleError> {
    // Outer Result: 0x00 is Ok. Anything else is a validity error, which
    // carries no pallet error to report.
    if bytes.first().copied()? != 0x00 {
        return None;
    }
    // Inner Result: 0x00 is a successful dispatch.
    if bytes.get(1).copied()? != 0x01 {
        return None;
    }
    if bytes.get(2).copied()? != DISPATCH_ERROR_MODULE {
        return None;
    }
    let pallet = bytes.get(3).copied()?;
    let error = bytes.get(4..8)?;
    let mut buf = [0u8; 4];
    buf.copy_from_slice(error);
    Some(ModuleError { pallet, error: buf })
}

/// Error names for the pallets this coordinator calls.
///
/// These tables are diagnostics only. The runtime is the authority, and a stale
/// table degrades a log line rather than a decision. The order below is the
/// declaration order, which is the SCALE index, taken from
/// `pallets/quantum-pow/src/lib.rs` lines 355 to 409 and
/// `pallets/miner-registry/src/lib.rs` lines 494 to 518. See
/// `docs/chain-storage-notes.md`.
const QUANTUM_POW_ERRORS: [&str; 25] = [
    "MinerAlreadyRegistered",
    "MinerNotRegistered",
    "TopologyAlreadyRegistered",
    "TopologyNotRegistered",
    "InvalidCurve",
    "GraphTooSmall",
    "InvalidTopology",
    "ProofLimitReached",
    "InvalidNonce",
    "NoSolutionsSubmitted",
    "InvalidSpinValues",
    "SolutionLengthMismatch",
    "InsufficientEnergy",
    "InsufficientDiversity",
    "InsufficientSolutions",
    "ArithmeticOverflow",
    "EmptyAllowedValues",
    "EncodingTooWide",
    "PackedSolutionLengthMismatch",
    "InvalidEncodedSpin",
    "PackedSolutionTooLarge",
    "TopologyNotMineable",
    "TopologyIsDefault",
    "MineableTopologyConflict",
    "InvalidDiversityConfig",
];

const MINER_REGISTRY_ERRORS: [&str; 22] = [
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
    "DescriptorNotFound",
    "DescriptorRequired",
    "InvalidQBlockId",
    "DuplicateParticipation",
];

/// Render a module error for an operator.
#[must_use]
pub fn describe_module_error(e: &ModuleError) -> String {
    let idx = u32::from_le_bytes(e.error) as usize;
    let (pallet_name, table): (&str, &[&str]) = match e.pallet {
        QUANTUM_POW_PALLET_INDEX => ("QuantumPow", &QUANTUM_POW_ERRORS),
        MINER_REGISTRY_PALLET_INDEX => ("MinerRegistry", &MINER_REGISTRY_ERRORS),
        other => return format!("pallet {other} error {idx} (no name table for this pallet)"),
    };
    match table.get(idx) {
        Some(name) => format!("{pallet_name}::{name}"),
        None => format!("{pallet_name} error {idx} (index beyond the known table)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ApplyExtrinsicResult` is `Result<Result<(), DispatchError>, _>`. The outer
    /// Ok is 0x00, the inner Err is 0x01, then `DispatchError::Module` is
    /// variant 3 followed by the pallet index and a four-byte error index.
    #[test]
    fn a_module_error_is_decoded_from_the_documented_layout() {
        let raw = [0x00, 0x01, 0x03, 10, 7, 0, 0, 0];
        let e = decode_dispatch_error(&raw).unwrap();
        assert_eq!(e.pallet, 10);
        assert_eq!(e.error, [7, 0, 0, 0]);
    }

    #[test]
    fn a_successful_dry_run_has_no_error() {
        // Outer Ok, inner Ok.
        assert!(decode_dispatch_error(&[0x00, 0x00]).is_none());
    }

    #[test]
    fn a_truncated_blob_yields_none_not_a_panic() {
        assert!(decode_dispatch_error(&[0x00, 0x01, 0x03, 10]).is_none());
        assert!(decode_dispatch_error(&[]).is_none());
    }

    /// The names are diagnostics only. A wrong mapping must never reach control
    /// flow, so an unknown index still produces a readable string.
    #[test]
    fn known_errors_are_named_and_unknown_ones_are_still_readable() {
        let known = describe_module_error(&ModuleError {
            pallet: QUANTUM_POW_PALLET_INDEX,
            error: [0, 0, 0, 0],
        });
        assert!(known.contains("QuantumPow"), "{known}");

        let unknown = describe_module_error(&ModuleError {
            pallet: 99,
            error: [200, 0, 0, 0],
        });
        assert!(unknown.contains("99"), "{unknown}");
        assert!(unknown.contains("200"), "{unknown}");
    }
}

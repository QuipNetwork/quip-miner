//! Startup compatibility check against the configured validator.
//!
//! The coordinator talks to the chain through hand-mirrored SCALE types
//! (`scale_types.rs`), a hardcoded pallet/call index, and the `QuantumPowApi`
//! runtime API. All three are pinned to a runtime version. Against a validator
//! that predates them, every one of those reads fails — and the failure
//! surfaces deep inside the feeder loop, one poll at a time, as an opaque
//! decode or "method not found" error.
//!
//! [`preflight`] moves that discovery to startup and names it, so an operator
//! sees "this validator does not expose `QuantumPowApi`" instead of a coordinator
//! that runs forever and mines nothing.

use super::{ChainError, RealChainClient};
use serde_json::Value;

/// Runtime API trait the coordinator mines against.
const QUANTUM_POW_API: &str = "QuantumPowApi";

/// Minimum `QuantumPowApi` version the coordinator can drive.
///
/// Version 2 is the first that takes a topology selector on `mining_snapshot`
/// (`Option<H256>`), which is the shape [`RealChainClient::fetch_mining_snapshot`]
/// encodes. A v1 runtime declares the same method name with a different
/// argument list, so the call would not decode — feature-detecting on the
/// reported version is exactly what the pallet's `#[api_version(2)]` is for.
const MIN_QUANTUM_POW_API: u32 = 2;

/// What the connected validator reports about itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatorInfo {
    /// Runtime `specName` (e.g. `quip-runtime`).
    pub spec_name: String,
    /// Runtime `specVersion`.
    pub spec_version: u32,
    /// Runtime `transactionVersion`; the signed-extrinsic encoding generation.
    pub transaction_version: u32,
    /// Reported `QuantumPowApi` version, or `None` when the runtime does not
    /// expose that API at all.
    pub quantum_pow_api: Option<u32>,
}

/// Why the connected validator cannot be mined against.
#[derive(Debug)]
pub enum PreflightError {
    /// The validator could not be reached or did not answer the version query.
    Unreachable(ChainError),
    /// The runtime version response was not the expected shape.
    Malformed(String),
    /// The runtime exposes no `QuantumPowApi`: not a `QuIP` mining validator.
    MissingApi(Box<ValidatorInfo>),
    /// The runtime exposes an older `QuantumPowApi` than the coordinator drives.
    ApiTooOld {
        /// What the validator reports.
        info: Box<ValidatorInfo>,
        /// The minimum this coordinator can drive.
        required: u32,
    },
}

impl std::fmt::Display for PreflightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreachable(e) => {
                write!(f, "cannot reach validator to check compatibility: {e}")
            }
            Self::Malformed(s) => write!(f, "validator runtime version malformed: {s}"),
            Self::MissingApi(info) => write!(
                f,
                "validator runtime {}/{} exposes no {QUANTUM_POW_API}; \
                 it is not a QuIP mining validator (upgrade the validator)",
                info.spec_name, info.spec_version
            ),
            Self::ApiTooOld { info, required } => write!(
                f,
                "validator runtime {}/{} exposes {QUANTUM_POW_API} v{}, but this \
                 coordinator drives v{required} or newer (upgrade the validator)",
                info.spec_name,
                info.spec_version,
                info.quantum_pow_api.unwrap_or(0)
            ),
        }
    }
}

impl std::error::Error for PreflightError {}

/// Substrate runtime-API identifier: blake2b of the trait name with an **8-byte
/// digest**. This is what `sp_api`'s macro emits as each API's `ID`, and the key
/// `state_getRuntimeVersion` reports it under.
///
/// Note this is not `blake2_256(..)[..8]` — blake2b keys its initial state on
/// the requested output length, so a 64-bit digest is a different value than a
/// truncated 256-bit one. `api_id_matches_substrate_core` pins the difference.
fn api_id(trait_name: &str) -> [u8; 8] {
    use blake2::digest::{Update as _, VariableOutput as _};
    use blake2::Blake2bVar;

    let mut id = [0u8; 8];
    let Ok(mut hasher) = Blake2bVar::new(8) else {
        return id;
    };
    hasher.update(trait_name.as_bytes());
    let _ = hasher.finalize_variable(&mut id);
    id
}

/// Find `trait_name`'s version in a `state_getRuntimeVersion` `apis` array,
/// which is a list of `[<0x-prefixed 8-byte id>, <version>]` pairs.
fn find_api_version(apis: &Value, trait_name: &str) -> Option<u32> {
    let want = api_id(trait_name);
    let want_hex = format!("0x{}", hex_lower(&want));
    apis.as_array()?.iter().find_map(|entry| {
        let pair = entry.as_array()?;
        let id = pair.first()?.as_str()?;
        if id.eq_ignore_ascii_case(&want_hex) {
            u32::try_from(pair.get(1)?.as_u64()?).ok()
        } else {
            None
        }
    })
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Parse a `state_getRuntimeVersion` response into [`ValidatorInfo`].
fn parse_runtime_version(rv: &Value) -> Result<ValidatorInfo, PreflightError> {
    let spec_name = rv
        .get("specName")
        .and_then(Value::as_str)
        .unwrap_or("<unknown>")
        .to_string();
    let spec_version = rv
        .get("specVersion")
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| PreflightError::Malformed("specVersion missing".into()))?;
    let transaction_version = rv
        .get("transactionVersion")
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| PreflightError::Malformed("transactionVersion missing".into()))?;
    let quantum_pow_api = rv
        .get("apis")
        .and_then(|apis| find_api_version(apis, QUANTUM_POW_API));
    Ok(ValidatorInfo {
        spec_name,
        spec_version,
        transaction_version,
        quantum_pow_api,
    })
}

impl RealChainClient {
    /// Check the configured validator can actually be mined against.
    ///
    /// Reports what the validator is at `info` either way, so the version pair
    /// is in the log before anything else happens.
    ///
    /// # Errors
    /// Returns [`PreflightError`] when the validator is unreachable, answers
    /// with a malformed runtime version, or exposes no usable `QuantumPowApi`.
    pub async fn preflight(&self) -> Result<ValidatorInfo, PreflightError> {
        let rv = self
            .runtime_version_raw()
            .await
            .map_err(PreflightError::Unreachable)?;
        let info = parse_runtime_version(&rv)?;

        tracing::info!(
            spec_name = %info.spec_name,
            spec_version = info.spec_version,
            transaction_version = info.transaction_version,
            quantum_pow_api = %crate::logging::display_option(info.quantum_pow_api),
            "validator runtime"
        );

        match info.quantum_pow_api {
            None => Err(PreflightError::MissingApi(Box::new(info))),
            Some(v) if v < MIN_QUANTUM_POW_API => Err(PreflightError::ApiTooOld {
                info: Box::new(info),
                required: MIN_QUANTUM_POW_API,
            }),
            Some(_) => Ok(info),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Substrate's well-known `Core` runtime-API id. If this matches, the
    /// derivation is the same one the node uses, so the `QuantumPowApi` lookup
    /// is looking for the right key.
    #[test]
    fn api_id_matches_substrate_core() {
        assert_eq!(hex_lower(&api_id("Core")), "df6acb689907609b");
    }

    #[test]
    fn finds_quantum_pow_api_version() {
        let want = format!("0x{}", hex_lower(&api_id(QUANTUM_POW_API)));
        let apis = json!([["0xdf6acb689907609b", 5], [want, 2]]);
        assert_eq!(find_api_version(&apis, QUANTUM_POW_API), Some(2));
    }

    #[test]
    fn api_lookup_is_case_insensitive_on_hex() {
        let want = format!("0X{}", hex_lower(&api_id(QUANTUM_POW_API)).to_uppercase());
        let apis = json!([[want, 7]]);
        assert_eq!(find_api_version(&apis, QUANTUM_POW_API), Some(7));
    }

    #[test]
    fn missing_api_reads_as_none() {
        let apis = json!([["0xdf6acb689907609b", 5]]);
        assert_eq!(find_api_version(&apis, QUANTUM_POW_API), None);
        assert_eq!(find_api_version(&json!([]), QUANTUM_POW_API), None);
        assert_eq!(find_api_version(&json!(null), QUANTUM_POW_API), None);
    }

    fn rv_with(api: Option<u32>) -> Value {
        let mut apis = vec![json!(["0xdf6acb689907609b", 5])];
        if let Some(v) = api {
            apis.push(json!([
                format!("0x{}", hex_lower(&api_id(QUANTUM_POW_API))),
                v
            ]));
        }
        json!({
            "specName": "quip-runtime",
            "specVersion": 114,
            "transactionVersion": 6,
            "apis": apis,
        })
    }

    #[test]
    fn parses_a_full_runtime_version() {
        let info = parse_runtime_version(&rv_with(Some(2))).unwrap();
        assert_eq!(info.spec_name, "quip-runtime");
        assert_eq!(info.spec_version, 114);
        assert_eq!(info.transaction_version, 6);
        assert_eq!(info.quantum_pow_api, Some(2));
    }

    #[test]
    fn missing_spec_version_is_malformed() {
        let rv = json!({ "specName": "x", "transactionVersion": 6, "apis": [] });
        assert!(matches!(
            parse_runtime_version(&rv),
            Err(PreflightError::Malformed(_))
        ));
    }

    /// The deployed v0.2.2-rc4 validator: spec 114, `QuantumPowApi` v2. This is
    /// the pairing the coordinator must accept.
    #[test]
    fn v0_2_2_rc4_runtime_is_accepted() {
        let info = parse_runtime_version(&rv_with(Some(2))).unwrap();
        assert!(info
            .quantum_pow_api
            .is_some_and(|v| v >= MIN_QUANTUM_POW_API));
    }

    #[test]
    fn error_messages_name_the_versions_and_the_remedy() {
        let info = Box::new(parse_runtime_version(&rv_with(Some(1))).unwrap());
        let too_old = PreflightError::ApiTooOld {
            info,
            required: MIN_QUANTUM_POW_API,
        }
        .to_string();
        assert!(too_old.contains("quip-runtime/114"), "{too_old}");
        assert!(too_old.contains("v1"), "{too_old}");
        assert!(too_old.contains("upgrade the validator"), "{too_old}");

        let missing =
            PreflightError::MissingApi(Box::new(parse_runtime_version(&rv_with(None)).unwrap()))
                .to_string();
        assert!(missing.contains("QuantumPowApi"), "{missing}");
        assert!(missing.contains("upgrade the validator"), "{missing}");
    }
}

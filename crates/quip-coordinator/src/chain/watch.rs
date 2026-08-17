//! Decoding `author_submitAndWatchExtrinsic` notifications.
//!
//! The status stream reports transport and pool progress only. It says whether
//! the extrinsic reached a block. It does not say whether the dispatch inside
//! that block succeeded, because that lives in the block's events. The outcome
//! module answers that question from chain state instead.

/// One notification from the transaction status subscription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxStatus {
    /// Accepted into the pool and ready for inclusion.
    Ready,
    /// Sent to peers.
    Broadcast,
    /// Included in the block with this hash.
    InBlock(String),
    /// Included in a finalized block with this hash.
    Finalized(String),
    /// Removed from the pool without inclusion. Worth resubmitting.
    Dropped(String),
    /// Rejected by the pool. Resubmitting the same bytes cannot help.
    Invalid(String),
    /// Held for a future nonce.
    Future,
    /// A status this build does not model, reported verbatim.
    Other(String),
}

/// Decode one status notification.
///
/// Unknown shapes become [`TxStatus::Other`] rather than an error, so a node
/// that adds a status does not break the submit loop.
#[must_use]
pub fn parse_tx_status(v: &serde_json::Value) -> TxStatus {
    if let Some(s) = v.as_str() {
        return match s.to_ascii_lowercase().as_str() {
            "ready" => TxStatus::Ready,
            "broadcast" => TxStatus::Broadcast,
            "future" => TxStatus::Future,
            _ => TxStatus::Other(s.to_string()),
        };
    }
    let Some(obj) = v.as_object() else {
        return TxStatus::Other(v.to_string());
    };
    for (k, val) in obj {
        let hash = val.as_str().unwrap_or_default().to_string();
        match k.to_ascii_lowercase().as_str() {
            "inblock" => return TxStatus::InBlock(hash),
            "finalized" => return TxStatus::Finalized(hash),
            // Usurped and retracted both mean this extrinsic is no longer on
            // track for inclusion, and both are worth another attempt.
            "dropped" | "usurped" | "retracted" | "finalitytimeout" => {
                return TxStatus::Dropped(format!("{k}: {hash}"))
            }
            "invalid" => return TxStatus::Invalid(format!("{k}: {hash}")),
            "broadcast" => return TxStatus::Broadcast,
            _ => {}
        }
    }
    TxStatus::Other(v.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn bare_string_statuses_are_recognised() {
        assert_eq!(parse_tx_status(&json!("ready")), TxStatus::Ready);
        assert_eq!(parse_tx_status(&json!("broadcast")), TxStatus::Broadcast);
        assert_eq!(parse_tx_status(&json!("future")), TxStatus::Future);
    }

    /// Substrate has used both camelCase and lowercase over time. Accept both
    /// so a node upgrade does not silently stop reporting inclusion.
    #[test]
    fn in_block_is_recognised_in_either_casing() {
        assert_eq!(
            parse_tx_status(&json!({"inBlock": "0xabc"})),
            TxStatus::InBlock("0xabc".into())
        );
        assert_eq!(
            parse_tx_status(&json!({"inblock": "0xabc"})),
            TxStatus::InBlock("0xabc".into())
        );
    }

    #[test]
    fn finalized_carries_its_block_hash() {
        assert_eq!(
            parse_tx_status(&json!({"finalized": "0xdef"})),
            TxStatus::Finalized("0xdef".into())
        );
    }

    #[test]
    fn terminal_rejections_are_distinguished_from_inclusion() {
        assert!(matches!(
            parse_tx_status(&json!({"dropped": null})),
            TxStatus::Dropped(_)
        ));
        assert!(matches!(
            parse_tx_status(&json!({"invalid": null})),
            TxStatus::Invalid(_)
        ));
        assert!(matches!(
            parse_tx_status(&json!({"usurped": "0x01"})),
            TxStatus::Dropped(_)
        ));
    }

    /// An unknown status must not be read as success. It is reported verbatim
    /// so an operator can see what the node actually said.
    #[test]
    fn an_unknown_status_is_reported_verbatim() {
        match parse_tx_status(&json!({"somethingNew": "0x02"})) {
            TxStatus::Other(s) => assert!(s.contains("somethingNew"), "{s}"),
            other => panic!("expected Other, got {other:?}"),
        }
    }
}

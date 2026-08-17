//! Enumerating the `QuantumComputeMempool.JobOrders` map.
//!
//! Order discovery used to read `JobProposed` events, which needs runtime
//! metadata and only sees the head block. Walking the storage map needs
//! neither. It also finds orders opened in earlier blocks, which the event
//! scan missed.

use super::extrinsic::twox128;

/// Length of the two `twox128` hashes that name a storage map.
const MAP_PREFIX_LEN: usize = 32;
/// Length of the `Blake2_128` hash that `Blake2_128Concat` puts before the key.
const BLAKE_HASH_LEN: usize = 16;

/// Storage prefix for every entry in `QuantumComputeMempool.JobOrders`.
#[must_use]
pub fn job_orders_prefix() -> Vec<u8> {
    let mut key = Vec::with_capacity(MAP_PREFIX_LEN);
    key.extend_from_slice(&twox128(b"QuantumComputeMempool"));
    key.extend_from_slice(&twox128(b"JobOrders"));
    key
}

/// Recover the order id from a full `JobOrders` storage key.
///
/// `Blake2_128Concat` stores the raw SCALE-encoded key after its hash, so the
/// trailing eight bytes are the little-endian `u64` order id.
///
/// Returns `None` when `key` is not a `JobOrders` key of the expected length.
#[must_use]
pub fn order_id_from_key(key: &[u8]) -> Option<u64> {
    let want = MAP_PREFIX_LEN + BLAKE_HASH_LEN + 8;
    if key.len() != want || !key.starts_with(&job_orders_prefix()) {
        return None;
    }
    let tail = key.get(want - 8..)?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(tail);
    Some(u64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::extrinsic::job_orders_storage_key;

    #[test]
    fn the_prefix_is_the_leading_half_of_a_full_order_key() {
        let prefix = job_orders_prefix();
        assert_eq!(prefix.len(), 32);
        let full = job_orders_storage_key(42);
        assert!(full.starts_with(&prefix), "prefix must lead the full key");
    }

    /// `Blake2_128Concat` appends the SCALE-encoded key, so the order id is
    /// recoverable from the key alone. That is what makes enumeration work.
    #[test]
    fn an_order_id_round_trips_through_its_storage_key() {
        for id in [0u64, 1, 42, u64::MAX] {
            let key = job_orders_storage_key(id);
            assert_eq!(order_id_from_key(&key), Some(id), "id {id}");
        }
    }

    #[test]
    fn a_key_that_is_too_short_yields_none() {
        assert_eq!(order_id_from_key(&[0u8; 8]), None);
    }

    #[test]
    fn a_key_from_a_different_map_yields_none() {
        // A key whose prefix does not match must not be misread as an order.
        let mut key = vec![0xAAu8; 32];
        key.extend_from_slice(&[0u8; 16]);
        key.extend_from_slice(&7u64.to_le_bytes());
        assert_eq!(order_id_from_key(&key), None);
    }
}

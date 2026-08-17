//! Reading `System.Account` over plain JSON-RPC.
//!
//! The funding gate needs one number: the free balance. Fetching it through a
//! metadata-aware client costs a second, stateful connection. The storage key
//! and the field offset are both stable, so this module builds the key and
//! reads the field directly.

use super::extrinsic::{blake2_128, twox128};

/// Byte offset of `data.free` inside a SCALE-encoded `AccountInfo`.
///
/// The leading fields are `nonce`, `consumers`, `providers`, and `sufficients`,
/// each a `u32`. `data.free` is the first field of `AccountData` and is a
/// `u128`.
const FREE_OFFSET: usize = 16;

/// Substrate storage key for `System.Account(account)`.
#[must_use]
pub fn system_account_storage_key(account: &[u8; 32]) -> Vec<u8> {
    let mut key = Vec::with_capacity(16 + 16 + 16 + 32);
    key.extend_from_slice(&twox128(b"System"));
    key.extend_from_slice(&twox128(b"Account"));
    // Blake2_128Concat: the 16-byte hash then the raw key.
    key.extend_from_slice(&blake2_128(account));
    key.extend_from_slice(account);
    key
}

/// Pull `data.free` out of a SCALE-encoded `AccountInfo`.
///
/// # Errors
/// Returns a message when the blob is shorter than the free field requires.
pub fn free_from_account_bytes(bytes: &[u8]) -> Result<u128, String> {
    let end = FREE_OFFSET + 16;
    let Some(slice) = bytes.get(FREE_OFFSET..end) else {
        return Err(format!(
            "System.Account blob is too short: {} bytes, need at least {end}",
            bytes.len()
        ));
    };
    let mut buf = [0u8; 16];
    buf.copy_from_slice(slice);
    Ok(u128::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `System` and `Account` twox128 hashes are fixed Substrate constants.
    /// Pinning them catches a broken hasher immediately.
    #[test]
    fn the_account_key_starts_with_the_known_system_account_prefix() {
        let key = system_account_storage_key(&[0u8; 32]);
        let prefix_bytes = key.get(..32).expect("key has a 32-byte pallet prefix");
        let mut prefix = String::new();
        for b in prefix_bytes {
            use std::fmt::Write as _;
            let _ = write!(prefix, "{b:02x}");
        }
        assert_eq!(
            prefix,
            "26aa394eea5630e07c48ae0c9558cef7b99d880ec681799c0cf30e8886371da9"
        );
    }

    /// `Blake2_128Concat` appends the raw key after the 16-byte hash, so the last
    /// 32 bytes must be the account itself.
    #[test]
    fn the_account_key_ends_with_the_raw_account() {
        let account = [7u8; 32];
        let key = system_account_storage_key(&account);
        assert_eq!(key.len(), 16 + 16 + 16 + 32);
        let tail = key
            .get(key.len() - 32..)
            .expect("key ends with the 32-byte account");
        assert_eq!(tail, &account[..]);
    }

    /// `AccountInfo` is nonce(u32), consumers(u32), providers(u32),
    /// sufficients(u32), then data.free as u128 little-endian.
    #[test]
    fn free_is_read_from_the_documented_offset() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&1_u32.to_le_bytes()); // nonce
        raw.extend_from_slice(&2_u32.to_le_bytes()); // consumers
        raw.extend_from_slice(&3_u32.to_le_bytes()); // providers
        raw.extend_from_slice(&4_u32.to_le_bytes()); // sufficients
        raw.extend_from_slice(&123_456_789_u128.to_le_bytes()); // data.free
        raw.extend_from_slice(&0_u128.to_le_bytes()); // data.reserved
        assert_eq!(free_from_account_bytes(&raw).unwrap(), 123_456_789);
    }

    #[test]
    fn a_short_account_blob_is_an_error_not_a_panic() {
        let err = free_from_account_bytes(&[0u8; 8]).unwrap_err();
        assert!(err.contains("too short"), "{err}");
    }
}

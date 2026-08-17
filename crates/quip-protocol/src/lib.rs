//! `QuIP` consensus SDK: wire codec, `ChaCha8` draw, energy scoring, diversity, and
//! `derive_nonce`.
//!
//! These primitives are golden-pinned for cross-language consensus. The Rust
//! implementations in this crate are the source of truth mirrored by the `PyO3`
//! bindings.

pub mod chacha8;
pub mod derive;
pub mod scoring;
pub mod session;
pub mod wire;

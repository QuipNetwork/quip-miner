//! Library facade for the mock coordinator conformance harness.
//!
//! Exposes [`driver`] for scripted miner sessions over a Unix domain socket.

/// Scripted mock coordinator that drives a miner through protocol conformance.
pub mod driver;

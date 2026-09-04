//! quip-coordinator: chain access, job production, routing, validation,
//! miner session server, and process supervision for the v0.3 protocol.

pub mod attempt;
pub mod chain;
pub mod config;
pub mod dashboard;
pub mod decay;
pub mod drive;
pub mod funding;
pub mod keygen;
pub mod liveness;
pub mod logging;
pub mod metrics;
pub mod presets;
/// `PoW` and mempool job construction helpers.
pub mod producer;
pub mod readiness;
pub mod round;
pub mod router;
pub mod runtime;
pub mod session;
pub mod stash;
pub mod supervisor;
pub mod survey;
pub mod timing;
pub mod topology;
pub mod validate;

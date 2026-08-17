//! CLI flags shared by every miner binary.
//!
//! Each binary defines its own `clap::Parser` struct that flattens
//! [`CommonArgs`] and adds any backend-specific flags (device index,
//! utilization ceiling, …).

use clap::Args;

/// Flags every miner accepts. Flatten into a binary's `Cli` with
/// `#[command(flatten)]`.
#[derive(Args, Debug, Clone)]
pub struct CommonArgs {
    /// Coordinator endpoint, e.g. `unix:///run/quip/coord.sock`.
    #[arg(long)]
    pub quip_coordinator: Option<String>,
    /// Miner id in Hello / Status. Defaults to `<backend>-0`.
    #[arg(long)]
    pub miner_id: Option<String>,
    /// Print the capabilities JSON and exit.
    #[arg(long)]
    pub capabilities: bool,
    /// Probe that the backend is runnable and exit.
    #[arg(long)]
    pub check: bool,
    /// Log level (accepted for compatibility; stderr is the default sink).
    #[arg(long, default_value = "info")]
    pub log_level: String,
    /// Sweeps per beta rung in the annealing schedule (>= 1). Overrides the
    /// default of 1 without going through the coordinator; e.g. raise it to
    /// converge Gibbs further. A miner-local setting for now (not on the wire).
    #[arg(long)]
    pub sweeps_per_beta: Option<usize>,
}

//! `quip-miner-exec`: a miner that shells out to a generic external solver.
//!
//! See [`quip_miner_exec`] for the solver contract. Handshake and session flow
//! are handled by [`quip_miner_core::run`]; this binary only parses the
//! solver-specific flags and builds an [`ExecSampler`].

use clap::Parser;
use quip_miner_core::{adapt::AdaptBounds, run, BackendIdentity, CommonArgs, OpenError};
use quip_miner_exec::ExecSampler;
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// External solver command template. `{model}` is replaced with the JSON
    /// model file path. Example: `--solver-cmd "my-solver --model {model}"`.
    /// Required for `--check` and session mode; not needed for
    /// `--capabilities`.
    #[arg(long)]
    solver_cmd: Option<String>,

    /// Per-job solver timeout in milliseconds. A solver that exceeds it is
    /// killed and the job rejected.
    #[arg(long, default_value_t = 30_000)]
    solver_timeout_ms: u64,

    /// Pipe the JSON model to the solver's stdin instead of a temp file. In
    /// this mode `--solver-cmd` need not contain `{model}`.
    #[arg(long)]
    solver_stdin: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(
        BackendIdentity {
            backend: "exec",
            algorithm: "external",
            max_nodes: 100_000,
            max_edges: 1_000_000,
            // The external solver's real sampling envelope is unknown, so the
            // coordinator adapts against conservative CPU-SA-like bounds.
            adapt: AdaptBounds {
                min_sweeps: 64,
                max_sweeps: 4096,
                min_reads: 64,
                max_reads: 512,
                reads_solution_min_factor: 4,
                reads_solution_max_factor: 8,
                reads_solution_floor_factor: 0,
            },
        },
        &cli.common,
        || {
            let cmd = cli
                .solver_cmd
                .as_deref()
                .ok_or_else(|| OpenError("--solver-cmd is required".to_string()))?;
            ExecSampler::new(cmd, cli.solver_timeout_ms, cli.solver_stdin).map_err(OpenError)
        },
    )
}

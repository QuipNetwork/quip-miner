//! Test-support miner with `max_reads() == 0`, so every job (`num_reads` ≥ 1) is
//! rejected `TooLarge`. Used by `tests/loop_conformance.rs` to exercise the
//! reads-cap path.

use clap::Parser;
use quip_miner_core::{
    run, BackendIdentity, CommonArgs, IsingGraph, SampleParams, Sampler, SamplerResult,
};
use quip_proto::v1::RejectReason;
use std::process::ExitCode;

struct CappedSampler;

impl Sampler for CappedSampler {
    fn sample(
        &self,
        _graph: &IsingGraph,
        _params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, RejectReason> {
        // Unreachable in practice: the harness rejects TooLarge before sampling.
        Ok(Vec::new())
    }

    fn max_reads(&self) -> u32 {
        0
    }
}

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(
        BackendIdentity {
            backend: "mock",
            algorithm: "sa",
            max_nodes: 100_000,
            max_edges: 1_000_000,
            adapt: quip_miner_core::adapt::AdaptBounds {
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
        || Ok(CappedSampler),
    )
}

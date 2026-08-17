//! Test-support miner: runs the core loop with a trivial sampler that returns
//! `num_reads` all-`+1` solutions, scored with the consensus scorer. Used by
//! `tests/loop_conformance.rs` to exercise the session loop without a backend.

use clap::Parser;
use quip_miner_core::{
    run, BackendIdentity, CommonArgs, IsingGraph, SampleParams, Sampler, SamplerResult,
};
use quip_proto::v1::RejectReason;
use quip_protocol::scoring::energy_milli;
use std::process::ExitCode;

struct MockSampler;

impl Sampler for MockSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, RejectReason> {
        let spins = vec![1i8; graph.num_nodes()];
        let energy = energy_milli(&spins, &graph.h, &graph.j, &graph.edges);
        Ok((0..params.num_reads)
            .map(|_| SamplerResult {
                spins: spins.clone(),
                energy_milli: energy,
            })
            .collect())
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
        || Ok(MockSampler),
    )
}

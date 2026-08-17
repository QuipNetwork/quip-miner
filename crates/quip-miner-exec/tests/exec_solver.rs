//! End-to-end `ExecSampler` tests driving a real subprocess. Unix-only: the
//! fake solvers are `sh` one-liners.
#![cfg(unix)]

use quip_miner_core::{IsingGraph, SampleParams, Sampler};
use quip_miner_exec::ExecSampler;
use quip_proto::v1::RejectReason;

/// A 2-node problem, so valid solutions have `spins.len() == 2`.
fn graph() -> IsingGraph {
    IsingGraph::new(vec![1.0, -1.0], vec![1.0], vec![(0, 1)])
}

const ONE_SOLUTION: &str = r#"[{\"spins\":[1,-1],\"energy_milli\":-5}]"#;

#[test]
fn file_mode_delivers_model_and_parses_solutions() {
    // The solver only emits a solution when a non-empty model file is present at
    // the substituted path, proving file delivery.
    let cmd =
        format!("sh -c 'test -s \"$1\" && echo \"{ONE_SOLUTION}\" || echo \"[]\"' _ {{model}}");
    let sampler = ExecSampler::new(&cmd, 5_000, false).expect("valid cmd");
    let out = sampler
        .sample(&graph(), &SampleParams::default())
        .expect("ok");
    assert_eq!(out.len(), 1);
    #[expect(
        clippy::indexing_slicing,
        reason = "len asserted to 1; single solution at index 0"
    )]
    {
        assert_eq!(out[0].spins, vec![1i8, -1]);
        assert_eq!(out[0].energy_milli, -5);
    }
}

#[test]
fn stdin_mode_delivers_model_on_stdin() {
    // `cat` drains stdin to EOF (newline-independent); a non-empty capture
    // proves the model arrived on stdin.
    let cmd =
        format!("sh -c 'in=$(cat); test -n \"$in\" && echo \"{ONE_SOLUTION}\" || echo \"[]\"'");
    let sampler = ExecSampler::new(&cmd, 5_000, true).expect("valid cmd");
    let out = sampler
        .sample(&graph(), &SampleParams::default())
        .expect("ok");
    assert_eq!(out.len(), 1);
}

#[test]
fn nonzero_exit_rejects_malformed() {
    let sampler = ExecSampler::new("sh -c 'exit 1' _ {model}", 5_000, false).expect("valid cmd");
    assert_eq!(
        sampler.sample(&graph(), &SampleParams::default()),
        Err(RejectReason::Malformed)
    );
}

#[test]
fn timeout_rejects_overloaded() {
    let sampler = ExecSampler::new("sh -c 'sleep 10' _ {model}", 150, false).expect("valid cmd");
    assert_eq!(
        sampler.sample(&graph(), &SampleParams::default()),
        Err(RejectReason::Overloaded)
    );
}

#[test]
fn missing_solver_binary_rejects_overloaded() {
    let sampler =
        ExecSampler::new("quip-no-such-solver-xyz {model}", 5_000, false).expect("valid cmd");
    assert_eq!(
        sampler.sample(&graph(), &SampleParams::default()),
        Err(RejectReason::Overloaded)
    );
}

# AGENTS.md — QuIP Protocol Project Instructions

Cross-tool instructions for AI coding assistants (Claude Code, Codex, Cursor, Gemini CLI).

QuIP v0.3 is a Rust workspace. The coordinator owns chain access and work routing. It spawns and supervises the miner subprocesses that do the sampling. The miner binaries ship from their own repos. The solver contract (wire protocol, consensus primitives, solver harness) lives in `quip.network/quip-solver-core` and is consumed from crates.io at v0.0.0.

For deeper detail, see the companion guides: `COORDINATOR.md` covers how the coordinator works, and `NEWMINER.md` covers adding a new miner. The solver contract itself is specified in the quip-solver-core repo (`SPEC.md`).

## Repository layout

- `crates/` — the Cargo workspace (root manifest: `Cargo.toml`):
  - `quip-coordinator` — the coordinator binary: chain access, routing, miner supervision.
  - `quip-miner-exec` — a miner that shells out to a generic external solver.
  - `quip-mock-miner` — the miner test double the coordinator tests spawn.
- `docker/` — image builds (`Dockerfile.quip-miner`, `Dockerfile.quip-miner-cuda`) and `config.toml`.
- `dwave_topologies/` — D-Wave embedding data.
- `docs/VERSIONING.md` — the release-tag standard.

The miner binaries live in their own repos: `quip.network/quip-miner-{cpu,cuda,metal,dwave}`. The `quip-proto` (wire types), `quip-protocol` (consensus math), and `quip-solver-core` (`Sampler` trait + session loop) crates come from crates.io; their source repo is `quip.network/quip-solver-core`, which also publishes the `quip-solver-core` Python wheel the D-Wave miner uses.

## CUDA card support requirement

The CUDA miner supports NVIDIA GPUs with compute capability 7.0 through 12.1
(the kernels need sm_70 for `__nanosleep`; NVRTC 12.9 supplies the 12.1
ceiling), host driver from a CUDA 12.9 branch (r535+). The cudarc pin in
`quip-miner-cuda` (`cuda-12090`) and the runtime image's `cuda-nvrtc-12-9`
package must move together — the CUDA major of the pin decides which
`libnvrtc.so.N` the miner dlopens, and `docker/Dockerfile.quip-miner-cuda`
asserts the load at build time. Widening or narrowing card support is a
reviewed change to `SUPPORTED_ARCHS` in `quip-miner-cuda`
(`tests/arch_coverage.rs` enforces it), not a side effect of a toolkit bump.

## Build and test

The workspace builds from the repo root. The toolchain is pinned to 1.97.1 (`rust-toolchain.toml`).

```bash
cargo build --workspace

# Test (mirrors CI; the CI test job excludes quip-coordinator, which has its own job).
cargo test --workspace --exclude quip-coordinator
cargo test -p quip-coordinator

# Lint and format.
cargo clippy --workspace --exclude quip-coordinator --all-targets -- -D warnings
cargo clippy -p quip-coordinator --all-targets -- -D warnings
cargo fmt --all --check
```

## Running the coordinator

`quip-coordinator` runs from a config file, or as a subcommand.

```bash
# Production: read the config, connect to the validators, spawn and supervise the
# configured miners, feed them work, submit proofs. Stops on SIGINT/SIGTERM.
quip-coordinator --config ./docker/config.toml

# Drive: spawn one miner and feed it synthetic problems. No chain, no submit —
# for benchmarking and matched-condition parity runs.
quip-coordinator drive --miner ./miners/quip-cpu-sa --source random \
  --topology-preset advantage2-system1 --count 100

# Seed a fresh chain: register the default topology and set its difficulty.
# Give exactly one of --sudo-key or --mnemonic-file.
quip-coordinator seed-chain --sudo-key //Alice
```

`seed-chain` defaults to `--validator ws://quip-validator:9944` and
`--topology-preset advantage2-system1`. Difficulty defaults are
`--min-solutions 5`, `--max-energy-milli -2500000`, and
`--min-diversity-milli 200`. `--sudo-key` accepts a `//DevUri`, a BIP39
mnemonic, a 32-byte hex master seed, or a keystore path. `--mnemonic-file`
reads a BIP39 phrase from a file. Pass `--topology` to use a specification JSON
file instead of a preset.

You can seed a chain only once. `register_topology` writes `DefaultTopology`
only when that value is unset. `seed-chain` refuses to run when a chain already
has a default topology. Wipe the chain data and restart the validator instead.

The coordinator binary embeds the `advantage2-system1` and `smoke` topology
presets. `drive --topology-preset` and `seed-chain` read those presets from the
binary.

The config registers miners and their launch plan. Each backend section (`[cpu]`, `[cuda.0]`, `[metal]`, `[dwave]`) becomes one supervised subprocess; `binary` selects the executable — `quip-cpu-sa`, the chromatic `quip-cpu-gibbs`, `quip-cpu-sb`, or one of quip-miner-cpu's experimental kernels when that release published it. See `crates/quip-coordinator/config.toml.example`. Miner binaries are fetched per host by `crates/quip-coordinator/tools/fetch-miners.sh` from the standalone repos and resolved on `PATH` or by absolute path.

`signer_key` accepts a keystore path, a 32-byte hex master seed, or a `//DevUri`
such as `//Alice`. It also accepts any substrate secret URI, including a BIP39
mnemonic phrase. `.env` holds credentials such as `DWAVE_API_KEY` — **never read or display its contents.**

## The Python wheel

This repo holds no Python. The `quip-solver-core` PyPI wheel (consensus primitives + generated gRPC stubs) ships from `quip.network/quip-solver-core`, and the D-Wave miner repo depends on it. Report wheel or stub issues there.

## Conventions

- Rust standards and lints follow the `lang-rust` guidance; `cargo clippy -- -D warnings` is the gate.
- Follow the existing crate layout — shared code goes in the crate that owns it, not a new top-level module.
- No `Co-Authored-By` or other assistant-attribution trailers in commits.
- Never read `.env`.

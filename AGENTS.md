# AGENTS.md — QuIP Protocol Project Instructions

Cross-tool instructions for AI coding assistants (Claude Code, Codex, Cursor, Gemini CLI).

QuIP v0.3 is a Rust workspace. The coordinator owns chain access and work routing; it spawns and supervises the miner subprocesses that do the sampling. The miner binaries ship from their own repos. A small PyO3 extension exposes the consensus primitives to Python.

For deeper detail, see the companion guides: `COORDINATOR.md` covers how the coordinator works, `MINER.md` covers the shared miner harness, and `NEWMINER.md` covers adding a new miner.

## Repository layout

- `crates/` — the Cargo workspace (root manifest: `Cargo.toml`):
  - `quip-proto` — protobuf/gRPC types generated from `proto/` (prost/tonic).
  - `quip-protocol` — consensus primitives: scoring, wire codec, ChaCha8 draw, session types. The source of truth for the math.
  - `quip-protocol-py` — PyO3 `cdylib` exposing `quip-protocol` as `quip_proto._core`. Built by maturin, not the default cargo build.
  - `quip-miner-core` — shared miner harness: the gRPC session loop and the `Sampler` trait each backend supplies.
  - `quip-coordinator` — the coordinator binary: chain access, routing, miner supervision.
  - `quip-mock-coordinator`, `quip-mock-miner` — test doubles.
- `python/` — the `quip_proto` SDK (maturin `python-source`): the `_core` re-export shim plus the generated `quip.v1` gRPC stubs.
- `proto/` — the normative `.proto` IDL.
- `conformance/` — pytest suite pinning the PyO3 SDK to golden vectors.
- `docker/` — image builds (`Dockerfile.quip-miner`, `Dockerfile.quip-miner-cuda`) and `config.toml`.
- `dwave_topologies/` — D-Wave embedding data.
- `docs/VERSIONING.md` — the release-tag standard.

The miner binaries live in their own repos: `quip.network/quip-miner-{cpu,cuda,metal,dwave}`.

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
# Build every crate except the PyO3 cdylib (maturin builds that one).
cargo build --workspace --exclude quip-protocol-py

# Test (mirrors CI; the CI test job also excludes quip-coordinator).
cargo test --workspace --exclude quip-coordinator --exclude quip-protocol-py

# Lint and format.
cargo clippy --workspace --exclude quip-coordinator --exclude quip-protocol-py --all-targets -- -D warnings
cargo fmt --all --check
```

The Python SDK and its conformance suite need maturin:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop -E dev     # builds quip_proto._core, installs quip_proto + dev extras
pytest conformance/        # 19 tests
```

The PyO3 crate is a workspace member held out of `default-members`, so a bare `cargo build`/`cargo test` skips it. Build it with maturin, or target it directly with `cargo build -p quip-protocol-py`.

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

The config registers miners and their launch plan. Each backend section (`[cpu]`, `[cuda.0]`, `[metal]`, `[dwave]`) becomes one supervised subprocess; `binary` selects the executable — `quip-cpu-sa`, or the chromatic `quip-cpu-gibbs`. See `crates/quip-coordinator/config.toml.example`. Miner binaries are fetched per host by `crates/quip-coordinator/tools/fetch-miners.sh` from the standalone repos and resolved on `PATH` or by absolute path.

`signer_key` accepts a keystore path, a 32-byte hex master seed, or a `//DevUri`
such as `//Alice`. It also accepts any substrate secret URI, including a BIP39
mnemonic phrase. `.env` holds credentials such as `DWAVE_API_KEY` — **never read or display its contents.**

## The PyO3 SDK

`quip_proto` is the only Python surface. It re-exports the Rust consensus primitives (`scoring`, `wire`, `ExitCode`) from `quip_proto._core` and the generated gRPC stubs from `quip.v1`. The Rust in `quip-protocol` is the source of truth; the Python can't drift from it. The D-Wave miner repo depends on this wheel.

Regenerate the `quip.v1` stubs from `proto/` with the pinned `grpcio-tools==1.82.1`; CI diffs the result against the checked-in copies, so a version bump and a stub regen go together.

## Conventions

- Rust standards and lints follow the `lang-rust` guidance; `cargo clippy -- -D warnings` is the gate.
- Follow the existing crate layout — shared code goes in the crate that owns it, not a new top-level module.
- No `Co-Authored-By` or other assistant-attribution trailers in commits.
- Never read `.env`.

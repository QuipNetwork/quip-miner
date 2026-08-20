# Architecture

This repository is the v0.3 mining stack for the
[quip-protocol-rs](https://gitlab.com/quip.network/quip-protocol-rs) chain. It
holds the coordinator: it follows the chain, stages Ising problems, and
submits proofs. The solver contract — the wire protocol, the consensus
primitives, and the shared solver harness — lives in
[quip-solver-core](https://gitlab.com/quip.network/quip-solver-core) and is
consumed from crates.io. The miners themselves live in their own repositories.
Consensus rules live in the chain pallet. This document maps how the pieces
fit and names the seams that keep them apart.

For task-level detail, read the two guides this document ties together:
`COORDINATOR.md` (the coordinator internals) and `NEWMINER.md` (adding a
backend). The solver side is specified in quip-solver-core's `SPEC.md`.
`docs/VERSIONING.md` covers release tags.

## Crate layout

The workspace is a strict dependency DAG under `crates/`. Lower crates never
import higher ones.

| Crate | Role | Depends on |
|-------|------|-----------|
| `quip-miner-exec` | Generic external-solver miner: JSON model + exec over the session protocol | `quip-solver-core` |
| `quip-coordinator` | The `quip-coordinator` binary: chain access, feeder, router, supervisor | `quip-proto`, `quip-protocol` |
| `quip-mock-miner` | Miner test double | `quip-proto`, `quip-protocol` |

Three dependencies come from crates.io at v0.0.0, published from the
quip-solver-core repository: `quip-proto` (generated protocol types and gRPC
service), `quip-protocol` (consensus primitives: `wire`, `session`, `scoring`,
`derive`, `chacha8`), and `quip-solver-core` (the `Sampler` trait, session
client, and adaptive params — it re-exports the other two).

`quip-coordinator` does not depend on `quip-solver-core`. The two sides share
nothing but the wire protocol (`quip-proto`) and the codec and session
primitives (`quip-protocol`). This is the central decision of the v0.3 design.
A miner is any process that speaks the protocol. Miners ship from separate
repositories, and the D-Wave miner ships from Python over the
`quip-solver-core` PyPI wheel.

## The five seams

The architecture rests on five boundaries. The source references below back
each one.

### 1. One blockchain touchpoint

Every chain interaction sits in `crates/quip-coordinator/src/chain/` behind the
`ChainClient` trait (`chain/mod.rs:43`). The trait has three methods:
`fetch_mining_snapshot`, `fetch_mempool_orders`, and `submit_proof`. Two types
provide it: `RealChainClient` over subxt and JSON-RPC (`chain/real.rs:103`)
and `FakeChain` for tests (`chain/fake.rs:44`). `subxt` appears in one source
file, `chain/real.rs`. The module owns everything downstream of the seam:
extrinsic building, SCALE types, proof encoding, snapshot decoding, mempool
decoding, and submission. Nothing outside `chain/` talks to the node.

### 2. Feeder and per-miner dispatch

Job staging runs through three files that form one control path. `producer/`
builds jobs: `pow.rs` derives the proof-of-work job from the chain snapshot,
`mempool.rs` converts fetched `JobOrder`s into wire `Job`s. `runtime::feeder_loop`
(`runtime.rs:100`) follows the chain head, bumps a generation on each new
`last_proof_block_hash`, and stages jobs per miner using an adaptive window read
off the router's drain rate (`runtime.rs:150`). `router.rs` holds the
capability-indexed staged queues and the dispatch credits. The target design
named a single module here; the code splits it into producer, feeder,
and router. The split is deliberate: the router is a reusable queue primitive,
and the feeder is the policy on top of it.

### 3. Spawn, kill, and the protocol kill line

`supervise_miner` (`supervisor.rs:219`) starts each miner: it spawns the binary
with a fresh per-spawn session token and applies the exit-code restart policy
(`restart_policy`, `supervisor.rs:25`). Clean exits respawn on demand,
operator and environment errors (64, 69, 77) never respawn, and crashes back off
by `2^consecutive` within a failure budget. `shutdown_all` (`supervisor.rs:176`)
ends the run: it sends an in-band `Shutdown`, waits the grace period, then kills
any survivor. The protocol kill line is the `Shutdown` and `Cancel` control
messages. The coordinator emits them (`session.rs:391` `send_cancel`,
`session.rs:896` `shutdown_msg`); the `quip-solver-core` harness honors them
and drains in-flight work (its session loop and `SPEC.md` section 5 cover the
cancel semantics).

### 4. The generic miner harness

`quip-solver-core` (crates.io, from the quip-solver-core repository) is the
standard interface every miner builds on. A backend supplies a `Sampler` (the
`sample` method is required; streaming, throttling, and config hooks have
defaults). The crate's `run` entry point handles the `--capabilities`,
`--check`, and `--solve` modes and drives the gRPC session loop. The crate is
library-only, with no binary of its own.

The CPU, CUDA, and Metal miners are native binaries in their own repositories
that call `run`. The D-Wave miner is Python, built on the `quip-solver-core`
PyPI wheel. For a backend that is a standalone executable rather than a
Rust `Sampler`, `quip-miner-exec` is the plug-in point: it serializes each job
to a JSON model, execs a configured external solver (by file or stdin), and
parses the solver's JSON solutions back over the same session protocol. One part
of the target design still differs from the source:

- The random and file job sources (the driver for benchmarking without a chain)
  live coordinator-side under `drive/` (`drive/random_source.rs`,
  `drive/list_source.rs`, `drive/harness.rs`), reached through the `drive`
  subcommand. They generate jobs, so the coordinator is their natural home, and
  the miner stays a pure solver.

### 5. Miners are isolated solvers

`quip-solver-core` carries no chain or consensus logic. From `quip-protocol`
it imports the wire codec and the session handshake, nothing else. A miner
cannot reach the node, directly or transitively. It receives an Ising problem,
samples it, and returns spins with energies. This isolation is why the
contract could move to its own repository: the solver side depends on the
chain through nothing but the wire.

## Job lifecycle

A proof-of-work job flows through the system in one pass:

1. `feeder_loop` calls `ChainClient::fetch_mining_snapshot` and reads the
   difficulty target, topology, and the last proof block hash.
2. On a new block hash, the feeder bumps the generation and cancels the prior
   one, so miners stop working on stale problems.
3. `producer::derive_pow_job` builds the job from the snapshot. Mempool orders
   take a parallel path through `job_order_to_job` and carry `generation = 0`,
   since a chain reorg doesn't cancel a user-submitted order.
4. The feeder stages jobs on each miner's queue up to the adaptive depth. The
   router dispatches them and tracks a dispatch-to-completion credit.
5. The miner samples and streams results back over the session.
6. `validate.rs` scores the returned set. A solution is energy-valid when its
   milli-energy is strictly below the gate floor. The coordinator accepts the
   set when `n_valid >= min_solutions` and `diversity_milli >= min_diversity_milli`
   (`validate.rs:143`).
7. On an accepted set, `chain/submit.rs` encodes and submits the
   `QuantumPow.submit_proof` extrinsic through `ChainClient`.

Jobs carry a `job_id`, a `generation`, an optional `deadline_ms`, the Ising
problem (edge list or CSR, plus little-endian `h` and `j` fields), and a
`provenance` marking proof-of-work versus mempool with the source `order_id`.
The identifiers that matter are `job_id`, the mempool `order_id`, and the
`generation` counter that gates cancellation.

## Parameters and where they live

The harness owns the adaptive-parameter mechanism; the layer that knows the
concrete values supplies them.

`quip-solver-core`'s `adapt` module holds the ground-state-energy model that
turns a difficulty target into `num_reads` and `num_sweeps`. The
`quip-solver-conformance` crate's adapt vectors pin cross-language parity.

Per-backend bounds are an `AdaptBounds` struct, not a fixed table in this
repository. Each miner binary constructs its own; `quip-miner-exec` uses
conservative CPU-SA-like bounds (`crates/quip-miner-exec/src/main.rs`). The
CUDA, Metal, and D-Wave bounds live in their own repositories.

Genesis and consensus defaults live in the chain pallet, not here. The
coordinator reads them from the mining snapshot. The devnet acceptance test
exercises the defaults it expects to receive: `min_solutions = 5` and
`min_diversity = 0.200` (`n = 5`, `n_milli = 200` in
`tests/devnet_submit.rs`). For the difficulty energy target, the per-topology
energy ranges, and the mempool section-election rules, read the pallet in
quip-protocol-rs. This repository consumes those values across the `ChainClient`
seam and doesn't define them.

## Testing and the doubles

Each side of the protocol has a test double. The coordinator side is
`quip-solver-conformance` (in the quip-solver-core repository): a scripted
coordinator, driven by the `quip-solver-drive` binary, that can drive a real
miner binary and inject malformed handshakes. The miner side is this
workspace's `quip-mock-miner`, the double the coordinator's integration tests
spawn. Keeping both independent of the production serve loop is what lets
them script failures the real path would never produce.

The `drive` subcommand runs the coordinator against synthetic problems with no
chain, using `FakeChain` and the `drive/` job sources. It's the fast offline
path for load and behavior testing.

## Build and run

The Rust workspace builds from the repository root:

```bash
cargo build --workspace
cargo test  --workspace
```

See `AGENTS.md` for the full command set.

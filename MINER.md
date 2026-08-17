# Miner core — the shared mining and helper modules

`quip-miner-core` (crate `crates/quip-miner-core`) is the harness every miner
binary is built on. A backend provides one thing — a `Sampler` that solves Ising
problems — and calls `run`. The harness owns the gRPC session,
the handshake, credits, job validation, adaptive parameters, reject reasons,
status, liveness, and exit codes. The CPU, CUDA, and Metal miners are thin
binaries over this crate; the D-Wave miner uses the same protocol from Python.

This document describes the modules. For the wire contract itself — the messages
and the behaviors the coordinator expects — see `MINER_PROTOCOL.md`. To add a
backend, see `NEWMINER.md`.

## The `Sampler` trait

`Sampler` (in `lib.rs`) is the one interface a backend fills in. Only `sample`
is required:

```rust
fn sample(&self, graph: &IsingGraph, params: &SampleParams)
    -> Result<Vec<SamplerResult>, RejectReason>;
```

The rest have defaults that describe a serial, uncapped CPU-shaped backend:

- `sample_stream` — pull jobs from a channel, keep up to `stream_width` in
  flight, emit each result in completion order. The default is a serial loop
  over `sample`. GPU backends override it to run a batch of models at once.
- `stream_width` — how many models the backend keeps in flight (default 1).
- `utilization` — a load figure for `Status` (default 0.0).
- `should_throttle` — whether to back off before the next job (default false).
- `max_reads` — the largest `num_reads` the backend accepts; a device-memory
  bound overrides the default of no cap. This is a reject threshold, not a
  silent clamp.
- `apply_config` — apply this backend's `Configure.backend_toml` subsection
  once, before any job (default: no settings).

`StreamJob` and `StreamResult` are the channel messages; `StreamResult` carries
the per-model device time reported in `SamplerMeta`.

## The `run` entry point

A binary calls `run(id, common, open)` (in `session.rs`):

- `id: BackendIdentity` — the backend name and algorithm, advertised in `Hello`
  and `--capabilities`.
- `common: &CommonArgs` — the shared CLI flags.
- `open: impl FnOnce() -> Result<S, OpenError>` — opens the device and builds
  the `Sampler`.

`run` installs the log subscriber first, then dispatches on the flags:

- `--capabilities` prints the capabilities JSON and exits without opening the
  device.
- `--check` opens the device as a runnable probe and exits (success, or
  `EnvIncompatible`).
- otherwise it enters session mode: it needs `--quip-coordinator <uri>`,
  defaults the miner id to `<backend>-0`, opens the device, starts a Tokio
  runtime, and runs `run_session`. Errors map to the shared exit codes.

## Logging

`logging.rs` installs the subscriber, and `run` calls it before anything else.
The `tracing` macros only dispatch to an installed subscriber, so without that
call every log statement in the miner does nothing, including the fatal-error
paths.

Verbosity comes from `--log-level {trace,debug,info,warn,error}`, default
`info`, which the coordinator passes to every miner it spawns. A non-empty
`RUST_LOG` overrides the flag. A miner inherits the coordinator's environment,
so `RUST_LOG` reaches the children too. A level outside that set fails at
startup instead of falling back to a default. `EnvFilter` reads an unknown word as
a target name, so a typo would otherwise parse cleanly and silence the miner.

Output goes to stderr. A supervised miner inherits the coordinator's streams,
so the coordinator's log receives its lines with no forwarding step. Stdout
carries only the `--capabilities` JSON.

What each level carries:

- `info` — one line per finished attempt: best energy, how many solutions clear
  the session target, the target itself, the resolved reads and sweeps, and
  wall and device time. The miner logs a losing attempt the same as a winning
  one, which separates a healthy miner that is not winning from a stalled one.
  A progress line follows every ten jobs.
- `debug` — one line per job received (size and resolved parameters), and
  cancellations.
- `warn` — rejects, with the reject reason.

## The session loop

`run_session` is the gRPC client against the coordinator's `MinerService`. It
runs the handshake (`Hello` → `Welcome`), applies `Configure` through the
backend's `apply_config`, then requests work and services the bidirectional
stream. It grants the coordinator `stream_width` credits up front and one per
finished job, feeds incoming jobs to the backend through `sample_stream`, and
returns each outcome as a `Result` or a `Reject`. It answers `Ping` and `Cancel`
with `Status`, honors an in-band `Shutdown`, and ends when the coordinator closes
the stream — there is no wall-clock idle timeout, since a miner grinding a hard
nonce legitimately receives nothing meanwhile. Because the harness owns all of
this, a backend never touches gRPC.

## Base problem types

`ising.rs` holds the types every backend shares:

- `IsingGraph` — the wire-parsed problem: dense biases `h`, flat couplings `j`,
  and an `edges` list. Each backend derives its own view from it (the CPU miner an
  adjacency list, the GPU miners a `CsrGraph`).
- `SampleParams` — the per-job knobs: `num_reads`, `num_sweeps`,
  `sweeps_per_beta`, an optional explicit `beta_range`, and a `seed`.
- `SamplerResult` — one completed read: `spins` in {-1, +1} and the consensus
  `energy_milli`.
- `Algorithm` — `Sa` (Metropolis simulated annealing) or `Gibbs` (single-site
  heat-bath).

## Job handling

`job.rs` sits between the wire and the `Sampler`. `parse_ising` decodes the
`IsingProblem` message into an `IsingGraph`; `prepare_job` validates it and
builds the `SampleParams`; `finalize_result` turns the backend's
`SamplerResult`s back into the wire result.

Parameter precedence runs through `pick_param`: a per-job value wins first, then
a `SetTarget` pin from the coordinator, then the miner's own adaptive value,
then the default — first non-zero wins. A value pinned through `SetTarget`
(`--num-reads`/`--num-sweeps`) bypasses adaptation — the drive path uses this
for matched-condition runs — while a normal chain job lets the miner adapt.

## Adaptive parameters

`adapt.rs` maps a difficulty target to a sampling budget. The coordinator
advertises a target energy; the miner computes its own `num_reads` and
`num_sweeps` from it. `energy_to_difficulty` turns the target into a normalized
difficulty in `[0, 1]` using a ground-state-energy model (`GSE ≈
-c·√(avg_degree)·N` plus an h-field term). `adapt_params` then scales the budget
between per-backend `AdaptBounds`.

The bounds differ by device: CPU/SA spans 64–4096 sweeps and 64–512 reads and
scales reads with `min_solutions`; CUDA spans 256–2048 sweeps and 64–256 reads
with fixed read bounds. This math is a port of the earlier Python
`energy_utils.py`, and cross-language parity is pinned by
`conformance/golden_adapt.json` — the Rust tests replay the same golden cases.

## Beta schedule

`beta.rs` builds the annealing temperature ladder.
`default_ising_beta_range` derives a hot and cold beta from the graph's
effective-field magnitudes when a job gives no explicit range: hot so the
worst-case flip is about 50%, cold for a low excitation rate on the smallest
gap. `geometric_beta_schedule` returns a geometric ladder from hot to cold in
`f64`; the CPU miner uses it directly, and the GPU backends cast each element to
`f32` at upload. The f64 math with a per-element cast is bit-identical to the
earlier per-backend code.

## CSR graph for GPU backends

`csr.rs` holds `CsrGraph`, the compressed-sparse-row form the GPU backends build
from an `IsingGraph` (`CsrGraph::from_base`). It stores each coupling once per
directed half-edge, so a local-field walk is O(degree). Kernel buffers are
`f32` (`h_f32`, `j_csr`); it keeps `f64` copies of `h`/`j` for consensus
scoring, so the score matches the CPU path.

## Config and CLI helpers

`config.rs` is the shared `apply_config` discipline so every backend handles its
`backend_toml` the same way. `config_override` resolves one setting with
config-over-CLI precedence and warns only when the config actually changes a
value; `warn_unknown_fields` surfaces a typo'd key rather than dropping it
silently. `SESSION_KEYS` lists the keys the session loop consumes (such as
`num_sweeps`), so a backend won't warn about them.

`cli.rs` holds `CommonArgs`, the flags every binary accepts —
`--quip-coordinator`, `--miner-id`, `--capabilities`, `--check`, `--log-level`,
`--sweeps-per-beta`. A binary flattens it into its own `clap` parser with
`#[command(flatten)]` and adds any device-specific flags.

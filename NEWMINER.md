# Adding a new miner

A miner is a standalone program that solves Ising problems and speaks the gRPC
miner protocol to the coordinator. You can add one two ways:

- **A native backend on `quip-miner-core`** — the shape the CPU, CUDA, and Metal
  miners use. You write a `Sampler` and the harness does the rest. This is the
  recommended path and most of this guide.
- **A client in another language** — speak the protocol directly with the
  `quip_proto` SDK. This is the shape the D-Wave miner uses from Python. See the
  last section.

Read `MINER.md` first for the harness API this guide builds on.

## A native backend

### 1. Start a crate

A miner lives in its own repository (the released ones are
`quip.network/quip-miner-{cpu,cuda,metal}`). Depend on `quip-miner-core` as a
git dependency and on `quip-proto` for the wire types. The crate is a binary,
not a workspace member of `quip-protocol`.

### 2. Write the `Sampler`

Only `sample` is required:

```rust
use quip_miner_core::{IsingGraph, SampleParams, SamplerResult, Sampler};
use quip_proto::v1::RejectReason;

struct MyBackend { /* device handle */ }

impl Sampler for MyBackend {
    fn sample(&self, graph: &IsingGraph, params: &SampleParams)
        -> Result<Vec<SamplerResult>, RejectReason>
    {
        // Solve `graph` `params.num_reads` times; return one SamplerResult per
        // read (spins in {-1,+1} and the consensus energy_milli). Map a device
        // failure to a RejectReason (Overloaded, TooLarge, …).
        todo!()
    }
}
```

Override the other trait methods only where your device differs from a serial
CPU:

- `sample_stream` and `stream_width` — run a batch of models at once instead of
  the default serial loop.
- `utilization` and `should_throttle` — report load and apply backpressure if
  the backend runs a governor.
- `max_reads` — reject a `num_reads` above a device-memory bound.
- `apply_config` — read this backend's settings from `Configure.backend_toml`.

Use `SamplerResult.energy_milli` from the shared scoring path so the score
matches every other backend. GPU backends build a `CsrGraph` and a beta ladder
from the helpers in `quip-miner-core`; a CPU-style backend can use `IsingGraph`
directly.

### 3. Write the binary

Flatten `CommonArgs` into your `clap` parser, add any device flags, build a
`BackendIdentity`, and call `run`:

```rust
use clap::Parser;
use quip_miner_core::{run, BackendIdentity, CommonArgs, OpenError};

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long)]
    device_index: Option<u32>,
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    let id = BackendIdentity { backend: "mybackend".into(), algorithm: "sa".into() };
    run(id, &cli.common, || open_device(&cli).map_err(OpenError))
}
```

`run` handles `--capabilities` and `--check` itself, and in session mode it
opens the device, connects to `--quip-coordinator`, and runs the whole session.
The coordinator passes the session token in the environment; the harness reads
it during the handshake, so your binary never handles authentication.

### 4. Configuration

The coordinator forwards each miner its own `config.toml` subsection verbatim in
`Configure.backend_toml`. Parse it in `apply_config` against your own schema and
use the shared helpers: `config_override` for config-over-CLI precedence with a
warning only on a real change, and `warn_unknown_fields` to surface a typo'd key
rather than dropping it. The session loop consumes a few keys itself (listed in
`SESSION_KEYS`), so don't warn about those.

## Naming and algorithm variants

Name the binary `quip-<backend>-<algorithm>` — `quip-mybackend-sa` for simulated
annealing, `quip-mybackend-gibbs` for Gibbs. A new algorithm for an existing device is just
another binary; an operator selects it in `config.toml` with `binary =
"quip-mybackend-gibbs"`. The coordinator's default when `binary` is omitted is
`quip-<backend>-sa`.

## Wiring a new backend into the coordinator

Adding a new algorithm to an existing backend needs no coordinator change — it
is a new binary the config points at. A brand-new backend category needs two
edits:

- **`crates/quip-coordinator/src/config.rs`** — `parse_config` recognizes the
  `[cpu]`, `[cuda.N]`, `[metal]`, and `[dwave]`/`[qpu]` sections. Add a branch
  for your `[<backend>]` section that pushes a launch entry with a stable miner
  id. `default_binary` already yields `quip-<backend>-sa`, so only the D-Wave
  special case needs an override.
- **`crates/quip-coordinator/tools/fetch-miners.sh`** — add a `fetch` call for
  your repository's assets and handle the backend in the `MINER_SET`/`auto`
  logic. The container images call this script, so they need no separate change.

## Publishing releases

`fetch-miners.sh` pulls each miner from its repository's GitLab generic-package
registry. It resolves each repository's newest release on its own, so a miner
reaches the images and the host install as soon as it cuts a release, and
nobody has to bump a pin here. Set the matching `<BACKEND>_MINERS_TAG` variable
to hold one miner at a specific tag.

Publish one asset per architecture named `<binary>-<arch>`
(`quip-mybackend-sa-amd64` for the amd64 build). The script downloads
`<binary>-<arch>` and saves it under the clean name `<binary>`, which is what
`config.toml`'s `binary` field and the coordinator's `PATH` lookup expect. A
real fetch stays inert until the miner repository has cut its first release.

## A non-Rust backend

A miner in another language speaks the same gRPC `MinerService` protocol
directly — `MINER_PROTOCOL.md` is the contract to implement against.
Use the `quip_proto` SDK: the generated `quip.v1` stubs for the wire,
and `quip_proto.scoring`/`quip_proto.wire` for the consensus math, so your
energies match the Rust path exactly (the Rust in `quip-protocol` is the source
of truth, exposed through PyO3). This is how the D-Wave miner works from Python.
Package it so the coordinator can launch it — the D-Wave miner ships as a pip
wheel whose console entry point is the configured `binary` (default
`quip-dwave-qa`).

## Testing a new miner

- `your-binary --capabilities` prints the capability JSON; `your-binary --check`
  probes that the device is runnable. Both exit without connecting.
- Drive mode runs the miner end to end with no chain:

  ```bash
  quip-coordinator drive --miner ./your-binary --source random \
    --topology-preset smoke --count 10
  ```

  It spawns the binary, feeds it generated problems, and prints per-job and
  aggregate timing. Pin `--num-reads`/`--num-sweeps` for a fixed comparison.
- If the backend adapts its own parameters, mirror the golden parity the other
  backends hold to (`conformance/golden_adapt.json`), so the budget matches
  across languages.

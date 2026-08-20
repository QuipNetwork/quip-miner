# Adding a new miner

A miner is a standalone program that solves Ising problems and speaks the gRPC
miner protocol to the coordinator. You can add one two ways:

- **A native backend on `quip-solver-core`** — the shape the CPU, CUDA, and
  Metal miners use. You write a `Sampler` and the harness does the rest.
- **A solver in another language** — speak the protocol directly. This is the
  shape the D-Wave miner uses from Python.

Both shapes are specified and documented in the
[quip-solver-core](https://gitlab.com/quip.network/quip-solver-core)
repository: `SPEC.md` is the contract, the crate README covers the Rust
`Sampler` path, and `examples/` holds a conformant mock solver in Rust, C++,
Python, and TypeScript. Depend on `quip-solver-core = "0.0.0"` from crates.io
for the Rust path, or on the `quip-solver-core` PyPI wheel for Python. A
solver is conformant when the repo's `quip-solver-drive` binary reports a
conformant session against it.

This guide covers what stays coordinator-side: naming, wiring the backend into
the coordinator, publishing releases, and testing against this coordinator.

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

## Testing a new miner

- Run `quip-solver-drive <your-binary> unix:///tmp/quip-check.sock` from the
  quip-solver-core repo. Exit code 0 means the solver is conformant.
- `your-binary --capabilities` prints the capability JSON; `your-binary --check`
  probes that the device is runnable. Both exit without connecting.
- Drive mode runs the miner end to end against this coordinator with no chain:

  ```bash
  quip-coordinator drive --miner ./your-binary --source random \
    --topology-preset smoke --count 10
  ```

  It spawns the binary, feeds it generated problems, and prints per-job and
  aggregate timing. Pin `--num-reads`/`--num-sweeps` for a fixed comparison.
- If the backend adapts its own parameters, mirror the golden parity the other
  backends hold to (the `quip-solver-conformance` crate's adapt vectors), so
  the budget matches across languages.

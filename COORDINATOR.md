# Coordinator — how the QuIP v0.3 coordinator works

`quip-coordinator` is the one process that touches the chain. It follows the
chain head and stages work for the miners. Each miner runs as a subprocess and
speaks the gRPC miner protocol to the coordinator over a local socket; the
coordinator collects the solutions and submits the winning proofs. The miners
never see the chain.

The crate is `crates/quip-coordinator`. This document describes the modules and
how they fit together. For build and run commands, see `AGENTS.md`.

## Run modes

`main.rs` parses one CLI:

- `quip-coordinator --config <config.toml>` — the production runtime
  (`run_config_path` → `runtime::run_runtime`). Connects to the validators,
  spawns and supervises the configured miners, feeds them chain work, submits
  proofs.
- `quip-coordinator drive …` — a synthetic driver (`drive` module). Spawns one
  miner and feeds it generated or replayed problems with no chain and no submit.
  Used for benchmarking and matched-condition parity runs.
- `quip-coordinator seed-chain …` — registers a default topology and sets its
  difficulty on a fresh chain. You can seed a chain only once. Wipe the chain
  data and restart the validator if `DefaultTopology` is already set.

## Logging

`logging.rs` installs the subscriber, and `main` calls it first. The `tracing`
macros only dispatch to an installed subscriber. Without that call every log
statement in the process does nothing and `RUST_LOG` has no effect.

Verbosity comes from `--log-level {trace,debug,info,warn,error}`, default
`info`. An explicit flag wins over `RUST_LOG`. When the flag is absent,
`RUST_LOG` applies. The default filter holds third-party crates at
`warn` and this crate's targets at the chosen level, so `--log-level debug`
stays readable. Use `RUST_LOG` when you need jsonrpsee or subxt internals.

All output goes to stderr, because `drive` prints its timing table to stdout.
Color is on only when stderr is a terminal, so a captured log file stays plain
text.

Spawned miners inherit these streams and log for themselves, so miner lines
appear alongside the coordinator's own. Both spawn paths — supervised and
`drive` — pass `--log-level` through to the child. `MINER.md` describes what a
miner logs at each level.

What each level carries:

- `info` — startup and the validator runtime, round transitions, miner spawn
  and registration, submit outcomes, and a one-minute throughput heartbeat.
- `debug` — per-poll feeder detail, difficulty changes, validator failover.
- `trace` — every RPC call.

The feeder logs state changes on transition, not per poll. It polls once a
second, so a validator that goes away logs one warning and then stays quiet
until reachability changes. The round machine logs each state at `trace` on
entry, once per transition, with the generation. A state held more than 10
seconds warns once.

## Startup checks

`main` refuses to start on three operator errors, all exit 64:

- **No backend section.** A config with no `[cpu]`, `[cuda.N]`, `[metal]`, or
  `[dwave]`/`[qpu]` launches no miners and can never mine. The error names the
  v0.2 `quip-miner` format when it sees that format's keys, because the two
  configs are not interchangeable.
- **An incompatible validator.** `chain::preflight` reads the runtime version
  and checks `QuantumPowApi`. The coordinator drives version 2 or newer, which
  is the first version whose `mining_snapshot` takes a topology selector.
- **An unfunded miner account.** `funding.rs` reads the account balance and,
  when the balance is below `min_balance_plancks` (2 UNIT), requests a top up
  from `faucet_url`. It retries with backoff for `funding_timeout_s` (10
  minutes by default), then gives up. An account that cannot pay transaction
  fees mines normally and fails every submit, which reads as a mining bug
  rather than an empty wallet.

The on-chain balance is the source of truth for funding, not the faucet's
reply. A faucet that answers 403 (already at its cap) or rate-limits is not a
failure by itself. Only HTTP 400 is permanent, because retrying a malformed
request cannot help. Set `faucet_url = ""` to turn autofunding off and fund
the account yourself.

An unreachable validator is not an error. The node manager starts the
coordinator and its validator together, so exiting because the node is still
booting would only crash-loop. The coordinator warns, starts, and retries.

## Waiting for the validator to sync

State 2 of the round machine waits until the node catches up. Every chain read
resolves against the node's best block. While the node imports history, that
block is far behind the real head. The miner account then reads as empty even
after the faucet has paid it, and any snapshot describes a round that ended
long ago.

`wait_until_synced` polls `system_health` every 5 seconds and warns with the
block from `system_syncState` every 30 seconds:

```
WARN validator is syncing at block 1108021 of 1240333 (132312 behind); waiting
     before funding and mining peers=8 waited_s=0
```

The wait has no ceiling while the node reports progress, because an initial
sync runs for hours and the funding budget is 10 minutes. Cutting it short is
what produced the misleading "miner account is not funded" exit, which is exit
64 and stops the supervisor respawning the coordinator.

The gate opens on `isSyncing`, the same signal the node manager reports. A node
flaps that flag near the tip, so the gate needs two consecutive clear polls once
it has seen a sync. The first answer settles a node that was never syncing, so a
normal start waits for nothing. When the node answers nothing at all for a
minute, the coordinator warns, starts, and leaves the retry to the feeder. Zero
peers on a node that expects peers draws its own warning, because that node has
no chain to catch up to.

`[miner].validators` is an ordered failover list, tried in order on every call.
When the key is absent it defaults to `["ws://quip-validator:9944",
"ws://127.0.0.1:9944"]`, matching v0.2. An explicit empty list stays empty.

## Runtime wiring

`run_runtime` (in `runtime.rs`) is the production entry point. It:

1. Binds a Unix-domain socket and serves the gRPC `MinerService` on it
   (`CoordinatorService`). Miners connect back over this socket.
2. Seeds each miner's `Configure` from the launch plan so the server can answer
   the handshake.
3. Spawns one `supervise_miner` task per configured miner.
4. Spawns the `feeder_loop`, which follows the chain head and keeps every
   miner's staged queue full.
5. Waits for `shutdown` (SIGINT/SIGTERM). On stop it flips a `watch` channel,
   drains the feeder and supervisors (so in-band `Shutdown` and any kill-after-
   grace finish), aborts the server, and removes the socket.

`CoordinatorState` (in `session.rs`) is the shared state behind an
`Arc<Mutex<…>>`: the `Router`, the in-flight map, the current `Topology` and
difficulty `target`, per-miner `Configure`, and the salt→job bookkeeping.

## The blockchain seam

All chain access sits behind one trait, `ChainClient` (`chain/mod.rs`):

- `fetch_mining_snapshot` — the mining inputs at a block: topology
  (nodes/edges, allowed h/J/spin ranges), difficulty gates
  (`max_energy_milli`, `min_solutions`, `min_diversity_milli`), and the round
  anchor (`last_proof_block_hash`).
- `fetch_mempool_orders` — open mempool orders eligible for this miner.
- `submit_proof` — hybrid-sign and submit a proof extrinsic, then classify the
  receipt.
- `declare_participation` — hybrid-sign and submit
  `MinerRegistry.participate` for one qblock, then classify the pallet
  error.
- `file_descriptor` — hybrid-sign and submit
  `MinerRegistry.set_descriptor` with a V2 payload, then classify the
  pallet error.
- `ensure_miner_registered` — read `QuantumPow.Miners` for the signing
  account and, when the account is absent, submit
  `QuantumPow.register_miner`.

One key produces two different 32-byte values, and they are not
interchangeable. The **account** is `blake2b_256(domain ++ public_key)`: it
signs, holds the balance, and keys every account storage map. The
**identity** is `blake2_256(account.encode())`, one further hash of that
account. The identity is the PoW nonce input, and it must equal what the
pallet computes in `account_to_bytes`, or the pallet rejects the proof with
`InvalidNonce`. Fund and look up the account. Derive nonces from the
identity.

### Confirming a submission

The transaction status stream reports pool and inclusion progress only. It
cannot say whether the dispatch inside the block succeeded, because that
answer lives in the block events and decoding those needs runtime metadata
this client does not carry. Each call instead names the storage entry its own
success writes, and the client reads that entry back at the inclusion block.

When the extrinsic is in the block but the storage entry is absent, the
client reports why in the terms of that storage rather than as one generic
failure. For a proof it reads `QuantumPow.QBlocks` at the inclusion block and
compares the winner:

| What the qblock slot holds | Meaning | Action |
| --- | --- | --- |
| Another account | Another miner won the race. The proof was sound. | `StopRoundStale`, logged at info |
| The signing account | The read raced the write | `StopRoundStale` |
| Nothing | The pallet rejected the proof itself. Check diversity, energy, and registration. | `classify_receipt` |

A lost race is the one included-but-failed case that says nothing is wrong
with the proof, so it must not appear as a rejection.

`system_dryRun` cannot answer this question. At the chain head the nonce is
already spent. At the parent block the replay leaves out whatever the block
ordered ahead of the extrinsic, which is the exact thing that beats a proof.

Naming the pallet error variant itself, such as `InsufficientDiversity`,
still needs metadata decoding. `classify_receipt` matches those names, so it
only ever sees them for errors the node reports before inclusion.

`RealChainClient` (`chain/real.rs`) is the live client over Substrate
JSON-RPC. `FakeChain` (`chain/fake.rs`) backs the tests. Supporting
modules: `extrinsic` (hybrid sr25519 + ML-DSA-44 signing, `load_hybrid_pair`,
`miner_identity_bytes`, `signer_account_bytes`), `snapshot`
(`MiningSnapshot`, `DecayParams`),
`mempool` (`JobOrder`), `submit` (`Proof`, `classify_receipt`, `SubmitAction`),
plus `proof_encode` and `scale_types` for SCALE codec. Nothing outside
`chain/` talks to the node directly.

## Job production — the feeder

`feeder_loop` (in `runtime.rs`) runs one poll every `poll_interval_ms`
(1000 ms in production). A win is visible on the next poll, so the worst-case
delay before `Cancel` is one poll interval plus the snapshot RPC.

When `last_proof_block_hash` changes, the feeder drives the round state
machine. Startup drives the same machine. Mining is the last state. A new
qblock head in any state returns the machine to the first state.

### Round state machine

The states, in order:

1. **Stop mining.** The feeder raises the generation, drops staged and
   in-flight PoW jobs of the dead generation, and broadcasts
   `Cancel{max_generation}`. Miners stay connected and idle. The gRPC server
   is not blocked.
2. **Validator is synced.** The feeder calls `wait_until_synced`.
3. **Account is funded.** The feeder calls `ensure_funded` with the signing
   account. One balance read is the common case. A mid-run failure is not
   fatal. The feeder warns, holds this state, and retries. It does not exit.
4. **Miner registered.** The coordinator reads `QuantumPow.Miners` for the
   signing account. If the account is absent, the coordinator submits
   `QuantumPow.register_miner`, which reserves the miner deposit from that
   account. Later rounds skip both the read and the submit. Three transient
   failures warn and hold this state. The pallet rejects every proof from an
   unregistered account, so mining waits until registration succeeds.
5. **Requirements for the next qblock downloaded.** Topology, energy target,
   required solution count, and diversity come from a fresh snapshot after
   sync and funding.
6. **Descriptor filed.** On the first walk after the process starts, the
   coordinator submits `MinerRegistry.set_descriptor` with a V2 payload.
   Later rounds skip the submit and log at `trace`. A missing required
   value, a pallet rejection, or three transient failures warn and advance.
   This state never holds mining.
7. **Participation declared.** The coordinator submits
   `MinerRegistry.participate` for the candidate qblock. The call already
   deduplicates per qblock. Three transient failures warn and advance. This
   state never holds mining.
8. **Start mining.** The feeder always sends `Topology` and `SetTarget` for
   the new round, then stages jobs. A job of the new generation cannot leave
   before those two messages.

### Node descriptor

`set_descriptor` is pallet 13, call 0. The coordinator files schema V2.

Values come from `[miner]`:

| Field | Config key | Default |
| --- | --- | --- |
| `node_id` | `[miner].node_id` | 64-char hex of the miner account |
| `node_name` | `[miner].node_name` | none. Required to file. |
| `public_host` | `[miner].public_host` | required |
| `public_port` | `[miner].public_port` | required |
| `rpc_endpoints` | `[miner].validators` | the validator list the coordinator already reads |
| `auto_mine` | `[miner].auto_mine` | `true` |
| `log_level` | `[miner].log_level` | `info` |
| `miners` | backend sections | one spec per launched miner |
| `system_info` | — | none |
| `runtime` | — | none |

`[miner].log_level` is the advertised node log level. It does not change
coordinator verbosity. Use `--log-level` or `RUST_LOG` for that.

`rest_host` and `rest_port` are v0.2 keys. The coordinator does not map
`rest_port` to `public_port`.

You must set `public_host` and `public_port`. `parse_config` rejects a
missing key. It also rejects a blank host and a port of 0.
`public_port` is the port a peer uses to reach this node from outside the
host. In the reference deployment that is the public front door, not the
validator peer port and not the dashboard port.

If `[miner].node_name` is missing, the coordinator warns once and names that
key. It does not file a descriptor. Mining still starts.

The pallet reserves `DescriptorDepositBase` plus
`DescriptorDepositPerByte` times the payload length. The runtime sets those
to 1 milliUNIT and 1 microUNIT. A typical coordinator descriptor costs about
0.001 UNIT. The funding floor is 2 UNIT (`min_balance_plancks`). The deposit
fits under that floor.

### Participation

The node manager reads `MinerRegistry::LatestParticipation` and compares it
to `QuantumPow::QBlockCount`. The coordinator must call
`MinerRegistry.participate` once per qblock or that check stays behind.

The pallet accepts only the current candidate qblock. That id is one past
`QuantumPowApi_latest_qblock_id`. The `new round` log line prints the last
minted id. The declaration uses the candidate.

`participate` takes one `MinerKind`. The coordinator derives that kind from
the miners it starts. A mixed fleet declares the highest-capability kind:
QPU, then ASIC, then Metal, then GPU, then CPU. The descriptor `miners`
list still carries every launched backend.

Pallet outcomes:

| Outcome | Action |
| --- | --- |
| success or `DuplicateParticipation` | treat as declared. Do not retry. |
| `InvalidQBlockId` | log at `debug`. Declare the new candidate next round. |
| `DescriptorRequired` | log at `warn` once, name the account, keep mining |
| transient chain error | retry up to three times in this state, then warn and advance |

A descriptor or participation failure never calls `process::exit` and never
holds mining.

Transitions:

| Current state | Event | Next state |
| --- | --- | --- |
| any | Shutdown | stop the machine |
| any | NewHead | Stop mining |
| any | Failed | same state (retry) |
| Stop mining | Succeeded | Validator is synced |
| Validator is synced | Succeeded | Account is funded |
| Account is funded | Succeeded | Requirements downloaded |
| Requirements downloaded | Succeeded | Descriptor filed |
| Descriptor filed | Succeeded | Participation declared |
| Participation declared | Succeeded | Start mining |
| Start mining | Succeeded | Start mining |

The feeder does the I/O. The transition function is pure. The feeder logs the
state at `trace` on each entry, once per transition, with the generation. A
state held more than 10 seconds warns once. An operator can read the current
state and the reason that mining has not started.

Startup drives states 2 through 6. A funding failure at startup is still exit
64. A missing snapshot at startup is a warning. The feeder retries that
download after miners connect.

One `info` line names the new generation, the cancelled staged-job count, the
miner count, and the refreshed target.

Mempool jobs carry generation `0` (`producer::mempool::job_order_to_job`).
Reseeds keep those jobs. PoW jobs carry the live generation. The feeder drops
them when their round ends. A cancelled in-flight job is not re-queued, not
scored, and not submitted. The miner refunds its credit with `JobRequest{1}`.

### Adaptive staging window

The staged depth per miner tracks how fast that miner drains work, so a
many-core backend stays saturated while a slow one does not hoard jobs. Each
poll samples the miner's dispatch counter (`Router::take_consumed`,
read-and-reset), smooths it with an EMA (`α = 0.3`), and sizes the window to
`ceil(ema × 2.0)` — about two poll-intervals of drain — with `buffer_depth` as
the floor and no ceiling. The EMA tracks queue drain. A job leaves the staged
queue at dispatch, so that is the signal the feeder sizes against. The default
floor is 256 (`main.rs`), generous enough to keep every miner fed from the
first poll. The one-minute heartbeat reports completions, not this dispatch
counter.

## Routing and credits

`Router` (`router.rs`) indexes miners by capability and holds a staged queue
per miner. `route` first-fits a job to a capable miner (by `supported_kinds`
and `max_nodes`/`max_edges` from the miner's `Hello`); `stage_on` targets a
specific miner, which is what the feeder uses.

Dispatch is gated by a **credit pool**, not a queue depth. The miner grants
`width` credits up front and one more per terminal event (`Result` or
`Reject`); each dispatch spends one credit (`next_job`). Dispatch stays
one-to-one with completion, and in-flight is bounded by the credits the miner
has actually issued. In-flight jobs live in `CoordinatorState`, so the
router keeps no separate outstanding counter.

On a reject with `UnsupportedKind`, the router records that the miner can't
serve that job kind and re-routes the job to another capable miner. On a miner
crash, `reclaim` returns its staged and in-flight jobs for re-queue.

## The miner session

`CoordinatorService<C>` is the server side of the gRPC `MinerService`
(`session.rs`) — the streaming miner protocol carried over the Unix socket. Per
session it runs the handshake (`Hello` → `Welcome`), applies `Configure`, then
services the bidirectional stream: a `JobRequest` grants credits, the
coordinator dispatches staged `Job`s, and each `Result` or `Reject` is a
terminal event that frees a credit and advances accounting. A periodic `Ping` (every 15s) draws a
`Status` that reports how busy the miner is and surfaces paused or stale-round
liveness; `Shutdown` ends the session in-band. The harness owns credits, reject
reasons, and liveness, so each backend only has to sample. `MINER_PROTOCOL.md`
specifies this contract from the miner's side.

## Supervision and shutdown

`supervise_miner` (`supervisor.rs`) owns one miner for the run. It spawns the
binary with the session token in the `QUIP_SESSION_TOKEN` environment variable
(never argv) and three arguments — `--quip-coordinator <socket-uri>`,
`--miner-id <id>`, and `--log-level <level>` — and sets `kill_on_drop`. The
child inherits the coordinator's stdout and stderr, so its log lines appear
directly in the coordinator's output. `--log-level` keeps the child's verbosity
matched, because those bytes never pass through the coordinator's own filter.

Restarts follow the child's exit code (`restart_policy`):

- `0` — clean exit; respawn on demand, reset backoff.
- `64` / `69` / `77` — operator, environment, or token error; don't respawn.
- `70` or a signal — crash; respawn after exponential backoff
  (`2^consecutive × base`, capped at `max`) under a failure budget. Too many
  crashes inside the window marks the miner unhealthy and stops respawning.

On a crash the miner's staged and in-flight jobs are re-queued. When the runtime
stops, each live miner gets an in-band `Shutdown` first, then a hard kill after
the grace period (`grace_ms`).

## Config to launch plan

`parse_config` (`config.rs`) turns `config.toml` into a `CoordinatorConfig`:
the `[miner]` section gives `validators`, `signer_key`, the required
descriptor keys (`public_host`, `public_port`), and the optional keys
(`node_name`, `log_level`, `node_id`, `auto_mine`). Each backend section
becomes one `LaunchEntry`. Section names map to miner ids `[cpu]`→`cpu-0`,
`[cuda.N]`→`cuda-N`, `[metal]`→`metal-0`, `[dwave]`/`[qpu]`→`qpu-0`. The
backend name stays on the entry so the coordinator can derive `MinerKind`.

`binary` selects the executable, defaulting to `quip-<backend>-sa` (and
`quip-dwave-qa` for D-Wave). The coordinator-owned keys (`binary`,
`queue_depth`, `idle_timeout_s`, `heartbeat_s`, `reconnect_window_s`) become
fields of the `Configure` message; every other key passes through verbatim in
`Configure.backend_toml`, which the coordinator forwards to the miner. This is
how a backend receives its own settings without the coordinator knowing them.

## Dashboard and REST surface

The optional `[dashboard]` section starts a read-only HTTP server. `listen`
gives the bind address. `data_dir` gives the directory that holds the
mining-attempt records. The coordinator writes every solved model as JSONL
under `data_dir/<qblock_id>/attempts.jsonl`. Omit the section to turn the
server off.

| Route | Response |
|-------|----------|
| `GET /healthz` | `ok` |
| `GET /qblocks` | JSON array of the available qblock ids |
| `GET /api/v1/status` | indexer status envelope |
| `GET /api/v1/stats` | indexer stats envelope |
| `GET /api/v1/mining/attempts?solution_number=N` | submission and attempts for one solution |
| anything else | static files under `data_dir` |

The server serves reads only. It exposes no control endpoint. Bind it to an
address the operator trusts, because it applies no authentication.

## Drive mode

`quip-coordinator drive` (the `drive` module and `DriveService` in `session.rs`)
spawns one miner and feeds it synthetic work with no chain and no submit. Jobs
come from a golden random draw (`--source random`, optionally with a topology
preset embedded in the binary) or a JSONL replay (`--source list`). Drive reports
per-job and total timing, and can pin
`num_reads`/`num_sweeps` through the `SetTarget` control-plane override to run
matched-condition throughput and parity comparisons. This is the path used for
CPU-versus-CUDA benchmarking.

The preset JSON stays under `crates/quip-coordinator/fixtures/drive/` as the
readable source. The binary copies each file in at compile time.

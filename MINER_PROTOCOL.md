# Miner protocol — the coordinator↔miner contract

A miner solves Ising problems and speaks one gRPC service to the coordinator.
This document specifies that service: the messages, the order they flow in, and
the behavior the coordinator expects from a miner. It's the contract every
backend follows, whether it builds on the `quip-miner-core` harness (CPU,
CUDA, Metal) or speaks the wire directly (D-Wave, from Python).

`proto/quip/v1/miner.proto` is the normative message definition. This document
is its behavioral companion: the `.proto` gives the fields, this gives the
rules. For the Rust harness behind the contract, see `MINER.md`; to add
a backend, see `NEWMINER.md`.

## Transport

The service is one bidirectional stream:

```proto
service MinerService {
  rpc Session(stream MinerMsg) returns (stream CoordMsg);
}
```

The coordinator serves it over a Unix-domain socket; the miner dials that socket
(passed as `--quip-coordinator unix://<path>`). One stream is one session, open
for the miner's lifetime. `MinerMsg` flows miner→coordinator and `CoordMsg`
flows coordinator→miner; both are `oneof` envelopes.

The protocol version is `1`. The coordinator passes the session token in the
`QUIP_SESSION_TOKEN` environment variable, never on the command line; the miner
echoes it in `Hello`.

## Handshake

Messages flow in this order. A miner that dispatches work before `Ready`, or
skips the token, breaks the session.

1. **Miner → `Hello`** — `miner_id`, `session_token`, `protocol_version = 1`,
   `backend`, `algorithm`, `supported_kinds`, `max_nodes`, `max_edges`, optional
   `features`. The capability fields tell the coordinator's router what this
   miner can serve.
2. **Coordinator → `Welcome`** — `protocol_version`. A version other than `1` is
   fatal; the miner exits `64`.
3. **Coordinator → `Configure`** — `queue_depth`, `idle_timeout_s`,
   `heartbeat_s`, `reconnect_window_s`, and `backend_toml` (this miner's config
   subsection, verbatim). The miner applies `backend_toml` against its own schema
   before any job.
4. **Coordinator → `Topology`, `SetTarget`** — sent immediately if a mining
   round is live, so a miner joining mid-round tracks the current problem and
   difficulty without waiting for the next reseed.
5. **Miner → `Ready`** — the miner is configured and can accept jobs. This
   unblocks dispatch.
6. **Miner → `JobRequest`** — grants the first batch of credits (see Credits).

The coordinator dispatches staged jobs as soon as it has both `Ready` and
credits for the miner.

## Credits

Dispatch is flow-controlled by a credit pool, not a fixed queue. The miner
grants credits, each dispatched `Job` spends one, and the miner returns one per
terminal outcome.

- The miner grants credits with `JobRequest{credits}`. It sizes the first grant
  to keep its pipeline full — enough to hold every in-flight model plus a
  prefetched next slot per lane.
- The coordinator spends one credit per `Job` it sends and never exceeds the
  granted pool. The miner's issued credits bound how much work is in flight.
- Every `Job` ends in exactly one terminal outcome: a `Result`, a `Reject`, or a
  cancellation (see Cancel). On each, the miner returns one credit with
  `JobRequest{credits: 1}` so the pool doesn't leak a slot — including when it
  rejects a job at prepare time.

Keeping the pool topped up is how a fast backend stays saturated; letting it
drain is how a busy or paused backend applies backpressure.

## Servicing a job

`Job` carries `job_id`, `kind`, `generation` (the round; `0` for mempool work),
`deadline_ms`, the `IsingProblem`, and `Provenance`. The miner answers with one
of:

- **`Result{job_id, solutions, meta}`** — one `Solution` per read: `spins_bytes`
  (one byte per spin, `0x01` = +1, `0xFF` = −1) and `energy_milli`. `meta`
  reports `reads`, `sweeps`, `device_access_time_us`, and `qpu_access_us`.
  Energies must come from the shared scoring path so every backend scores a
  solution identically.
- **`Reject{job_id, reason}`** — the miner won't run this job. The reason
  drives the coordinator:

  | Reason | Coordinator response |
  |---|---|
  | `UNSUPPORTED_KIND` | Records that the miner can't serve this kind; re-routes to another capable miner. |
  | `TOO_LARGE` | Job exceeds the miner's `max_nodes`/`max_edges` or `max_reads`. |
  | `TOPOLOGY_MISMATCH` / `TOPOLOGY_MISSING` | Job names a topology hash the miner hasn't cached. |
  | `MALFORMED` / `EXPIRED` / `OVERLOADED` / `SHUTTING_DOWN` | Job dropped from the in-flight set. |

The coordinator validates every `Result` itself and submits the winning proof to
the chain. A miner never sees the chain, the difficulty gate, or the proof.

## Parameters — adapt or pin

The miner picks its sampling budget (`num_reads`, `num_sweeps`, anneal time) by
precedence; the first non-zero value wins:

1. A per-job override on the `IsingProblem`.
2. A pin from `SetTarget` (`num_reads` / `num_sweeps` / `anneal_time_us`).
3. The miner's own adaptive value, computed from the target energy.
4. The backend default.

`SetTarget.max_energy_milli` is both the acceptance ceiling and the adaptation
target: with no explicit override, the miner scales its budget from it. A
non-zero pin bypasses adaptation — the drive and parity paths use this to run
matched conditions across backends.

## Control-plane pushes

Beyond jobs, the coordinator pushes control messages the miner must handle:

- **`Topology{hash, nodes, edges, allowed_h_milli}`** — the current problem
  graph. Sent at handshake and on every reseed. The miner caches it; a `Job` may
  name the graph by `topology_hash` instead of carrying `edges` inline.
- **`SetTarget{…}`** — the difficulty target. Sent after `Topology` and whenever
  difficulty changes. See Parameters.
- **`Cancel{max_generation}`** — abandon every generation at or below
  `max_generation`. This fires when a new chain round makes staged work stale.
  The miner advances its cancel watermark, skips buffered jobs from abandoned
  generations at dequeue, aborts in-flight stale work at its next checkpoint, and
  replies with a `Status`. A cancelled job produces no `Result`; its credit is
  still returned, and the abandoned generation is reported in
  `Status.abandoned_generation` rather than as a solution.
- **`Ping{}`** — a liveness probe. The miner replies promptly with `Status`.
- **`Shutdown{grace_ms}`** — end the session. The miner stops pulling new jobs,
  drains in-flight results within `grace_ms`, sends those `Result`s, then exits
  `0`.

## Status and liveness

`Status{miner_id, utilization, jobs_done, abandoned_generation, sampler_stats}`
is the miner's health report. The miner sends it in reply to `Ping` and
`Cancel`.

`sampler_stats` is a string map the coordinator reads for liveness:

- `state` — `mining` or `paused`.
- `reason` — the pause reason (`budget`), when `state = paused`.
- `generation` — the round the miner is working.

The coordinator folds this into per-miner liveness and logs the transitions:
entering or leaving `paused`, and a stale round (the miner reporting a
`generation` below the live one). A budget-paused miner stops requesting jobs,
which is indistinguishable from idle without this signal — the reason the
coordinator pings.

The coordinator pings every live miner every 15 seconds. The miner runs no
wall-clock idle timeout: a backend grinding a hard nonce for minutes can
legitimately receive nothing from the coordinator meanwhile — it's busy, not
dead. A dead peer surfaces as a closed stream or a transport error;
detecting it falls to HTTP/2 keepalive, not an application quiet period.
`Configure` still carries `idle_timeout_s`, `heartbeat_s`, and
`reconnect_window_s` as coordinator-side knobs.

## Exit codes

The harness maps a session outcome to a process exit code, and the supervisor's
restart policy keys on it. A miner speaking the wire directly exits the same
codes.

| Code | Name | Meaning | Coordinator |
|---|---|---|---|
| `0` | Clean | Normal exit after `Shutdown` or a closed stream. | Respawn on demand; reset backoff. |
| `64` | ConfigInvalid | Bad CLI/config, or a `Welcome` version ≠ 1. | Don't respawn. |
| `69` | EnvIncompatible | Host can't run this miner (`--check` failed). | Don't respawn. |
| `70` | InternalFatal | Unexpected internal failure. | Crash: respawn after backoff, under a failure budget. |
| `77` | TokenRejected | `QUIP_SESSION_TOKEN` missing or rejected. | Don't respawn. |

The optional `Fatal{exit_code, reason, restart_required}` message lets a miner
report the reason in-band before it exits, but the exit code is what the
supervisor acts on.

## Message reference

| Message | Direction | Purpose |
|---|---|---|
| `Hello` | M→C | Identify and advertise capabilities. |
| `Welcome` | C→M | Accept the session; state the protocol version. |
| `Configure` | C→M | Session knobs and the backend's verbatim config. |
| `Topology` | C→M | Current problem graph. |
| `SetTarget` | C→M | Difficulty target and optional parameter pins. |
| `Ready` | M→C | Configured; ready to accept jobs. |
| `JobRequest` | M→C | Grant credits. |
| `Job` | C→M | One Ising problem to solve. |
| `Result` | M→C | Solutions plus sampler metadata. |
| `Reject` | M→C | Decline a job, with a reason. |
| `Cancel` | C→M | Abandon stale generations. |
| `Ping` | C→M | Liveness probe. |
| `Status` | M→C | Health, load, and liveness state. |
| `Shutdown` | C→M | End the session within a grace window. |
| `Fatal` | M→C | Report a fatal reason before exit. |

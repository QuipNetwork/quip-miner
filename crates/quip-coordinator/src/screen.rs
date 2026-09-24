//! Probe-screen feed for the CUDA miner.
//!
//! A `PoW` instance is drawn from its nonce, and the instance sets almost all of
//! the energy a solve can reach. `quip-screen` (quip-miner-cuda) anneals many
//! fresh nonces briefly on the GPU and reports the deepest; the feeder then
//! gives the CUDA miner those salts instead of counter salts, so its full solve
//! is spent on instances that are likely to clear the target.
//!
//! The sidecar derives nonces exactly as the feeder does
//! (`derive_nonce(last_proof_block_hash, miner_identity, salt)`), and the feeder
//! derives every job from a screened salt with the ordinary
//! `derive_pow_job`, so validation, stash and submission are
//! unchanged.
//!
//! Environment:
//! - `QUIP_SCREEN_BIN`: path to `quip-screen`. Unset or empty: off.
//! - `QUIP_SCREEN_ARGS`: extra `quip-screen` arguments, whitespace separated,
//!   passed after `--spec <file>` (e.g. `--sweeps 512 serve --keep-per-s 40`).
//!   `serve` is appended when absent.
//! - `QUIP_SCREEN_FALLBACK=1`: top the CUDA miner up with counter salts while no
//!   screened salt is queued (default: leave the GPU to the screen).
//!
//! Protocol (line based): the feeder writes `R <generation> <prev hash hex>
//! <identity hex>` when the round changes; the sidecar writes `K <generation>
//! <salt hex> <probe energy>` per kept nonce and `S <nonces/s> <cutoff>
//! <kept/s> ...` every 10 s. Closing stdin stops the sidecar. If the sidecar
//! exits, the feeder falls back to counter salts.

use crate::chain::snapshot::MiningSnapshot;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{ChildStdin, Command, Stdio};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Screened salts held per round; the deepest are served first.
const QUEUE_CAP: usize = 20_000;

/// Screened salts of the current round, deepest first.
#[derive(Debug, Default)]
pub struct SaltQueue {
    generation: u64,
    heap: BinaryHeap<(Reverse<i64>, [u8; 32])>,
}

impl SaltQueue {
    /// Start a new round: drop every salt of the previous one.
    pub fn reset(&mut self, generation: u64) {
        self.generation = generation;
        self.heap.clear();
    }

    /// Queue a salt reported for `generation`; salts of another round are
    /// dropped. At the cap the queue keeps its deepest half, so a long round
    /// still admits later, deeper salts. Returns whether it was queued.
    pub fn push(&mut self, generation: u64, salt: [u8; 32], energy: i64) -> bool {
        if generation != self.generation {
            return false;
        }
        if self.heap.len() >= QUEUE_CAP {
            // Ascending by `Reverse(energy)` is shallowest first.
            let mut kept = std::mem::take(&mut self.heap).into_sorted_vec();
            kept.reverse();
            kept.truncate(QUEUE_CAP / 2);
            self.heap = kept.into_iter().collect();
        }
        self.heap.push((Reverse(energy), salt));
        true
    }

    /// The deepest queued salt of `generation`.
    pub fn pop(&mut self, generation: u64) -> Option<([u8; 32], i64)> {
        if generation != self.generation {
            return None;
        }
        self.heap.pop().map(|(Reverse(e), salt)| (salt, e))
    }

    /// Salts waiting.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether no salt is waiting.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

#[derive(Default)]
struct Sidecar {
    stdin: Option<ChildStdin>,
    alive: bool,
    topology: Vec<u8>,
    announced: Option<(u64, [u8; 32])>,
    queue: SaltQueue,
    received: u64,
    fed: u64,
    best_fed: Option<i64>,
    /// Set when the sidecar could not start or exited on its own: counter
    /// salts from then on, no respawn loop.
    failed: bool,
    /// Bumped per spawn, so a reader thread only reports its own sidecar's exit.
    epoch: u64,
}

fn sidecar() -> MutexGuard<'static, Sidecar> {
    static S: OnceLock<Mutex<Sidecar>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(Sidecar::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn bin() -> Option<&'static str> {
    static B: OnceLock<Option<String>> = OnceLock::new();
    B.get_or_init(|| {
        std::env::var("QUIP_SCREEN_BIN")
            .ok()
            .filter(|v| !v.trim().is_empty())
    })
    .as_deref()
}

/// Whether the screen is configured.
#[must_use]
pub fn enabled() -> bool {
    bin().is_some()
}

/// Whether the CUDA miner should get counter salts while the queue is empty:
/// `QUIP_SCREEN_FALLBACK=1`, or no sidecar running.
#[must_use]
pub fn fallback() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("QUIP_SCREEN_FALLBACK").is_ok_and(|v| v == "1"))
        || !sidecar().alive
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            // Writing to a String cannot fail.
            let _ = write!(s, "{b:02x}");
            s
        })
}

fn unhex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}

/// The snapshot's topology as the drive/seed-chain spec JSON `quip-screen`
/// reads: nodes, edges in chain order, and the allowed values.
#[must_use]
pub fn spec_json(snap: &MiningSnapshot) -> String {
    serde_json::json!({
        "nodes": snap.nodes,
        "edges": snap.edges.iter().map(|&(u, v)| [u, v]).collect::<Vec<_>>(),
        "allowed_h_milli": snap.allowed_h_milli,
        "allowed_j_milli": snap.allowed_j_milli,
    })
    .to_string()
}

/// Parse one sidecar line into the queue. Returns the stats line fields, if
/// it was one.
fn ingest(line: &str) -> Option<String> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    match parts.as_slice() {
        ["K", generation, salt, energy] => {
            if let (Ok(g), Some(salt), Ok(e)) = (
                generation.parse::<u64>(),
                unhex32(salt),
                energy.parse::<i64>(),
            ) {
                let mut s = sidecar();
                s.received += 1;
                let _queued = s.queue.push(g, salt, e);
            }
            None
        }
        ["S", rest @ ..] => Some(rest.join(" ")),
        _ => None,
    }
}

fn spawn(s: &mut Sidecar, snap: &MiningSnapshot) {
    let Some(path) = bin() else { return };
    let spec: PathBuf = std::env::temp_dir().join(format!(
        "quip-screen-{}.json",
        hex(snap.topology_hash.get(..8).unwrap_or(&snap.topology_hash))
    ));
    if let Err(e) = std::fs::write(&spec, spec_json(snap)) {
        tracing::error!(error = %e, path = %spec.display(), "screen: cannot write the topology spec; counter salts only");
        s.failed = true;
        return;
    }
    let extra: Vec<String> = std::env::var("QUIP_SCREEN_ARGS")
        .unwrap_or_default()
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let mut args = vec!["--spec".to_owned(), spec.display().to_string()];
    let has_serve = extra.iter().any(|a| a == "serve");
    args.extend(extra);
    if !has_serve {
        args.push("serve".to_owned());
    }
    let child = Command::new(path)
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(error = %e, bin = path, "screen: cannot start the sidecar; counter salts only");
            s.failed = true;
            return;
        }
    };
    let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
        tracing::error!("screen: sidecar pipes missing; counter salts only");
        s.failed = true;
        return;
    };
    s.epoch += 1;
    let epoch = s.epoch;
    s.stdin = Some(stdin);
    s.alive = true;
    s.topology.clone_from(&snap.topology_hash);
    s.announced = None;
    tracing::info!(bin = path, ?args, "screen: sidecar started");
    let reader = std::thread::Builder::new()
        .name("screen-reader".into())
        .spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if let Some(stats) = ingest(&line) {
                    let mut s = sidecar();
                    tracing::info!(
                        sidecar = %stats,
                        received = s.received,
                        fed = s.fed,
                        queued = s.queue.len(),
                        best_fed = ?s.best_fed,
                        "screen stats"
                    );
                    s.best_fed = None;
                }
            }
            let current = {
                let mut s = sidecar();
                let current = s.epoch == epoch;
                if current {
                    s.alive = false;
                    s.stdin = None;
                    s.failed = true;
                }
                current
            };
            if current {
                tracing::error!("screen: sidecar exited; counter salts only");
            }
            if let Err(e) = child.wait() {
                tracing::warn!(error = %e, "screen: cannot reap the sidecar");
            }
        });
    if let Err(e) = reader {
        tracing::error!(error = %e, "screen: cannot start the reader thread");
        s.alive = false;
        s.stdin = None;
        s.failed = true;
    }
}

/// Tell the sidecar which round to screen, starting it on first use. A
/// topology change restarts it with a fresh spec. Cheap when nothing changed.
pub fn announce(generation: u64, snap: &MiningSnapshot, miner_identity: [u8; 32]) {
    let mut s = sidecar();
    if s.alive && s.topology != snap.topology_hash {
        tracing::info!("screen: topology changed; restarting the sidecar");
        // Closing stdin ends the sidecar; its reader thread then marks it dead.
        s.stdin = None;
        s.alive = false;
        s.failed = false;
    }
    if !s.alive && !s.failed && s.stdin.is_none() {
        spawn(&mut s, snap);
    }
    if s.announced == Some((generation, snap.last_proof_block_hash)) {
        return;
    }
    s.queue.reset(generation);
    s.announced = Some((generation, snap.last_proof_block_hash));
    let line = format!(
        "R {generation} {} {}\n",
        hex(&snap.last_proof_block_hash),
        hex(&miner_identity)
    );
    let broken = s.stdin.as_mut().is_some_and(|w| {
        w.write_all(line.as_bytes())
            .and_then(|()| w.flush())
            .is_err()
    });
    if broken {
        s.stdin = None;
        s.alive = false;
    }
}

/// The deepest screened salt queued for `generation`, if any.
#[must_use]
pub fn pop(generation: u64) -> Option<[u8; 32]> {
    let mut s = sidecar();
    let (salt, energy) = s.queue.pop(generation)?;
    s.fed += 1;
    s.best_fed = Some(s.best_fed.map_or(energy, |b| b.min(energy)));
    Some(salt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_serves_deepest_first_within_its_round() {
        let mut q = SaltQueue::default();
        q.reset(7);
        assert!(q.push(7, [1; 32], -14_400));
        assert!(q.push(7, [2; 32], -14_500));
        assert!(!q.push(6, [3; 32], -14_900), "a stale round is dropped");
        assert_eq!(q.pop(8), None, "a newer round sees nothing");
        assert_eq!(q.pop(7), Some(([2; 32], -14_500)));
        assert_eq!(q.pop(7), Some(([1; 32], -14_400)));
        assert!(q.is_empty());
    }

    #[test]
    fn a_full_queue_keeps_its_deepest_half_and_admits_new_salts() {
        let mut q = SaltQueue::default();
        q.reset(1);
        for i in 0..QUEUE_CAP {
            let e = -i64::try_from(i).unwrap_or(0);
            assert!(q.push(1, [0; 32], e));
        }
        assert!(
            q.push(1, [9; 32], -1_000_000),
            "a deeper salt is admitted when full"
        );
        assert_eq!(q.len(), QUEUE_CAP / 2 + 1);
        assert_eq!(q.pop(1), Some(([9; 32], -1_000_000)));
        let shallowest_kept = -i64::try_from(QUEUE_CAP / 2).unwrap_or(0) + 1;
        let mut last = i64::MIN;
        while let Some((_, e)) = q.pop(1) {
            assert!(
                e >= last && e <= shallowest_kept,
                "deepest half, served deepest first"
            );
            last = e;
        }
    }

    #[test]
    fn reset_drops_the_previous_round() {
        let mut q = SaltQueue::default();
        q.reset(1);
        assert!(q.push(1, [1; 32], -1));
        q.reset(2);
        assert!(q.is_empty());
    }

    #[test]
    fn hex_round_trips() {
        let b: [u8; 32] = core::array::from_fn(|i| u8::try_from(i * 7 % 256).unwrap_or(0));
        assert_eq!(unhex32(&hex(&b)), Some(b));
        assert_eq!(unhex32("zz"), None);
    }

    #[test]
    fn stats_lines_are_passed_through_and_salts_parsed() {
        assert_eq!(
            ingest("S 24000 -14397 40.1 r4x512").as_deref(),
            Some("24000 -14397 40.1 r4x512")
        );
        assert_eq!(ingest("K not-a-number 00 1"), None);
    }
}

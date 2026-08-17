//! Coordinator-side miner liveness: read the self-reported `Status` state and
//! surface mining↔paused transitions and stale-round reports as info logs.
//! The miner populates `sampler_stats`; this module only interprets it.
//!
//! Miners are active pollers, so a budget-paused miner simply stops requesting
//! jobs — indistinguishable from idle without an explicit signal. The periodic
//! [`liveness_loop`] pings live miners; each `Status` reply carries the miner's
//! `state`/`reason`/`generation` in `sampler_stats`, which [`evaluate_status`]
//! folds into per-miner [`MinerLiveness`] and the transitions worth logging.

use crate::session::{coord, CoordinatorState};
use quip_proto::v1::{coord_msg, Ping};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{watch, Mutex};

/// Last-known self-reported liveness for one miner.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct MinerLiveness {
    /// Whether the miner last reported `state=paused`.
    pub paused: bool,
    /// Pause reason from `sampler_stats` when paused.
    pub reason: Option<String>,
    /// Last reported generation from `sampler_stats`, if any.
    pub generation: Option<u64>,
}

/// A change worth logging, derived from a ping-reply `Status`.
#[derive(Debug)]
pub enum LivenessEvent {
    /// Miner transitioned into the paused state.
    EnteredPaused {
        /// Pause reason from the miner's stats, if provided.
        reason: Option<String>,
    },
    /// Miner left the paused state and resumed mining.
    Resumed,
    /// Miner reported a generation behind the feeder's live round.
    StaleRound {
        /// Generation the miner reported.
        reported: u64,
        /// Feeder's current generation.
        current: u64,
    },
}

/// Fold a `Status`'s `sampler_stats` into the prior liveness, returning the new
/// liveness and the transitions to log. `current_generation` is the feeder's
/// live round; a reported generation below it is stale.
#[expect(
    clippy::implicit_hasher,
    reason = "callers pass the default hasher; generalizing is a public API change"
)]
pub fn evaluate_status(
    prev: &MinerLiveness,
    stats: &HashMap<String, String>,
    current_generation: u64,
) -> (MinerLiveness, Vec<LivenessEvent>) {
    let paused = stats.get("state").map(String::as_str) == Some("paused");
    let reason = if paused {
        stats.get("reason").cloned()
    } else {
        None
    };
    let generation = stats.get("generation").and_then(|g| g.parse::<u64>().ok());

    let mut events = Vec::new();
    if paused && !prev.paused {
        events.push(LivenessEvent::EnteredPaused {
            reason: reason.clone(),
        });
    } else if !paused && prev.paused {
        events.push(LivenessEvent::Resumed);
    }
    if let Some(g) = generation {
        if current_generation > 0 && g < current_generation && prev.generation != Some(g) {
            events.push(LivenessEvent::StaleRound {
                reported: g,
                current: current_generation,
            });
        }
    }

    (
        MinerLiveness {
            paused,
            reason,
            generation,
        },
        events,
    )
}

/// Send a `Ping` to every live miner. Clones the outbound senders under the
/// lock, drops it, then awaits the sends so the fan-out never holds the state
/// lock across I/O. Returns the number of miners pinged.
pub async fn ping_once(state: &Arc<Mutex<CoordinatorState>>) -> usize {
    let senders: Vec<_> = {
        let st = state.lock().await;
        st.outbound.values().cloned().collect()
    };
    let n = senders.len();
    for tx in senders {
        let _ = tx.send(Ok(coord(coord_msg::Msg::Ping(Ping {})))).await;
    }
    n
}

/// Periodically ping live miners until `stop` flips, so their `Status` replies
/// surface paused/stale transitions (logged by the session `Status` handler).
pub async fn liveness_loop(
    state: Arc<Mutex<CoordinatorState>>,
    interval: Duration,
    mut stop: watch::Receiver<bool>,
) {
    let mut ticker = tokio::time::interval(interval);
    loop {
        tokio::select! {
            _ = ticker.tick() => { let _ = ping_once(&state).await; }
            _ = stop.changed() => { if *stop.borrow() { break; } }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn mining_to_paused_emits_entered_paused_with_reason() {
        let prev = MinerLiveness::default();
        let (next, events) = evaluate_status(
            &prev,
            &stats(&[("state", "paused"), ("reason", "budget")]),
            5,
        );
        assert!(next.paused);
        assert_eq!(next.reason.as_deref(), Some("budget"));
        assert!(matches!(
            events.as_slice(),
            [LivenessEvent::EnteredPaused { reason }] if reason.as_deref() == Some("budget")
        ));
    }

    #[test]
    fn staying_paused_emits_nothing() {
        let prev = MinerLiveness {
            paused: true,
            reason: Some("budget".into()),
            generation: None,
        };
        let (_next, events) = evaluate_status(
            &prev,
            &stats(&[("state", "paused"), ("reason", "budget")]),
            5,
        );
        assert!(events.is_empty());
    }

    #[test]
    fn paused_to_mining_emits_resumed() {
        let prev = MinerLiveness {
            paused: true,
            reason: Some("budget".into()),
            generation: Some(5),
        };
        let (next, events) = evaluate_status(
            &prev,
            &stats(&[("state", "mining"), ("generation", "5")]),
            5,
        );
        assert!(!next.paused);
        assert!(events.iter().any(|e| matches!(e, LivenessEvent::Resumed)));
    }

    #[test]
    fn stale_generation_emits_stale_round_once_per_distinct_value() {
        let prev = MinerLiveness::default();
        let (next, events) = evaluate_status(
            &prev,
            &stats(&[("state", "mining"), ("generation", "4")]),
            7,
        );
        assert!(events.iter().any(|e| matches!(
            e,
            LivenessEvent::StaleRound {
                reported: 4,
                current: 7
            }
        )));
        let (_n2, events2) = evaluate_status(
            &next,
            &stats(&[("state", "mining"), ("generation", "4")]),
            7,
        );
        assert!(!events2
            .iter()
            .any(|e| matches!(e, LivenessEvent::StaleRound { .. })));
    }

    #[test]
    fn current_generation_is_not_stale() {
        let prev = MinerLiveness::default();
        let (_next, events) = evaluate_status(
            &prev,
            &stats(&[("state", "mining"), ("generation", "7")]),
            7,
        );
        assert!(!events
            .iter()
            .any(|e| matches!(e, LivenessEvent::StaleRound { .. })));
    }

    #[tokio::test]
    async fn ping_once_sends_ping_to_registered_miners() {
        use tokio::sync::mpsc;

        let state = Arc::new(Mutex::new(CoordinatorState::new()));
        let (tx, mut rx) = mpsc::channel(4);
        state.lock().await.register_outbound("cpu-0", tx);

        let n = ping_once(&state).await;
        assert_eq!(n, 1);
        let msg = rx.recv().await.expect("a message").expect("ok");
        assert!(matches!(msg.msg, Some(coord_msg::Msg::Ping(_))));
    }
}

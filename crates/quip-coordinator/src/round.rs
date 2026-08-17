//! Named state machine for one mining round.
//!
//! The feeder and startup drive this machine and perform the I/O. Transitions
//! are a pure function of the current [`RoundState`] and a [`RoundEvent`].

use std::fmt;

/// Where the coordinator is in the round lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RoundState {
    /// Stop every miner. Broadcast `Cancel{max_generation}`.
    StopMining,
    /// Wait until the validator has caught up.
    ValidatorSynced,
    /// Confirm the miner account can pay submit fees.
    AccountFunded,
    /// Confirm the signing account is in `QuantumPow.Miners`.
    MinerRegistered,
    /// Download topology, target, minimum solutions, and diversity.
    RequirementsDownloaded,
    /// File a node descriptor. Submits only on the first walk after process start.
    DescriptorFiled,
    /// File a participation marker for the candidate qblock.
    ParticipationDeclared,
    /// Broadcast the requirements and stage jobs.
    StartMining,
}

/// What happened in the current state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RoundEvent {
    /// The current step finished. Advance.
    Succeeded,
    /// The current step failed. Stay and retry.
    Failed,
    /// A new qblock head arrived. Return to [`RoundState::StopMining`].
    NewHead,
    /// The process is stopping. The machine ends.
    Shutdown,
}

impl RoundState {
    /// First state of every round.
    #[must_use]
    pub(crate) const fn start() -> Self {
        Self::StopMining
    }

    /// Next state, or `None` when the machine should stop.
    #[must_use]
    pub(crate) fn transition(self, event: RoundEvent) -> Option<Self> {
        match event {
            RoundEvent::Shutdown => None,
            RoundEvent::NewHead => Some(Self::StopMining),
            RoundEvent::Failed => Some(self),
            RoundEvent::Succeeded => match self {
                Self::StopMining => Some(Self::ValidatorSynced),
                Self::ValidatorSynced => Some(Self::AccountFunded),
                Self::AccountFunded => Some(Self::MinerRegistered),
                Self::MinerRegistered => Some(Self::RequirementsDownloaded),
                Self::RequirementsDownloaded => Some(Self::DescriptorFiled),
                Self::DescriptorFiled => Some(Self::ParticipationDeclared),
                Self::ParticipationDeclared => Some(Self::StartMining),
                Self::StartMining => Some(self),
            },
        }
    }

    /// Snake-case name for the log.
    #[must_use]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::StopMining => "stop_mining",
            Self::ValidatorSynced => "validator_synced",
            Self::AccountFunded => "account_funded",
            Self::MinerRegistered => "miner_registered",
            Self::RequirementsDownloaded => "requirements_downloaded",
            Self::DescriptorFiled => "descriptor_filed",
            Self::ParticipationDeclared => "participation_declared",
            Self::StartMining => "start_mining",
        }
    }

    /// Why the coordinator is in this state. Logged on entry.
    #[must_use]
    pub(crate) const fn reason(self) -> &'static str {
        match self {
            Self::StopMining => "stopping miners; a new qblock ended the round",
            Self::ValidatorSynced => "waiting until the validator is synced",
            Self::AccountFunded => "confirming the miner account can pay submit fees",
            Self::MinerRegistered => "registering the signing account as a miner on chain",
            Self::RequirementsDownloaded => "downloading the next qblock requirements",
            Self::DescriptorFiled => "filing the node descriptor",
            Self::ParticipationDeclared => "declaring participation for the candidate qblock",
            Self::StartMining => "starting mining",
        }
    }

    /// Log this state once, when the machine enters it.
    ///
    /// A healthy walk leaves each state without a retry in well under a
    /// second. That is `trace`. A retry, or a state held more than 10
    /// seconds, uses [`Self::log_unhealthy`].
    pub(crate) fn log_entry(self, generation: u64) {
        tracing::trace!(state = %self, generation, "{}", self.reason());
    }

    /// Warn once for a state that is retrying or has been held too long.
    pub(crate) fn log_unhealthy(self, generation: u64) {
        tracing::warn!(state = %self, generation, "{}", self.reason());
    }
}

impl fmt::Display for RoundState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::{RoundEvent, RoundState};

    fn step(state: RoundState, event: RoundEvent) -> RoundState {
        let next = state.transition(event);
        assert!(
            next.is_some(),
            "expected {state:?} + {event:?} to continue the round"
        );
        match next {
            Some(s) => s,
            None => state,
        }
    }

    const ALL: [RoundState; 8] = [
        RoundState::StopMining,
        RoundState::ValidatorSynced,
        RoundState::AccountFunded,
        RoundState::MinerRegistered,
        RoundState::RequirementsDownloaded,
        RoundState::DescriptorFiled,
        RoundState::ParticipationDeclared,
        RoundState::StartMining,
    ];

    #[test]
    fn full_round_visits_each_state_in_order() {
        let mut state = RoundState::start();
        let mut seen = vec![state];
        for _ in 0..7 {
            state = step(state, RoundEvent::Succeeded);
            seen.push(state);
        }
        assert_eq!(seen, ALL);
    }

    #[test]
    fn failed_step_retries_the_same_state() {
        for state in ALL {
            assert_eq!(
                state.transition(RoundEvent::Failed),
                Some(state),
                "{state:?} must stay and retry"
            );
        }
    }

    #[test]
    fn new_head_in_states_3_4_or_5_returns_to_stop_mining() {
        for state in [
            RoundState::AccountFunded,
            RoundState::RequirementsDownloaded,
            RoundState::StartMining,
        ] {
            assert_eq!(
                state.transition(RoundEvent::NewHead),
                Some(RoundState::StopMining),
                "{state:?}"
            );
        }
    }

    #[test]
    fn new_head_from_descriptor_or_participation_returns_to_stop_mining() {
        for state in [
            RoundState::DescriptorFiled,
            RoundState::ParticipationDeclared,
        ] {
            assert_eq!(
                state.transition(RoundEvent::NewHead),
                Some(RoundState::StopMining),
                "{state:?}"
            );
        }
    }

    #[test]
    fn new_head_in_any_state_returns_to_stop_mining() {
        for state in ALL {
            assert_eq!(
                state.transition(RoundEvent::NewHead),
                Some(RoundState::StopMining),
                "{state:?}"
            );
        }
    }

    #[test]
    fn a_win_during_mining_restarts_the_round() {
        let mut state = RoundState::start();
        for _ in 0..7 {
            state = step(state, RoundEvent::Succeeded);
        }
        assert_eq!(state, RoundState::StartMining);
        state = step(state, RoundEvent::NewHead);
        assert_eq!(state, RoundState::StopMining);
    }

    #[test]
    fn start_mining_keeps_mining_until_a_new_head() {
        assert_eq!(
            RoundState::StartMining.transition(RoundEvent::Succeeded),
            Some(RoundState::StartMining)
        );
    }

    #[test]
    fn shutdown_stops_the_machine_from_every_state() {
        for state in ALL {
            assert_eq!(state.transition(RoundEvent::Shutdown), None, "{state:?}");
        }
    }
}

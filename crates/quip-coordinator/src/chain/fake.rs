//! Scripted chain for tests: fixed snapshot, optional mempool order, captured submits.

use super::sync::{SyncSource, SyncStatus};
use super::{
    ChainClient, ChainError, DecayParams, DescriptorOutcome, JobOrder, MinerInfo, MiningSnapshot,
    NodeDescriptorV2Input, ParticipationOutcome, Proof, RegistrationOutcome, SubmitAction,
    SubmitReceipt,
};
use crate::funding::BalanceSource;
use async_trait::async_trait;
use std::sync::Mutex;

/// Test double: returns a scripted snapshot / order and records submits.
pub struct FakeChain {
    snapshot: Mutex<Option<MiningSnapshot>>,
    orders: Mutex<Vec<JobOrder>>,
    /// Captured proofs from [`ChainClient::submit_proof`].
    pub submitted: Mutex<Vec<Proof>>,
    /// Optional scripted submit receipt (default `Success`, no chain detail).
    submit_result: Mutex<Result<SubmitReceipt, ChainError>>,
    /// Scripted `latest_qblock_id` (default `None`).
    qblock_id: Mutex<Option<u64>>,
    /// Scripted `fetch_decay_params` (default `None`).
    decay_params: Mutex<Option<DecayParams>>,
    /// Scripted free balance. Defaults to a funded account so existing feeder
    /// tests keep mining without extra setup.
    balance: Mutex<Result<u128, String>>,
    /// Scripted sync status. Defaults to a caught-up validator.
    sync: Mutex<Result<SyncStatus, String>>,
    balance_calls: Mutex<usize>,
    sync_calls: Mutex<usize>,
    /// Captured `qblock_id` values from [`ChainClient::declare_participation`].
    participations: Mutex<Vec<u64>>,
    /// Scripted participate result (default `Declared`).
    participation_result: Mutex<Result<ParticipationOutcome, ChainError>>,
    /// Captured payloads from [`ChainClient::file_descriptor`].
    descriptors: Mutex<Vec<NodeDescriptorV2Input>>,
    /// Scripted descriptor result (default `Filed`).
    descriptor_result: Mutex<Result<DescriptorOutcome, ChainError>>,
    /// Scripted `QuantumPow.Miners` presence for the signing account.
    registered: Mutex<bool>,
    /// Scripted `register_miner` result (default `Registered`).
    registration_result: Mutex<Result<RegistrationOutcome, ChainError>>,
    /// Calls to [`ChainClient::ensure_miner_registered`], submitted or not.
    registration_calls: Mutex<usize>,
    /// Calls that reached the submit path (the account was not registered).
    registration_submits: Mutex<usize>,
    /// Accounts passed to [`BalanceSource::free_balance`], in call order.
    balance_accounts: Mutex<Vec<[u8; 32]>>,
    /// Scripted `QuantumPow.Miners[account]` value (default `None`).
    miner_info: Mutex<Option<MinerInfo>>,
}

impl FakeChain {
    /// Build a fake chain with a fixed snapshot and optional single order.
    #[must_use]
    pub fn new(snapshot: MiningSnapshot, order: Option<JobOrder>) -> Self {
        Self {
            snapshot: Mutex::new(Some(snapshot)),
            orders: Mutex::new(order.into_iter().collect()),
            submitted: Mutex::new(Vec::new()),
            submit_result: Mutex::new(Ok(SubmitReceipt::action_only(SubmitAction::Success))),
            qblock_id: Mutex::new(None),
            decay_params: Mutex::new(None),
            balance: Mutex::new(Ok(u128::MAX)),
            sync: Mutex::new(Ok(SyncStatus {
                is_syncing: false,
                peers: 0,
                should_have_peers: false,
                current_block: Some(1),
                highest_block: Some(1),
            })),
            balance_calls: Mutex::new(0),
            sync_calls: Mutex::new(0),
            participations: Mutex::new(Vec::new()),
            participation_result: Mutex::new(Ok(ParticipationOutcome::Declared)),
            descriptors: Mutex::new(Vec::new()),
            descriptor_result: Mutex::new(Ok(DescriptorOutcome::Filed)),
            registered: Mutex::new(false),
            registration_result: Mutex::new(Ok(RegistrationOutcome::Registered)),
            registration_calls: Mutex::new(0),
            registration_submits: Mutex::new(0),
            balance_accounts: Mutex::new(Vec::new()),
            miner_info: Mutex::new(None),
        }
    }

    /// Script the next `fetch_latest_qblock_id` return value.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    pub fn set_qblock_id(&self, id: Option<u64>) {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.qblock_id.lock().unwrap() = id;
        }
    }

    /// Script the next `fetch_decay_params` return value.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    pub fn set_decay_params(&self, params: Option<DecayParams>) {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.decay_params.lock().unwrap() = params;
        }
    }

    /// Replace the scripted mining snapshot (`None` → empty snapshot fetch).
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    pub fn set_snapshot(&self, snap: Option<MiningSnapshot>) {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.snapshot.lock().unwrap() = snap;
        }
    }

    /// Replace the scripted open mempool orders.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    pub fn set_orders(&self, orders: Vec<JobOrder>) {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.orders.lock().unwrap() = orders;
        }
    }

    /// Number of proofs captured via `submit_proof`.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    #[must_use]
    pub fn submitted_count(&self) -> usize {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            self.submitted.lock().unwrap().len()
        }
    }

    /// Drain and return all captured submits.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    #[must_use]
    pub fn take_submitted(&self) -> Vec<Proof> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            std::mem::take(&mut *self.submitted.lock().unwrap())
        }
    }

    /// Script the next `submit_proof` action. The receipt carries no chain
    /// detail; use [`Self::set_submit_receipt`] when a test needs the hashes.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    pub fn set_submit_result(&self, action: Result<SubmitAction, ChainError>) {
        let receipt = action.map(SubmitReceipt::action_only);
        self.set_submit_receipt(receipt);
    }

    /// Script the next `submit_proof` receipt in full.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    pub fn set_submit_receipt(&self, receipt: Result<SubmitReceipt, ChainError>) {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.submit_result.lock().unwrap() = receipt;
        }
    }

    /// Script the next `free_balance` return value.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    pub fn set_balance(&self, balance: Result<u128, String>) {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.balance.lock().unwrap() = balance;
        }
    }

    /// Script the next `sync_status` return value.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    pub fn set_sync(&self, status: Result<SyncStatus, String>) {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.sync.lock().unwrap() = status;
        }
    }

    /// How many times `free_balance` has been called.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    #[must_use]
    pub fn balance_calls(&self) -> usize {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.balance_calls.lock().unwrap()
        }
    }

    /// How many times `sync_status` has been called.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    #[must_use]
    pub fn sync_calls(&self) -> usize {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.sync_calls.lock().unwrap()
        }
    }

    /// Script the next `declare_participation` return value.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    pub fn set_participation_result(&self, result: Result<ParticipationOutcome, ChainError>) {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.participation_result.lock().unwrap() = result;
        }
    }

    /// How many times `declare_participation` has been called.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    #[must_use]
    pub fn participation_calls(&self) -> usize {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            self.participations.lock().unwrap().len()
        }
    }

    /// Drain and return captured participation qblock ids.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    #[must_use]
    pub fn take_participations(&self) -> Vec<u64> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            std::mem::take(&mut *self.participations.lock().unwrap())
        }
    }

    /// Script the next `file_descriptor` return value.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    pub fn set_descriptor_result(&self, result: Result<DescriptorOutcome, ChainError>) {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.descriptor_result.lock().unwrap() = result;
        }
    }

    /// How many times `file_descriptor` has been called.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    #[must_use]
    pub fn descriptor_calls(&self) -> usize {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            self.descriptors.lock().unwrap().len()
        }
    }

    /// Script whether `QuantumPow.Miners` already holds the signing account.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    pub fn set_registered(&self, registered: bool) {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.registered.lock().unwrap() = registered;
        }
    }

    /// Script the next `ensure_miner_registered` submit result.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    pub fn set_registration_result(&self, result: Result<RegistrationOutcome, ChainError>) {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.registration_result.lock().unwrap() = result;
        }
    }

    /// How many times `ensure_miner_registered` has been called.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    #[must_use]
    pub fn registration_calls(&self) -> usize {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.registration_calls.lock().unwrap()
        }
    }

    /// How many `register_miner` extrinsics the fake chain was asked to submit.
    /// An already-registered account submits none.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    #[must_use]
    pub fn registration_submits(&self) -> usize {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.registration_submits.lock().unwrap()
        }
    }

    /// Drain and return the accounts passed to `free_balance`, in call order.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    #[must_use]
    pub fn take_balance_accounts(&self) -> Vec<[u8; 32]> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            std::mem::take(&mut *self.balance_accounts.lock().unwrap())
        }
    }

    /// Drain and return captured descriptor payloads.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    #[must_use]
    pub fn take_descriptors(&self) -> Vec<NodeDescriptorV2Input> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            std::mem::take(&mut *self.descriptors.lock().unwrap())
        }
    }

    /// Script the value `fetch_miner_info` returns.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    pub fn set_miner_info(&self, info: Option<MinerInfo>) {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.miner_info.lock().unwrap() = info;
        }
    }
}

#[async_trait]
impl ChainClient for FakeChain {
    async fn fetch_mining_snapshot(
        &self,
        _at: Option<[u8; 32]>,
        _miner_account: [u8; 32],
        _topology_hash: Option<[u8; 32]>,
    ) -> Result<Option<MiningSnapshot>, ChainError> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            Ok(self.snapshot.lock().unwrap().clone())
        }
    }

    async fn fetch_mempool_orders(
        &self,
        _miner_account: [u8; 32],
    ) -> Result<Vec<JobOrder>, ChainError> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            Ok(self.orders.lock().unwrap().clone())
        }
    }

    async fn submit_proof(&self, proof: &Proof) -> Result<SubmitReceipt, ChainError> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            self.submitted.lock().unwrap().push(proof.clone());
            // Can't move out of Mutex guard for Result with ChainError (not Clone);
            // reconstruct Success/Retry/etc from a stored pattern.
            match &*self.submit_result.lock().unwrap() {
                Ok(r) => Ok(*r),
                Err(ChainError::Unavailable(s)) => Err(ChainError::Unavailable(s.clone())),
                Err(ChainError::Decode(s)) => Err(ChainError::Decode(s.clone())),
                Err(ChainError::Submit(s)) => Err(ChainError::Submit(s.clone())),
            }
        }
    }

    async fn ensure_miner_registered(&self) -> Result<RegistrationOutcome, ChainError> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.registration_calls.lock().unwrap() += 1;
            // Mirrors the real client: read the map first, submit only when the
            // account is absent.
            if *self.registered.lock().unwrap() {
                return Ok(RegistrationOutcome::AlreadyRegistered);
            }
            *self.registration_submits.lock().unwrap() += 1;
            match &*self.registration_result.lock().unwrap() {
                Ok(o) => {
                    *self.registered.lock().unwrap() = true;
                    Ok(*o)
                }
                Err(ChainError::Unavailable(s)) => Err(ChainError::Unavailable(s.clone())),
                Err(ChainError::Decode(s)) => Err(ChainError::Decode(s.clone())),
                Err(ChainError::Submit(s)) => Err(ChainError::Submit(s.clone())),
            }
        }
    }

    async fn file_descriptor(
        &self,
        descriptor: &NodeDescriptorV2Input,
    ) -> Result<DescriptorOutcome, ChainError> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            self.descriptors.lock().unwrap().push(descriptor.clone());
            match &*self.descriptor_result.lock().unwrap() {
                Ok(o) => Ok(*o),
                Err(ChainError::Unavailable(s)) => Err(ChainError::Unavailable(s.clone())),
                Err(ChainError::Decode(s)) => Err(ChainError::Decode(s.clone())),
                Err(ChainError::Submit(s)) => Err(ChainError::Submit(s.clone())),
            }
        }
    }

    async fn declare_participation(
        &self,
        qblock_id: u64,
    ) -> Result<ParticipationOutcome, ChainError> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            self.participations.lock().unwrap().push(qblock_id);
            match &*self.participation_result.lock().unwrap() {
                Ok(o) => Ok(*o),
                Err(ChainError::Unavailable(s)) => Err(ChainError::Unavailable(s.clone())),
                Err(ChainError::Decode(s)) => Err(ChainError::Decode(s.clone())),
                Err(ChainError::Submit(s)) => Err(ChainError::Submit(s.clone())),
            }
        }
    }

    async fn fetch_latest_qblock_id(&self) -> Result<Option<u64>, ChainError> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            Ok(*self.qblock_id.lock().unwrap())
        }
    }

    async fn fetch_decay_params(
        &self,
        _topology_hash: [u8; 32],
    ) -> Result<Option<DecayParams>, ChainError> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            Ok(self.decay_params.lock().unwrap().clone())
        }
    }

    async fn fetch_miner_info(&self, _account: [u8; 32]) -> Result<Option<MinerInfo>, ChainError> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            Ok(*self.miner_info.lock().unwrap())
        }
    }
}

#[async_trait]
impl BalanceSource for FakeChain {
    async fn free_balance(&self, account: [u8; 32]) -> Result<u128, String> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.balance_calls.lock().unwrap() += 1;
            self.balance_accounts.lock().unwrap().push(account);
            self.balance.lock().unwrap().clone()
        }
    }
}

#[async_trait]
impl SyncSource for FakeChain {
    async fn sync_status(&self) -> Result<SyncStatus, ChainError> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            *self.sync_calls.lock().unwrap() += 1;
            self.sync
                .lock()
                .unwrap()
                .clone()
                .map_err(ChainError::Unavailable)
        }
    }
}

//! Scripted chain for tests: fixed snapshot, optional mempool order, captured submits.

use super::{
    ChainClient, ChainError, DecayParams, JobOrder, MiningSnapshot, Proof, QBlockRecord,
    SubmitAction, TopologyInputs,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Mutex;

/// Test double: returns a scripted snapshot / order and records submits.
pub struct FakeChain {
    snapshot: Mutex<Option<MiningSnapshot>>,
    orders: Mutex<Vec<JobOrder>>,
    /// Captured proofs from [`ChainClient::submit_proof`].
    pub submitted: Mutex<Vec<Proof>>,
    /// Optional scripted submit result (default Success).
    submit_result: Mutex<Result<SubmitAction, ChainError>>,
    /// Scripted `latest_qblock_id` (default `None`).
    qblock_id: Mutex<Option<u64>>,
    /// Scripted `fetch_decay_params` (default `None`).
    decay_params: Mutex<Option<DecayParams>>,
    /// Scripted `fetch_qblock_by_id` results, keyed by qblock id.
    qblocks: Mutex<HashMap<u64, QBlockRecord>>,
    /// Scripted `fetch_topology_meta` results, keyed by topology hash.
    topologies: Mutex<HashMap<[u8; 32], TopologyInputs>>,
}

impl FakeChain {
    /// Build a fake chain with a fixed snapshot and optional single order.
    #[must_use]
    pub fn new(snapshot: MiningSnapshot, order: Option<JobOrder>) -> Self {
        Self {
            snapshot: Mutex::new(Some(snapshot)),
            orders: Mutex::new(order.into_iter().collect()),
            submitted: Mutex::new(Vec::new()),
            submit_result: Mutex::new(Ok(SubmitAction::Success)),
            qblock_id: Mutex::new(None),
            decay_params: Mutex::new(None),
            qblocks: Mutex::new(HashMap::new()),
            topologies: Mutex::new(HashMap::new()),
        }
    }

    /// Script a qblock returned by `fetch_qblock_by_id(id)`.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    pub fn set_qblock(&self, id: u64, rec: QBlockRecord) {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            let _ = self.qblocks.lock().unwrap().insert(id, rec);
        }
    }

    /// Script a topology returned by `fetch_topology_meta(hash)`.
    ///
    /// # Panics
    /// Panics if a prior holder poisoned this mutex.
    pub fn set_topology(&self, hash: [u8; 32], inputs: TopologyInputs) {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            let _ = self.topologies.lock().unwrap().insert(hash, inputs);
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

    async fn submit_proof(&self, proof: &Proof) -> Result<SubmitAction, ChainError> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            self.submitted.lock().unwrap().push(proof.clone());
            // Can't move out of Mutex guard for Result with ChainError (not Clone);
            // reconstruct Success/Retry/etc from a stored pattern.
            match &*self.submit_result.lock().unwrap() {
                Ok(a) => Ok(*a),
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

    async fn fetch_qblock_by_id(&self, qblock_id: u64) -> Result<Option<QBlockRecord>, ChainError> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            Ok(self.qblocks.lock().unwrap().get(&qblock_id).cloned())
        }
    }

    async fn fetch_topology_meta(
        &self,
        topology_hash: [u8; 32],
    ) -> Result<Option<TopologyInputs>, ChainError> {
        #[expect(
            clippy::unwrap_used,
            reason = "test double; Mutex poison is a test failure"
        )]
        {
            Ok(self.topologies.lock().unwrap().get(&topology_hash).cloned())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> MiningSnapshot {
        MiningSnapshot {
            last_proof_block_hash: [0u8; 32],
            topology_hash: vec![9u8; 32],
            nodes: vec![0, 1],
            edges: vec![(0, 1)],
            allowed_h_milli: vec![-1000, 0, 1000],
            allowed_j_milli: vec![-1000, 1000],
            allowed_spin_milli: vec![-1000, 1000],
            min_solutions: 1,
            max_energy_milli: i64::MAX,
            min_diversity_milli: 0,
            block_number: 0,
        }
    }

    fn sample_qblock() -> QBlockRecord {
        QBlockRecord {
            qblock_id: 1,
            miner: [1; 32],
            salt: [2; 32],
            energy_milli: -1,
            reward: 0,
            submitted_at: 0,
            last_proof_block_hash: [0; 32],
            topology_hash: [9; 32],
            device_access_time_us: 0,
            max_energy_milli: -1,
            min_solutions: 1,
            min_diversity_milli: 0,
            nonce: [7; 32],
        }
    }

    #[tokio::test]
    async fn fake_returns_scripted_qblock_and_topology() {
        let fake = FakeChain::new(sample_snapshot(), None);
        assert!(fake.fetch_qblock_by_id(1).await.unwrap().is_none());
        let rec = sample_qblock();
        fake.set_qblock(1, rec.clone());
        assert_eq!(fake.fetch_qblock_by_id(1).await.unwrap(), Some(rec));
        assert!(fake.fetch_topology_meta([9; 32]).await.unwrap().is_none());
        let topo = TopologyInputs {
            nodes: vec![0, 1],
            edges: vec![(0, 1)],
            allowed_h_milli: vec![-1000, 0, 1000],
            allowed_j_milli: vec![-1000, 1000],
            allowed_spin_milli: vec![-1000, 1000],
        };
        fake.set_topology([9; 32], topo.clone());
        assert_eq!(fake.fetch_topology_meta([9; 32]).await.unwrap(), Some(topo));
    }
}

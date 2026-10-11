//! Mempool job-order types and solver eligibility.

use super::scale_types::{JobMode, SolverType};

/// A mempool job order ready for conversion to a wire `Job`.
#[derive(Debug, Clone)]
pub struct JobOrder {
    /// Opaque order id (typically LE-encoded `u64` from chain storage).
    pub order_id: Vec<u8>,
    /// On-chain solver access policy.
    pub mode: JobMode,
    /// Topology node ids.
    pub nodes: Vec<u32>,
    /// Topology undirected edges as `(u, v)` node-id pairs.
    pub edges: Vec<(u32, u32)>,
    /// Per-node linear fields in milli units (aligned with `nodes`).
    pub h_milli: Vec<i32>,
    /// Per-edge couplings in milli units (aligned with `edges`).
    pub j_milli: Vec<i32>,
    /// Optional energy gate (milli); `None` when the order has no min energy.
    pub min_energy_milli: Option<i64>,
    /// Optional diversity gate (milli).
    pub min_diversity_milli: Option<u32>,
    /// Optional minimum valid-solution count.
    pub min_solutions: Option<u32>,
    /// Soft deadline as absolute Unix milliseconds, projected from the blocks
    /// left before the order expires. Miners compare it with their own clock.
    pub deadline_ms: u64,
}

impl JobOrder {
    /// Mirror the mempool pallet's account-or-type access rule.
    ///
    /// The caller must supply the account's registered mempool solver type.
    #[must_use]
    pub fn is_eligible_for(&self, solver_account: &[u8; 32], solver_type: SolverType) -> bool {
        match &self.mode {
            JobMode::Open => true,
            JobMode::Bid {
                miners,
                miner_types,
            } => {
                let account_match = miners
                    .as_ref()
                    .is_some_and(|allowed| allowed.contains(solver_account));
                let type_tag = match solver_type {
                    SolverType::Cpu => 0,
                    SolverType::Gpu => 1,
                    SolverType::QpuDwave => 2,
                    SolverType::QpuIbm => 3,
                    SolverType::QpuIonq => 4,
                    SolverType::QpuPasqal => 5,
                    SolverType::Asic => 6,
                };
                let type_match = miner_types
                    .as_ref()
                    .is_some_and(|allowed| allowed.contains(&type_tag));
                account_match || type_match
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::JobOrder;
    use crate::chain::scale_types::{JobMode, SolverType};

    fn order(mode: JobMode) -> JobOrder {
        JobOrder {
            order_id: 1u64.to_le_bytes().to_vec(),
            mode,
            nodes: vec![0, 1],
            edges: vec![(0, 1)],
            h_milli: vec![0, 0],
            j_milli: vec![-1000],
            min_energy_milli: None,
            min_diversity_milli: None,
            min_solutions: None,
            deadline_ms: 0,
        }
    }

    #[test]
    fn excluding_bid_is_ineligible() {
        let bid = order(JobMode::Bid {
            miners: Some(vec![[2; 32]]),
            miner_types: Some(vec![1]),
        });
        assert!(!bid.is_eligible_for(&[1; 32], SolverType::Cpu));
    }

    #[test]
    fn open_order_accepts_each_registered_solver_type() {
        let open = order(JobMode::Open);
        for solver_type in [
            SolverType::Cpu,
            SolverType::Gpu,
            SolverType::QpuDwave,
            SolverType::QpuIbm,
            SolverType::QpuIonq,
            SolverType::QpuPasqal,
            SolverType::Asic,
        ] {
            assert!(open.is_eligible_for(&[1; 32], solver_type));
        }
    }

    #[test]
    fn either_bid_filter_can_admit_the_solver() {
        let by_account = order(JobMode::Bid {
            miners: Some(vec![[1; 32]]),
            miner_types: Some(vec![1]),
        });
        assert!(by_account.is_eligible_for(&[1; 32], SolverType::Cpu));

        let by_type = order(JobMode::Bid {
            miners: Some(vec![[2; 32]]),
            miner_types: Some(vec![0]),
        });
        assert!(by_type.is_eligible_for(&[1; 32], SolverType::Cpu));
    }

    #[test]
    fn each_bid_type_tag_matches_only_its_registered_type() {
        let types = [
            (0, SolverType::Cpu),
            (1, SolverType::Gpu),
            (2, SolverType::QpuDwave),
            (3, SolverType::QpuIbm),
            (4, SolverType::QpuIonq),
            (5, SolverType::QpuPasqal),
            (6, SolverType::Asic),
        ];
        for (allowed_tag, allowed_type) in types {
            let bid = order(JobMode::Bid {
                miners: None,
                miner_types: Some(vec![allowed_tag]),
            });
            for (_, candidate_type) in types {
                assert_eq!(
                    bid.is_eligible_for(&[1; 32], candidate_type),
                    candidate_type == allowed_type,
                );
            }
        }
    }

    #[test]
    fn empty_and_absent_bid_filters_admit_nobody() {
        for (miners, miner_types) in [
            (None, None),
            (Some(Vec::new()), None),
            (None, Some(Vec::new())),
            (Some(Vec::new()), Some(Vec::new())),
        ] {
            let bid = order(JobMode::Bid {
                miners,
                miner_types,
            });
            assert!(!bid.is_eligible_for(&[1; 32], SolverType::Cpu));
        }

        let by_account = order(JobMode::Bid {
            miners: Some(vec![[1; 32]]),
            miner_types: Some(Vec::new()),
        });
        assert!(by_account.is_eligible_for(&[1; 32], SolverType::Cpu));

        let by_type = order(JobMode::Bid {
            miners: Some(Vec::new()),
            miner_types: Some(vec![0]),
        });
        assert!(by_type.is_eligible_for(&[1; 32], SolverType::Cpu));
    }
}

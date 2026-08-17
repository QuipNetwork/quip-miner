//! Convert mempool `JobOrder`s to wire `Job`s with inline `EdgeList`.

use crate::chain::mempool::JobOrder;
use quip_proto::v1::{ising_problem, EdgeList, IsingProblem, Job, JobKind, Provenance};
use quip_protocol::wire::encode_i32_le;

/// Convert a mempool order into an `ISING_SAMPLE` job with inline edges.
///
/// `generation = 0` (mempool is not cancelled by `PoW` generation swaps).
/// Missing gate floors default to 0.
#[must_use]
pub fn job_order_to_job(order: &JobOrder) -> Job {
    let (u, v): (Vec<u32>, Vec<u32>) = order.edges.iter().copied().unzip();
    Job {
        job_id: order.order_id.clone(),
        kind: JobKind::IsingSample as i32,
        generation: 0,
        deadline_ms: order.deadline_ms,
        ising: Some(IsingProblem {
            graph: Some(ising_problem::Graph::Edges(EdgeList { u, v })),
            h_milli_le32: encode_i32_le(&order.h_milli),
            j_milli_le32: encode_i32_le(&order.j_milli),
            num_reads: 0,
            num_sweeps: 0,
            anneal_time_us: 0,
        }),
        provenance: Some(Provenance {
            is_pow: false,
            order_id: order.order_id.clone(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_proto::v1::ising_problem;

    fn sample_order() -> JobOrder {
        JobOrder {
            order_id: b"order-1".to_vec(),
            nodes: vec![0, 1],
            edges: vec![(0, 1)],
            h_milli: vec![1000, -1000],
            j_milli: vec![500],
            min_energy_milli: Some(-1000),
            min_diversity_milli: None,
            min_solutions: Some(1),
            deadline_ms: 9_999_999,
        }
    }

    #[test]
    fn converts_to_inline_edge_job() {
        let job = job_order_to_job(&sample_order());
        assert_eq!(job.generation, 0);
        assert!(!job.provenance.as_ref().unwrap().is_pow);
        assert_eq!(job.provenance.as_ref().unwrap().order_id, b"order-1");
        let ising = job.ising.unwrap();
        assert!(matches!(ising.graph, Some(ising_problem::Graph::Edges(_))));
    }
}

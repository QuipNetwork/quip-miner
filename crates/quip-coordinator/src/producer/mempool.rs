//! Convert mempool `JobOrder`s to wire `Job`s with inline `EdgeList`.

use crate::chain::mempool::JobOrder;
use quip_proto::v1::{ising_problem, EdgeList, IsingProblem, Job, JobKind, Provenance};
use quip_protocol::wire::encode_i32_le;
use std::collections::HashMap;

/// Convert a mempool order into an `ISING_SAMPLE` job with inline edges.
///
/// `order.edges` names topology node *labels*, the ids the pallet stores and
/// resolves through its own `TopologyIndex`. The wire's inline `EdgeList` is
/// positional: each endpoint is an index into the job's own site list, which
/// here is `order.nodes` in received order. Every endpoint is therefore mapped
/// through the same label-to-position table the `PoW` path caches per session,
/// rather than forwarded verbatim.
/// Forwarding a label verbatim leaves an endpoint naming a site the order did
/// not place on: past the end of `h` the miner refuses the job as `Malformed`,
/// and the coordinator's own `energy_in_place` drops every row it scores, so
/// the order is never answered.
///
/// `h_milli` and `j_milli` already arrive in that positional order (aligned
/// with `nodes` and `edges` respectively), so the mapping touches the endpoints
/// alone and the consensus `j`-zip invariant is preserved.
///
/// `generation = 0` (mempool is not cancelled by `PoW` generation swaps).
/// Missing gate floors default to 0.
///
/// Returns `None` when `order.nodes` cannot index `order.edges`: a repeated
/// node id, which collapses two positions onto one and misaligns every site
/// after it, or an edge endpoint that is not a node of the order at all.
/// Neither order describes a graph, so the feeder skips it with a warning
/// rather than staging a job whose only outcomes are a reject or a wrong
/// answer.
#[must_use]
pub fn job_order_to_job(order: &JobOrder) -> Option<Job> {
    let mut pos: HashMap<u32, u32> = HashMap::with_capacity(order.nodes.len());
    for (i, &node) in order.nodes.iter().enumerate() {
        let index = u32::try_from(i).ok()?;
        if pos.insert(node, index).is_some() {
            return None;
        }
    }
    let mut u = Vec::with_capacity(order.edges.len());
    let mut v = Vec::with_capacity(order.edges.len());
    for &(a, b) in &order.edges {
        u.push(*pos.get(&a)?);
        v.push(*pos.get(&b)?);
    }
    Some(Job {
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
    })
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

    /// A gapped node list: the labels an order places on are whatever the
    /// topology calls them, and only `nodes: [0, 1]` makes labels and positions
    /// coincide. This is the order shape that was silently never answered.
    fn gapped_order() -> JobOrder {
        JobOrder {
            nodes: vec![0, 12, 30],
            edges: vec![(0, 12), (12, 30)],
            h_milli: vec![1000, -1000, 250],
            j_milli: vec![500, -500],
            ..sample_order()
        }
    }

    fn edges_of(job: &Job) -> EdgeList {
        match job.ising.as_ref().unwrap().graph.as_ref().unwrap() {
            ising_problem::Graph::Edges(e) => e.clone(),
            ising_problem::Graph::TopologyHash(_) => panic!("expected inline edges"),
        }
    }

    #[test]
    fn converts_to_inline_edge_job() {
        let job = job_order_to_job(&sample_order()).unwrap();
        assert_eq!(job.generation, 0);
        assert!(!job.provenance.as_ref().unwrap().is_pow);
        assert_eq!(job.provenance.as_ref().unwrap().order_id, b"order-1");
        let ising = job.ising.unwrap();
        assert!(matches!(ising.graph, Some(ising_problem::Graph::Edges(_))));
    }

    #[test]
    fn maps_edge_labels_to_dense_positions() {
        let job = job_order_to_job(&gapped_order()).unwrap();
        let edges = edges_of(&job);
        assert_eq!(edges.u, vec![0, 1]);
        assert_eq!(edges.v, vec![1, 2]);
    }

    /// Mapping is one-for-one: no endpoint is dropped or reordered, so the
    /// consensus `j`-zip invariant (one coupling per edge, in edge order)
    /// survives it.
    #[test]
    fn maps_every_edge_exactly_once() {
        let order = gapped_order();
        let edges = edges_of(&job_order_to_job(&order).unwrap());
        assert_eq!(edges.u.len(), order.edges.len());
        assert_eq!(edges.v.len(), order.edges.len());
        assert_eq!(edges.u.len(), order.j_milli.len());
    }

    #[test]
    fn rejects_an_edge_endpoint_that_is_not_a_node() {
        let order = JobOrder {
            edges: vec![(0, 99)],
            ..gapped_order()
        };
        assert!(job_order_to_job(&order).is_none());
    }

    #[test]
    fn rejects_a_repeated_node_id() {
        let order = JobOrder {
            nodes: vec![0, 12, 12],
            ..gapped_order()
        };
        assert!(job_order_to_job(&order).is_none());
    }
}

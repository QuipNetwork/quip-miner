//! Drive-mode harness: a pull-based `JobSource` seam feeding synthetic work
//! to a spawned miner, so it can be exercised and measured without a chain.
//!
//! Scope (sub-project #1 of the v0.3 coordinator work): golden-draw random
//! problems and a JSONL replay list, both file/in-memory and bounded. Network
//! sources and by-hash topology are later sub-projects; they plug into the
//! same `JobSource` / topology-provider seams.

pub mod harness;
pub mod list_source;
pub mod random_source;
pub mod report;
pub mod topology_spec;

pub use harness::{run_drive, DriveManyParams, DriveManyReport};
pub use list_source::{ListSource, ListSourceError};
pub use random_source::RandomSource;
pub use report::{aggregate, print_table, write_jsonl, Aggregate, JobRow};
pub use topology_spec::{parse_topology_spec, TopologySpec, TopologySpecError};

use quip_proto::v1::Job;

/// Pull-based seam feeding jobs into the drive harness.
///
/// Sync is sufficient for sub-project #1: random draw and file read are both
/// in-memory and bounded (`--count` / a finite JSONL file). The harness drains
/// a source to exhaustion before staging its jobs with the router.
pub trait JobSource {
    /// Next job to route, or `None` when the source is exhausted.
    fn next_job(&mut self) -> Option<Job>;
}

/// One-shot source wrapping a fixed list of jobs.
pub struct VecSource(std::vec::IntoIter<Job>);

impl VecSource {
    /// Wrap `jobs` as a [`JobSource`] that yields them in order once.
    #[must_use]
    pub fn new(jobs: Vec<Job>) -> Self {
        Self(jobs.into_iter())
    }
}

impl JobSource for VecSource {
    fn next_job(&mut self) -> Option<Job> {
        self.0.next()
    }
}

/// Drain every job a bounded source has to offer.
pub fn drain_all(source: &mut dyn JobSource) -> Vec<Job> {
    let mut jobs = Vec::new();
    while let Some(j) = source.next_job() {
        jobs.push(j);
    }
    jobs
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_proto::v1::{IsingProblem, JobKind, Provenance};

    fn job(n: u64) -> Job {
        Job {
            job_id: n.to_le_bytes().to_vec(),
            kind: JobKind::IsingSample as i32,
            generation: n,
            deadline_ms: 9_999_999,
            ising: Some(IsingProblem {
                graph: None,
                h_milli_le32: vec![0; 4],
                j_milli_le32: vec![],
                num_reads: 0,
                num_sweeps: 0,
                anneal_time_us: 0,
            }),
            provenance: Some(Provenance {
                is_pow: true,
                order_id: vec![],
            }),
        }
    }

    #[test]
    fn vec_source_yields_in_order_then_exhausts() {
        let mut src = VecSource::new(vec![job(1), job(2), job(3)]);
        assert_eq!(src.next_job().unwrap().generation, 1);
        assert_eq!(src.next_job().unwrap().generation, 2);
        assert_eq!(src.next_job().unwrap().generation, 3);
        assert!(src.next_job().is_none());
    }

    #[test]
    fn drain_all_collects_every_job() {
        let mut src = VecSource::new(vec![job(1), job(2)]);
        let jobs = drain_all(&mut src);
        assert_eq!(jobs.len(), 2);
    }
}

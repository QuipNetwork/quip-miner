//! Shared miner harness for the v0.3 Ising miners.
//!
//! Holds the generic gRPC session loop, job validation, the base Ising types,
//! the CSR representation used by GPU backends, and the beta schedule. Each
//! miner provides a [`Sampler`] and calls [`run`]; the harness owns everything
//! else (Hello/Welcome, Configure, credits, Reject reasons, Status, Shutdown,
//! idle timeout, exit codes).

pub mod adapt;
pub mod beta;
pub mod cli;
pub mod config;
pub mod csr;
mod display;
pub mod ising;
mod job;
pub mod logging;
mod session;

pub use cli::CommonArgs;
pub use csr::CsrGraph;
pub use ising::{Algorithm, IsingGraph, SampleParams, SamplerResult};
pub use session::{run, BackendIdentity, OpenError};

use quip_proto::v1::RejectReason;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Control-plane cancellation watermark. The session bumps it on `Cancel`; the
/// sampler reads it to skip jobs from abandoned generations. Cheap to clone
/// (one `Arc<AtomicU64>`); monotonic, so out-of-order or repeated cancels are
/// idempotent.
#[derive(Clone, Default)]
pub struct CancelGuard(Arc<AtomicU64>);

impl CancelGuard {
    /// Abandon every generation `<= generation`.
    pub fn cancel_through(&self, generation: u64) {
        let _ = self.0.fetch_max(generation, Ordering::Relaxed);
    }

    /// True when this job's generation has been abandoned. Generation `0`
    /// (mempool jobs) is never reseed-cancelled — mirrors the coordinator
    /// router's `cancel` (`generation <= max && generation != 0`).
    #[must_use]
    pub fn is_cancelled(&self, generation: u64) -> bool {
        generation != 0 && generation <= self.0.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod cancel_tests {
    use super::CancelGuard;

    #[test]
    fn cancel_guard_marks_generations_at_or_below_watermark() {
        let g = CancelGuard::default();
        assert!(!g.is_cancelled(5)); // nothing cancelled yet
        g.cancel_through(5);
        assert!(g.is_cancelled(5)); // <= watermark
        assert!(g.is_cancelled(4));
        assert!(!g.is_cancelled(6)); // > watermark, still live
    }

    #[test]
    fn cancel_guard_never_cancels_generation_zero() {
        let g = CancelGuard::default();
        g.cancel_through(10);
        assert!(!g.is_cancelled(0)); // mempool jobs are never reseed-cancelled
    }

    #[test]
    fn cancel_guard_watermark_is_monotonic() {
        let g = CancelGuard::default();
        g.cancel_through(7);
        g.cancel_through(3); // a lower value must not lower the watermark
        assert!(g.is_cancelled(7));
        assert!(g.is_cancelled(4));
    }
}

/// One job entering the streaming sampler.
pub struct StreamJob {
    /// Opaque job id from the coordinator (echoed on Result/Reject).
    pub job_id: Vec<u8>,
    /// Wire-parsed Ising problem for this job.
    pub graph: IsingGraph,
    /// Resolved sampling knobs for this job.
    pub params: SampleParams,
    /// Reseed round the job belongs to; what a `Cancel` invalidates. `0` for
    /// mempool jobs (never reseed-cancelled).
    pub generation: u64,
}

/// One job leaving the streaming sampler, in completion order.
pub struct StreamResult {
    /// Opaque job id, matching the inbound [`StreamJob`].
    pub job_id: Vec<u8>,
    /// Completed samples, reject reason, or cancelled generation.
    pub outcome: StreamOutcome,
    /// Per-model device/sample time in microseconds, reported in `SamplerMeta`.
    pub device_access_time_us: u64,
}

/// Outcome of one streamed job.
pub enum StreamOutcome {
    /// Ran to completion (or a real reject).
    Completed(Result<Vec<SamplerResult>, RejectReason>),
    /// Abandoned because its generation was cancelled; the coordinator has moved
    /// on, so nothing is sent upstream — only the local credit is refunded to
    /// keep pipeline depth for the live round.
    Cancelled,
}

/// A backend that samples Ising problems for the miner harness.
///
/// Implementations own their device and algorithm. Only [`sample`](Sampler::sample)
/// is required; the other methods default to a no-governor, uncapped backend
/// (the CPU miner's shape).
pub trait Sampler: Send + Sync + 'static {
    /// Sample one job. Device errors map to a reject reason
    /// (`Overloaded`, `TooLarge`, …).
    ///
    /// # Errors
    ///
    /// Returns a [`RejectReason`] when the backend cannot complete the job
    /// (device overload, size limit, malformed device state, …).
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, RejectReason>;

    /// Stream-process jobs: pull from `jobs`, keep up to [`stream_width`] models
    /// in flight, emit each result to `out` in completion order. Blocks until
    /// `jobs` closes and all in-flight finish. Runs on a blocking thread (uses
    /// `blocking_recv`/`blocking_send`), so it must not be called from an async
    /// task directly. Default: serial loop over [`sample`](Sampler::sample).
    ///
    /// [`stream_width`]: Sampler::stream_width
    fn sample_stream(
        &self,
        mut jobs: tokio::sync::mpsc::Receiver<StreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        cancel: CancelGuard,
    ) {
        while let Some(j) = jobs.blocking_recv() {
            // Skip a job the coordinator abandoned on reseed; refund the credit
            // (via the session's Cancelled handling) so the pipeline keeps depth.
            // The serial default can only check at dequeue; backends that own a
            // sweep/read loop poll `cancel` at their finer checkpoints.
            if cancel.is_cancelled(j.generation) {
                if out
                    .blocking_send(StreamResult {
                        job_id: j.job_id,
                        outcome: StreamOutcome::Cancelled,
                        device_access_time_us: 0,
                    })
                    .is_err()
                {
                    break;
                }
                continue;
            }
            if self.should_throttle() {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            let t0 = std::time::Instant::now();
            let result = self.sample(&j.graph, &j.params);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "per-job sample duration in micros fits u64 for any realistic device runtime"
            )]
            let device_access_time_us = t0.elapsed().as_micros() as u64;
            if out
                .blocking_send(StreamResult {
                    job_id: j.job_id,
                    outcome: StreamOutcome::Completed(result),
                    device_access_time_us,
                })
                .is_err()
            {
                break;
            }
        }
    }

    /// Number of models the backend keeps in flight. Default 1 (serial).
    fn stream_width(&self) -> usize {
        1
    }

    /// Current utilization for Status messages. `0.0` when no governor.
    fn utilization(&self) -> f64 {
        0.0
    }

    /// Whether to briefly back off before the next job (governor backpressure).
    fn should_throttle(&self) -> bool {
        false
    }

    /// Largest `num_reads` this backend accepts. Defaults to no cap; a backend
    /// with a device-memory bound overrides it.
    fn max_reads(&self) -> u32 {
        u32::MAX
    }

    /// Apply this backend's configuration from `Configure.backend_toml` — the
    /// verbatim `config.toml` subsection the coordinator forwards. Called once
    /// when `Configure` arrives, before any job. Each backend parses against its
    /// own schema, applies recognized fields (config overrides CLI, see
    /// [`config::config_override`]), and warns on unknown fields
    /// ([`config::warn_unknown_fields`]). Default: no configurable settings (the
    /// CPU miner's shape).
    fn apply_config(&self, _backend_toml: &str) {}
}

#[cfg(test)]
mod stream_tests {
    use super::*;

    struct OneResultSampler;
    impl Sampler for OneResultSampler {
        fn sample(
            &self,
            graph: &IsingGraph,
            _params: &SampleParams,
        ) -> Result<Vec<SamplerResult>, RejectReason> {
            Ok(vec![SamplerResult {
                spins: vec![1i8; graph.h.len()],
                energy_milli: 0,
            }])
        }
    }

    fn tiny_graph() -> IsingGraph {
        IsingGraph::new(vec![1.0, -1.0], vec![1.0], vec![(0, 1)])
    }

    #[test]
    fn default_sample_stream_returns_every_result_once() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("rt");
        let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(8);
        let (res_tx, mut res_rx) = tokio::sync::mpsc::channel::<StreamResult>(8);

        let worker = std::thread::spawn(move || {
            OneResultSampler.sample_stream(job_rx, res_tx, CancelGuard::default());
        });

        rt.block_on(async {
            for i in 0u8..5 {
                job_tx
                    .send(StreamJob {
                        job_id: vec![i],
                        graph: tiny_graph(),
                        params: SampleParams::default(),
                        generation: 1,
                    })
                    .await
                    .expect("send job");
            }
            drop(job_tx); // close the stream so the worker exits

            let mut seen: Vec<u8> = Vec::new();
            while let Some(r) = res_rx.recv().await {
                assert!(matches!(r.outcome, StreamOutcome::Completed(Ok(_))));
                #[expect(
                    clippy::indexing_slicing,
                    reason = "test jobs use single-byte ids (vec![i])"
                )]
                seen.push(r.job_id[0]);
            }
            seen.sort_unstable();
            assert_eq!(seen, vec![0, 1, 2, 3, 4]);
        });
        worker.join().expect("worker join");
    }

    struct CountingSampler(Arc<std::sync::atomic::AtomicUsize>);
    impl Sampler for CountingSampler {
        fn sample(
            &self,
            graph: &IsingGraph,
            _params: &SampleParams,
        ) -> Result<Vec<SamplerResult>, RejectReason> {
            use std::sync::atomic::Ordering;
            let _ = self.0.fetch_add(1, Ordering::SeqCst);
            Ok(vec![SamplerResult {
                spins: vec![1i8; graph.h.len()],
                energy_milli: 0,
            }])
        }
    }

    #[test]
    fn sample_stream_skips_cancelled_generation_without_sampling() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("rt");
        let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(8);
        let (res_tx, mut res_rx) = tokio::sync::mpsc::channel::<StreamResult>(8);
        let calls = Arc::new(AtomicUsize::new(0));
        let cancel = CancelGuard::default();
        cancel.cancel_through(3); // generations 1..=3 abandoned

        let sampler = CountingSampler(Arc::clone(&calls));
        let worker = std::thread::spawn(move || sampler.sample_stream(job_rx, res_tx, cancel));

        rt.block_on(async {
            job_tx
                .send(StreamJob {
                    job_id: vec![1],
                    graph: tiny_graph(),
                    params: SampleParams::default(),
                    generation: 2, // stale
                })
                .await
                .expect("send");
            job_tx
                .send(StreamJob {
                    job_id: vec![2],
                    graph: tiny_graph(),
                    params: SampleParams::default(),
                    generation: 5, // live
                })
                .await
                .expect("send");
            drop(job_tx);

            let mut cancelled: std::collections::HashMap<u8, bool> =
                std::collections::HashMap::new();
            while let Some(r) = res_rx.recv().await {
                #[expect(
                    clippy::indexing_slicing,
                    reason = "test jobs use single-byte ids (vec![1]/vec![2])"
                )]
                let _ =
                    cancelled.insert(r.job_id[0], matches!(r.outcome, StreamOutcome::Cancelled));
            }
            assert_eq!(cancelled.get(&1), Some(&true)); // stale -> Cancelled
            assert_eq!(cancelled.get(&2), Some(&false)); // live -> Completed
        });
        worker.join().expect("worker join");
        assert_eq!(calls.load(Ordering::SeqCst), 1); // sample ran only for the live job
    }
}

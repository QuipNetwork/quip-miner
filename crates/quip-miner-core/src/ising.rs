//! Base Ising problem type and per-job sampling knobs, shared by all backends.
//!
//! The base [`IsingGraph`] holds the wire-parsed `(h, j, edges)`. Backends
//! derive their own representation from it: the CPU miner builds an adjacency
//! list, the GPU miners build [`crate::csr::CsrGraph`].

/// Sampling algorithm selected by the binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    /// Metropolis-Hastings simulated annealing (neal default).
    Sa,
    /// Single-site heat-bath Gibbs along the same beta ladder.
    Gibbs,
}

/// Per-job sampling knobs.
#[derive(Clone, Debug)]
pub struct SampleParams {
    /// Independent reads (samples) to produce.
    pub num_reads: usize,
    /// Annealing sweeps per read.
    pub num_sweeps: usize,
    /// Sweeps spent at each beta rung.
    pub sweeps_per_beta: usize,
    /// Optional `(hot_beta, cold_beta)`. `None` → auto from biases.
    pub beta_range: Option<(f64, f64)>,
    /// PRNG seed for this job.
    pub seed: u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            num_reads: 1,
            num_sweeps: 64,
            sweeps_per_beta: 1,
            beta_range: None,
            seed: 0,
        }
    }
}

/// One completed read: spins in {-1,+1} and consensus milli-energy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SamplerResult {
    /// Spin configuration, one entry per variable, values in `{-1, +1}`.
    pub spins: Vec<i8>,
    /// Consensus energy of `spins`, in milli units.
    pub energy_milli: i64,
}

/// Wire-parsed Ising problem: dense biases, flat couplings, and edge list.
#[derive(Clone, Debug)]
pub struct IsingGraph {
    /// Linear biases, one per variable.
    pub h: Vec<f64>,
    /// Couplings aligned with `edges`.
    pub j: Vec<f64>,
    /// Undirected edge list `(u, v)` in received order.
    pub edges: Vec<(usize, usize)>,
}

impl IsingGraph {
    /// Store flat `h` / `j` / edge lists as the base problem.
    #[must_use]
    pub fn new(h: Vec<f64>, j: Vec<f64>, edges: Vec<(usize, usize)>) -> Self {
        Self { h, j, edges }
    }

    /// Number of variables (length of `h`).
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.h.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_nodes_counts_biases() {
        let g = IsingGraph::new(
            vec![1.0, -0.5, 0.0, 0.25],
            vec![1.0, -1.0, -1.0, 1.0],
            vec![(0, 1), (1, 2), (2, 3), (0, 3)],
        );
        assert_eq!(g.num_nodes(), 4);
        assert_eq!(g.edges.len(), 4);
    }
}

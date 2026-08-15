//! Drive-mode topology provider: parses a JSON topology-spec file into a
//! `Topology` + synthetic `MiningSnapshot` (chain-only fields defaulted).
//!
//! Sub-project #1 ships this file-spec provider only. Topology-by-hash from
//! the network is sub-project #2, behind the same seam (a `TopologySpec`).

use crate::chain::MiningSnapshot;
use crate::topology::{Topology, DEFAULT_SPIN_SET};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GatesJson {
    #[serde(default)]
    min_solutions: u32,
    #[serde(default = "default_max_energy_milli")]
    max_energy_milli: i64,
    #[serde(default)]
    min_diversity_milli: u32,
}

fn default_max_energy_milli() -> i64 {
    i64::MAX
}

impl Default for GatesJson {
    fn default() -> Self {
        Self {
            min_solutions: 0,
            max_energy_milli: default_max_energy_milli(),
            min_diversity_milli: 0,
        }
    }
}

#[derive(Debug, Deserialize)]
struct TopologySpecJson {
    nodes: Vec<u32>,
    #[serde(default)]
    edges: Vec<(u32, u32)>,
    allowed_h_milli: Vec<i32>,
    allowed_j_milli: Vec<i32>,
    /// Allowed solution-spin values; defaults to the binary set `±1000` when
    /// omitted, matching a chain topology with no explicit spin spec.
    #[serde(default)]
    allowed_spin_milli: Vec<i32>,
    #[serde(default)]
    gates: Option<GatesJson>,
}

/// Errors parsing/validating a drive-mode topology-spec file.
#[derive(Debug, PartialEq)]
pub enum TopologySpecError {
    /// JSON decode failure (message is the serde error).
    Json(String),
    /// `nodes` array was empty.
    EmptyNodes,
    /// `allowed_h_milli` array was empty.
    EmptyAllowedH,
    /// `allowed_j_milli` array was empty.
    EmptyAllowedJ,
    /// An `edges` endpoint is a node id not present in `nodes`. Edges reference
    /// native node ids (possibly sparse), which the miner and consensus scorer
    /// map to dense positions — so an id just has to exist in `nodes`.
    EdgeUnknownNode {
        /// Zero-based index of the bad edge in the `edges` array.
        edge_index: usize,
        /// Node id that was not present in `nodes`.
        endpoint: u32,
    },
}

impl std::fmt::Display for TopologySpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "invalid topology spec JSON: {e}"),
            Self::EmptyNodes => write!(f, "topology spec: nodes must be non-empty"),
            Self::EmptyAllowedH => {
                write!(f, "topology spec: allowed_h_milli must be non-empty")
            }
            Self::EmptyAllowedJ => {
                write!(f, "topology spec: allowed_j_milli must be non-empty")
            }
            Self::EdgeUnknownNode {
                edge_index,
                endpoint,
            } => write!(
                f,
                "topology spec: edge {edge_index} references node id {endpoint}, \
                 which is not in the topology"
            ),
        }
    }
}

impl std::error::Error for TopologySpecError {}

/// Parsed drive-mode topology spec: a `Topology` (for the wire message) plus
/// the golden-draw + gate parameters needed to build a synthetic
/// `MiningSnapshot`.
#[derive(Debug, Clone)]
pub struct TopologySpec {
    /// Topology wire message (nodes, edges, hash) sent to the miner.
    pub topology: Topology,
    /// Allowed h-field values (milli) for golden draw.
    pub allowed_h_milli: Vec<i32>,
    /// Allowed J-coupling values (milli) for golden draw.
    pub allowed_j_milli: Vec<i32>,
    /// Allowed solution-spin values (milli); defaults to `±1000` when omitted.
    pub allowed_spin_milli: Vec<i32>,
    /// Minimum number of solutions required to pass quality gates.
    pub min_solutions: u32,
    /// Maximum (worst) energy (milli) accepted by quality gates.
    pub max_energy_milli: i64,
    /// Minimum diversity (milli) required to pass quality gates.
    pub min_diversity_milli: u32,
}

impl TopologySpec {
    /// Build a synthetic `MiningSnapshot`: chain-only fields (last proof
    /// block hash, block number) are zeroed since drive mode has no chain.
    #[must_use]
    pub fn to_snapshot(&self) -> MiningSnapshot {
        MiningSnapshot {
            head_hash: [0u8; 32],
            last_proof_block_hash: [0u8; 32],
            topology_hash: self.topology.hash.clone(),
            nodes: self.topology.nodes.clone(),
            edges: self.topology.edge_pairs(),
            allowed_h_milli: self.allowed_h_milli.clone(),
            allowed_j_milli: self.allowed_j_milli.clone(),
            allowed_spin_milli: self.allowed_spin_milli.clone(),
            min_solutions: self.min_solutions,
            max_energy_milli: self.max_energy_milli,
            min_diversity_milli: self.min_diversity_milli,
            block_number: 0,
        }
    }
}

fn validate_edges(nodes: &[u32], edges: &[(u32, u32)]) -> Result<(), TopologySpecError> {
    let node_set: std::collections::HashSet<u32> = nodes.iter().copied().collect();
    for (edge_index, &(u, v)) in edges.iter().enumerate() {
        for endpoint in [u, v] {
            if !node_set.contains(&endpoint) {
                return Err(TopologySpecError::EdgeUnknownNode {
                    edge_index,
                    endpoint,
                });
            }
        }
    }
    Ok(())
}

/// Parse and validate a drive-mode topology-spec JSON document.
///
/// # Errors
///
/// Returns [`TopologySpecError`] when the JSON is invalid, required fields are
/// empty, or an edge references an unknown node id.
pub fn parse_topology_spec(text: &str) -> Result<TopologySpec, TopologySpecError> {
    let raw: TopologySpecJson =
        serde_json::from_str(text).map_err(|e| TopologySpecError::Json(e.to_string()))?;
    if raw.nodes.is_empty() {
        return Err(TopologySpecError::EmptyNodes);
    }
    if raw.allowed_h_milli.is_empty() {
        return Err(TopologySpecError::EmptyAllowedH);
    }
    if raw.allowed_j_milli.is_empty() {
        return Err(TopologySpecError::EmptyAllowedJ);
    }
    validate_edges(&raw.nodes, &raw.edges)?;

    let allowed_spin_milli = if raw.allowed_spin_milli.is_empty() {
        DEFAULT_SPIN_SET.to_vec()
    } else {
        raw.allowed_spin_milli
    };
    let topology = Topology::from_nodes_edges(
        raw.nodes,
        raw.edges,
        &raw.allowed_h_milli,
        &raw.allowed_j_milli,
        &allowed_spin_milli,
    );
    let gates = raw.gates.unwrap_or_default();
    Ok(TopologySpec {
        topology,
        allowed_h_milli: raw.allowed_h_milli,
        allowed_j_milli: raw.allowed_j_milli,
        allowed_spin_milli,
        min_solutions: gates.min_solutions,
        max_energy_milli: gates.max_energy_milli,
        min_diversity_milli: gates.min_diversity_milli,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"{
        "nodes": [0, 1, 2, 3],
        "edges": [[0, 1], [1, 2], [2, 3], [0, 3]],
        "allowed_h_milli": [-1000, 0, 1000],
        "allowed_j_milli": [-1000, 1000]
    }"#;

    #[test]
    fn parses_valid_spec_with_default_gates() {
        let spec = parse_topology_spec(VALID).unwrap();
        assert_eq!(spec.topology.nodes.len(), 4);
        assert_eq!(spec.topology.edge_pairs().len(), 4);
        assert_eq!(spec.min_solutions, 0);
        assert_eq!(spec.max_energy_milli, i64::MAX);
        assert_eq!(spec.min_diversity_milli, 0);
    }

    #[test]
    fn parses_explicit_gates() {
        let text = r#"{
            "nodes": [0, 1],
            "edges": [[0, 1]],
            "allowed_h_milli": [1000],
            "allowed_j_milli": [1000],
            "gates": { "min_solutions": 5, "max_energy_milli": -100, "min_diversity_milli": 200 }
        }"#;
        let spec = parse_topology_spec(text).unwrap();
        assert_eq!(spec.min_solutions, 5);
        assert_eq!(spec.max_energy_milli, -100);
        assert_eq!(spec.min_diversity_milli, 200);
    }

    #[test]
    fn snapshot_reuses_topology_hash() {
        let spec = parse_topology_spec(VALID).unwrap();
        let snap = spec.to_snapshot();
        assert_eq!(snap.topology_hash, spec.topology.hash);
        assert_eq!(snap.nodes, spec.topology.nodes);
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(matches!(
            parse_topology_spec("not json"),
            Err(TopologySpecError::Json(_))
        ));
    }

    #[test]
    fn rejects_empty_nodes() {
        let text = r#"{"nodes":[],"edges":[],"allowed_h_milli":[1],"allowed_j_milli":[1]}"#;
        assert_eq!(
            parse_topology_spec(text).unwrap_err(),
            TopologySpecError::EmptyNodes
        );
    }

    #[test]
    fn rejects_empty_allowed_sets() {
        let no_h = r#"{"nodes":[0],"edges":[],"allowed_h_milli":[],"allowed_j_milli":[1]}"#;
        assert_eq!(
            parse_topology_spec(no_h).unwrap_err(),
            TopologySpecError::EmptyAllowedH
        );
        let no_j = r#"{"nodes":[0],"edges":[],"allowed_h_milli":[1],"allowed_j_milli":[]}"#;
        assert_eq!(
            parse_topology_spec(no_j).unwrap_err(),
            TopologySpecError::EmptyAllowedJ
        );
    }

    #[test]
    fn rejects_edge_with_unknown_node_id() {
        let text = r#"{
            "nodes": [0, 1],
            "edges": [[0, 5]],
            "allowed_h_milli": [1000],
            "allowed_j_milli": [1000]
        }"#;
        assert_eq!(
            parse_topology_spec(text).unwrap_err(),
            TopologySpecError::EdgeUnknownNode {
                edge_index: 0,
                endpoint: 5,
            }
        );
    }

    #[test]
    fn accepts_sparse_native_node_ids() {
        // Native D-Wave-style sparse ids (gaps): edges reference ids, not
        // positions. The miner and consensus scorer map ids → positions.
        let text = r#"{
            "nodes": [0, 12, 2400],
            "edges": [[0, 12], [12, 2400]],
            "allowed_h_milli": [-1000, 0, 1000],
            "allowed_j_milli": [-1000, 1000]
        }"#;
        let spec = parse_topology_spec(text).unwrap();
        assert_eq!(spec.topology.nodes, vec![0, 12, 2400]);
        assert_eq!(spec.topology.edge_pairs(), vec![(0, 12), (12, 2400)]);
    }
}

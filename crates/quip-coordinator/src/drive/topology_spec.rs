//! Drive-mode topology providers: JSON specs and registered chain metadata.
//!
//! Both produce a validated `TopologySpec` and a synthetic `MiningSnapshot`.

use crate::chain::extrinsic::hex_encode;
use crate::chain::scale_types::{require_set_values, TopologyMetaScale};
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
    /// Convert registered chain metadata into a drive spec.
    ///
    /// Keeps node, edge, and value-set order. Difficulty gates use the same
    /// defaults as a JSON spec without gates. Registration height is excluded
    /// from topology identity.
    ///
    /// # Errors
    /// Returns an error for unsupported value specs, invalid graph data,
    /// or a computed topology hash that differs from `hash`.
    pub fn from_chain_meta(hash: [u8; 32], meta: TopologyMetaScale) -> Result<Self, String> {
        let raw = TopologySpecJson {
            nodes: meta.nodes,
            edges: meta.edges,
            allowed_h_milli: require_set_values(&meta.allowed_h_values)
                .map_err(|error| format!("chain topology allowed_h_milli: {error}"))?,
            allowed_j_milli: require_set_values(&meta.allowed_j_values)
                .map_err(|error| format!("chain topology allowed_j_milli: {error}"))?,
            allowed_spin_milli: require_set_values(&meta.allowed_spin_values)
                .map_err(|error| format!("chain topology allowed_spin_milli: {error}"))?,
            gates: None,
        };
        let spec = validate_spec(raw).map_err(|error| error.to_string())?;
        if spec.topology.hash.as_slice() != hash.as_slice() {
            return Err(format!(
                "topology hash mismatch: chain reported {}, computed {}",
                hex_encode(&hash),
                hex_encode(&spec.topology.hash)
            ));
        }
        Ok(spec)
    }

    /// Build a synthetic `MiningSnapshot` for generated or replayed drive work.
    /// Round hashes, block number, and runtime version use synthetic defaults.
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
            spec_version: 0,
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
        serde_json::from_str(text).map_err(|error| TopologySpecError::Json(error.to_string()))?;
    validate_spec(raw)
}

fn validate_spec(raw: TopologySpecJson) -> Result<TopologySpec, TopologySpecError> {
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
    use crate::chain::scale_types::TopologyMetaScale;
    use quantum_validation::AllowedValueSpec;

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

    fn chain_meta_fixture() -> Result<([u8; 32], TopologyMetaScale), String> {
        let spec = parse_topology_spec(crate::presets::preset_spec("aglais")?)
            .map_err(|error| error.to_string())?;
        let hash = spec
            .topology
            .hash
            .as_slice()
            .try_into()
            .map_err(|_| "topology hash is not 32 bytes".to_string())?;
        let edges = spec.topology.edge_pairs();
        Ok((
            hash,
            TopologyMetaScale {
                nodes: spec.topology.nodes,
                edges,
                allowed_h_values: AllowedValueSpec::Set(spec.allowed_h_milli),
                allowed_j_values: AllowedValueSpec::Set(spec.allowed_j_milli),
                allowed_spin_values: AllowedValueSpec::Set(spec.allowed_spin_milli),
                registered_at: 7,
            },
        ))
    }

    #[test]
    fn from_chain_meta_preserves_aglais_and_uses_neutral_gates() {
        let (hash, meta) = chain_meta_fixture().expect("valid Aglais metadata");
        let expected_nodes = meta.nodes.clone();
        let expected_edges = meta.edges.clone();
        let spec = TopologySpec::from_chain_meta(hash, meta).expect("valid chain topology");
        assert_eq!(spec.topology.hash, hash.to_vec());
        assert_eq!(spec.topology.nodes, expected_nodes);
        assert_eq!(spec.topology.edge_pairs(), expected_edges);
        assert_eq!(spec.allowed_h_milli, vec![0]);
        assert_eq!(spec.allowed_j_milli, vec![-1000, 1000]);
        assert_eq!(spec.allowed_spin_milli, vec![-1000, 1000]);
        assert_eq!(spec.min_solutions, 0);
        assert_eq!(spec.max_energy_milli, i64::MAX);
        assert_eq!(spec.min_diversity_milli, 0);
        let snapshot = spec.to_snapshot();
        assert_eq!(snapshot.head_hash, [0; 32]);
        assert_eq!(snapshot.last_proof_block_hash, [0; 32]);
        assert_eq!(snapshot.block_number, 0);
        assert_eq!(snapshot.spec_version, 0);
        assert_eq!(snapshot.topology_hash, hash.to_vec());
    }

    #[test]
    fn from_chain_meta_does_not_hash_registration_height() {
        let (hash, mut meta) = chain_meta_fixture().expect("valid Aglais metadata");
        meta.registered_at = u32::MAX;
        let spec = TopologySpec::from_chain_meta(hash, meta).expect("valid chain topology");
        assert_eq!(spec.topology.hash, hash.to_vec());
    }

    #[test]
    fn from_chain_meta_rejects_ranges_and_empty_sets() {
        let (hash, meta) = chain_meta_fixture().expect("valid Aglais metadata");
        let mut integer_h = meta.clone();
        integer_h.allowed_h_values = AllowedValueSpec::IntegerRange { min: -1, max: 1 };
        let mut continuous_j = meta.clone();
        continuous_j.allowed_j_values = AllowedValueSpec::ContinuousRange {
            min: -1000,
            max: 1000,
        };
        let mut empty_spin = meta;
        empty_spin.allowed_spin_values = AllowedValueSpec::Set(vec![]);
        for (meta, message) in [
            (integer_h, "requires AllowedValueSpec::Set"),
            (continuous_j, "requires AllowedValueSpec::Set"),
            (empty_spin, "Set is empty"),
        ] {
            let error = TopologySpec::from_chain_meta(hash, meta)
                .expect_err("expected unsupported allowed-value error");
            assert!(error.contains(message), "{error}");
        }
    }

    #[test]
    fn from_chain_meta_rejects_bad_graphs_and_hashes() {
        let (hash, meta) = chain_meta_fixture().expect("valid Aglais metadata");
        let mut empty = meta.clone();
        empty.nodes.clear();
        let error =
            TopologySpec::from_chain_meta(hash, empty).expect_err("expected empty graph error");
        assert!(error.contains("nodes must be non-empty"));
        let mut bad_edge = meta.clone();
        bad_edge.edges.push((938, u32::MAX));
        let error =
            TopologySpec::from_chain_meta(hash, bad_edge).expect_err("expected unknown node error");
        assert!(error.contains("references node id"));
        let mut incorrect_hash = hash;
        *incorrect_hash.first_mut().expect("nonempty hash") ^= 1;
        let error = TopologySpec::from_chain_meta(incorrect_hash, meta)
            .expect_err("expected hash mismatch");
        assert!(error.contains("topology hash mismatch"));
    }
}

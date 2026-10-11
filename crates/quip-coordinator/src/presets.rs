//! Built-in topology specs, embedded in the binary.
//!
//! JSON fixtures provide the base specs. Aglais derives from the embedded
//! Advantage2 spec once, without a second graph file or a filesystem read.

use std::sync::LazyLock;

/// D-Wave Advantage2 hardware graph: 4577 native nodes and 41515 edges.
const ADVANTAGE2_SYSTEM1: &str = include_str!("../fixtures/drive/advantage2-system1.spec.json");

/// A small graph for fast smoke runs.
const SMOKE: &str = include_str!("../fixtures/drive/smoke.spec.json");

static AGLAIS: LazyLock<Result<String, String>> = LazyLock::new(|| {
    let mut spec: serde_json::Value = serde_json::from_str(ADVANTAGE2_SYSTEM1)
        .map_err(|error| format!("parse embedded Advantage2 topology: {error}"))?;
    let edges = spec
        .get_mut("edges")
        .and_then(serde_json::Value::as_array_mut)
        .ok_or("embedded Advantage2 topology has no edges array")?;
    let before = edges.len();
    // Compacted positions (880, 2695) name native IDs (938, 2812).
    let removed = serde_json::json!([938, 2812]);
    edges.retain(|edge| edge != &removed);
    if edges.len() + 1 != before {
        return Err("Aglais derivation requires exactly one native edge (938, 2812)".into());
    }
    let h = spec
        .get_mut("allowed_h_milli")
        .ok_or("embedded Advantage2 topology has no allowed_h_milli")?;
    *h = serde_json::json!([0]);
    serde_json::to_string(&spec)
        .map_err(|error| format!("serialize derived Aglais topology: {error}"))
});

/// Every preset name this binary understands.
pub const PRESET_NAMES: [&str; 3] = ["advantage2-system1", "aglais", "smoke"];

/// Return the embedded or derived topology-spec JSON for `name`.
///
/// # Errors
/// Returns an error for an unknown name or an invalid embedded base spec.
pub fn preset_spec(name: &str) -> Result<&'static str, String> {
    match name {
        "advantage2-system1" => Ok(ADVANTAGE2_SYSTEM1),
        "aglais" => AGLAIS.as_ref().map(String::as_str).map_err(Clone::clone),
        "smoke" => Ok(SMOKE),
        other => Err(format!(
            "unknown topology preset {other:?}; valid presets: {}",
            PRESET_NAMES.join(", ")
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drive::parse_topology_spec;

    #[test]
    fn advantage2_preset_matches_the_committed_fixture() {
        let text = preset_spec("advantage2-system1").expect("preset resolves");
        let spec = parse_topology_spec(text).expect("preset parses");
        assert_eq!(spec.topology.nodes.len(), 4577);
        assert_eq!(spec.topology.edges.0.len(), 41515);
        assert_eq!(spec.allowed_h_milli, vec![-1000, 0, 1000]);
        assert_eq!(spec.allowed_j_milli, vec![-1000, 1000]);
    }

    #[test]
    fn smoke_preset_resolves_and_parses() {
        let text = preset_spec("smoke").expect("preset resolves");
        assert!(parse_topology_spec(text).is_ok());
    }

    #[test]
    fn unknown_preset_names_the_valid_choices() {
        let err = preset_spec("nope").expect_err("unknown preset is an error");
        assert!(err.contains("nope"), "error names the input: {err}");
        assert!(
            err.contains("advantage2-system1"),
            "error lists choices: {err}"
        );
    }

    #[test]
    fn every_listed_preset_resolves() {
        for name in PRESET_NAMES {
            assert!(
                preset_spec(name).is_ok(),
                "listed preset {name} must resolve"
            );
        }
    }

    #[test]
    fn aglais_matches_chain_identity_and_preserves_native_order() {
        let base = parse_topology_spec(
            preset_spec("advantage2-system1").expect("Advantage2 preset resolves"),
        )
        .expect("Advantage2 preset parses");
        let spec = parse_topology_spec(preset_spec("aglais").expect("Aglais preset resolves"))
            .expect("Aglais preset parses");
        let removed = (
            *base.topology.nodes.get(880).expect("position 880 exists"),
            *base.topology.nodes.get(2695).expect("position 2695 exists"),
        );
        assert_eq!(removed, (938, 2812));
        assert_eq!(spec.topology.nodes.len(), 4_577);
        assert_eq!(spec.topology.nodes, base.topology.nodes);
        let mut expected_edges = base.topology.edge_pairs();
        assert_eq!(
            expected_edges
                .iter()
                .filter(|&&edge| edge == removed)
                .count(),
            1
        );
        expected_edges.retain(|&edge| edge != removed);
        let edges = spec.topology.edge_pairs();
        assert_eq!(edges.len(), 41_514);
        assert_eq!(edges, expected_edges);
        assert!(!edges.contains(&(938, 2812)));
        assert!(!edges.contains(&(2812, 938)));
        let positions: std::collections::HashMap<_, _> = spec
            .topology
            .nodes
            .iter()
            .enumerate()
            .map(|(position, &node)| (node, position))
            .collect();
        for &(u, v) in &edges {
            let compacted = (
                *positions.get(&u).expect("edge endpoint exists"),
                *positions.get(&v).expect("edge endpoint exists"),
            );
            assert_ne!(compacted, (880, 2695));
            assert_ne!(compacted, (2695, 880));
        }
        assert_eq!(spec.allowed_h_milli, vec![0]);
        assert_eq!(spec.allowed_j_milli, vec![-1000, 1000]);
        assert_eq!(spec.allowed_spin_milli, vec![-1000, 1000]);
        assert_eq!(
            crate::chain::extrinsic::hex_encode(&spec.topology.hash),
            "0xcbec1eb4e9dcc7ae343b386902d4dbba19b694a66cebf95bc06bd9260c3270e7"
        );
        assert_eq!(spec.min_solutions, base.min_solutions);
        assert_eq!(spec.max_energy_milli, base.max_energy_milli);
        assert_eq!(spec.min_diversity_milli, base.min_diversity_milli);
    }

    #[test]
    fn aglais_resolution_keeps_the_original_fixture() {
        let before = preset_spec("advantage2-system1").expect("Advantage2 preset resolves");
        let _ = preset_spec("aglais").expect("Aglais preset resolves");
        let after = preset_spec("advantage2-system1").expect("Advantage2 preset resolves");
        assert_eq!(before, after);
        let base = parse_topology_spec(after).expect("Advantage2 preset parses");
        assert_eq!(base.topology.edge_pairs().len(), 41_515);
        assert_eq!(base.allowed_h_milli, vec![-1000, 0, 1000]);
        assert!(PRESET_NAMES.contains(&"aglais"));
        let error = preset_spec("unknown").expect_err("unknown preset is an error");
        assert!(error.contains("aglais"));
    }

    #[test]
    fn aglais_random_jobs_have_zero_fields_and_the_chain_edge_count() {
        use crate::drive::{drain_all, RandomSource};
        use crate::producer::problem::problem_milli;

        let spec = parse_topology_spec(preset_spec("aglais").expect("Aglais preset resolves"))
            .expect("Aglais preset parses");
        let mut source = RandomSource::new(&spec, [0; 32], 7, 2, 0);
        let jobs = drain_all(&mut source);
        assert_eq!(jobs.len(), 2);
        for job in jobs {
            let problem = job.ising.as_ref().expect("job has an Ising problem");
            let (h, j) = problem_milli(problem).expect("problem uses milli coefficients");
            assert_eq!(h, vec![0; 4_577]);
            assert_eq!(j.len(), 41_514);
            assert!(j.iter().all(|value| [-1000, 1000].contains(value)));
            assert_eq!(
                problem.graph,
                Some(quip_proto::v1::ising_problem::Graph::TopologyHash(
                    spec.topology.hash.clone()
                ))
            );
        }
    }
}

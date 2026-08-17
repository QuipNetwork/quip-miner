//! Built-in topology specs, embedded in the binary.
//!
//! The specs live as JSON under `fixtures/drive/` so a human can read and diff
//! them. `include_str!` copies each one into the binary at compile time. The
//! release container copies only the coordinator binary, with no source tree
//! and no fixture directory, so a preset must not resolve through the
//! filesystem.

/// `advantage2_system1`: the D-Wave Advantage2 hardware graph with defects,
/// 4577 nodes and 41515 edges. This is the graph the public testnet registered
/// as its default topology.
const ADVANTAGE2_SYSTEM1: &str = include_str!("../fixtures/drive/advantage2-system1.spec.json");

/// A small graph for fast smoke runs.
const SMOKE: &str = include_str!("../fixtures/drive/smoke.spec.json");

/// Every preset name this binary understands.
pub const PRESET_NAMES: [&str; 2] = ["advantage2-system1", "smoke"];

/// Return the embedded topology-spec JSON for `name`.
///
/// # Errors
/// Returns an error naming the input and the valid choices when `name` is not
/// a known preset.
pub fn preset_spec(name: &str) -> Result<&'static str, String> {
    match name {
        "advantage2-system1" => Ok(ADVANTAGE2_SYSTEM1),
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
}

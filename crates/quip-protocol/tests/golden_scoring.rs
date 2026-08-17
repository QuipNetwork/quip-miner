//! Golden energy, diversity, and truncation vectors vs `conformance/golden_vectors.json`.
#![expect(
    clippy::indexing_slicing,
    reason = "golden JSON keys are fixed by the fixture"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "golden spins/edges and int(energy*1000) truncation are in-range by fixture"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "milli values cast to f64 for local energy path parity"
)]

use quip_protocol::scoring::{energy_milli, set_diversity};
use serde_json::Value;
use std::fs;

#[expect(
    clippy::unwrap_used,
    reason = "integration-test helper; missing golden fixture should panic"
)]
fn golden() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/golden_vectors.json"
    );
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn energy_matches_golden() {
    for case in golden()["energy"].as_array().unwrap() {
        let spins: Vec<i8> = case["spins"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i8)
            .collect();
        let h: Vec<f64> = case["h_milli"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as f64 / 1000.0)
            .collect();
        let j: Vec<f64> = case["j_milli"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as f64 / 1000.0)
            .collect();
        let edges: Vec<(usize, usize)> = case["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| {
                (
                    e[0].as_u64().unwrap() as usize,
                    e[1].as_u64().unwrap() as usize,
                )
            })
            .collect();
        assert_eq!(
            energy_milli(&spins, &h, &j, &edges),
            case["energy_milli"].as_i64().unwrap()
        );
    }
}

#[test]
fn diversity_matches_golden() {
    for case in golden()["diversity"].as_array().unwrap() {
        let sols: Vec<Vec<i8>> = case["solutions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                s.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap() as i8)
                    .collect()
            })
            .collect();
        assert!((set_diversity(&sols) - case["diversity"].as_f64().unwrap()).abs() < 1e-9);
    }
}

// The golden `truncation` section (added by a Task-4 fix) pins that Rust's
// `(e * 1000.0) as i64` truncates toward zero identically to Python's
// int(e*1000) — the cases are chosen so truncation != rounding. This is the
// cross-language guard that a rounding impl cannot pass.
#[test]
fn truncation_matches_golden() {
    for case in golden()["truncation"].as_array().unwrap() {
        let energy = case["energy"].as_f64().unwrap();
        assert_eq!(
            (energy * 1000.0) as i64,
            case["energy_milli"].as_i64().unwrap()
        );
    }
}

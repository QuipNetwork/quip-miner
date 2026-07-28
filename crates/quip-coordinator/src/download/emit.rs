//! Bucket, rank, and emit the `hardest_models` dataset (`instances.jsonl` per
//! topology bucket + a single `manifest.json`).

use crate::chain::qblock::TopologyInputs;
use crate::download::record::InstanceRecord;
use std::collections::BTreeMap;
use std::path::Path;

/// Group records by `topology_hash`, rank each bucket hardest-first
/// (`energy_milli` ascending, ties broken by `nonce` for determinism), and
/// truncate to `cap` entries.
#[must_use]
pub fn bucket_and_rank(
    records: Vec<InstanceRecord>,
    cap: usize,
) -> BTreeMap<String, Vec<InstanceRecord>> {
    let mut buckets: BTreeMap<String, Vec<InstanceRecord>> = BTreeMap::new();
    for rec in records {
        buckets
            .entry(rec.topology_hash.clone())
            .or_default()
            .push(rec);
    }
    for bucket in buckets.values_mut() {
        bucket.sort_by(|a, b| {
            a.energy_milli
                .cmp(&b.energy_milli)
                .then(a.nonce.cmp(&b.nonce))
        });
        bucket.truncate(cap);
    }
    buckets
}

/// One `manifest.json` bucket entry: instance count, energy band, and the
/// topology spec so miners can redraw h/J offline without the chain.
#[derive(serde::Serialize)]
pub struct ManifestEntry {
    /// Human-readable alias (`"chain-default"` for the priority bucket, else
    /// an 8-hex-char prefix of the topology hash).
    pub alias: String,
    /// Instance count in this bucket.
    pub n: usize,
    /// Hardest (most negative) energy in the bucket.
    pub energy_band_min_milli: i64,
    /// Easiest (least negative) energy in the bucket.
    pub energy_band_max_milli: i64,
    /// Topology node count.
    pub nodes: usize,
    /// Topology edge count.
    pub edges: usize,
    /// Allowed linear-field values (milli).
    pub allowed_h_milli: Vec<i32>,
    /// Allowed coupling values (milli).
    pub allowed_j_milli: Vec<i32>,
    /// Allowed spin values (milli).
    pub allowed_spin_milli: Vec<i32>,
}

/// Build one bucket's manifest entry from its ranked records and topology.
fn manifest_entry(
    hash: &str,
    recs: &[InstanceRecord],
    topo: &TopologyInputs,
    default: Option<&str>,
) -> ManifestEntry {
    let alias = if default == Some(hash) {
        "chain-default".to_string()
    } else {
        hash.get(..8).unwrap_or(hash).to_string()
    };
    // recs is ranked ascending (hardest/most-negative first).
    let energy_band_min_milli = recs.first().map_or(0, |r| r.energy_milli);
    let energy_band_max_milli = recs.last().map_or(0, |r| r.energy_milli);
    ManifestEntry {
        alias,
        n: recs.len(),
        energy_band_min_milli,
        energy_band_max_milli,
        nodes: topo.nodes.len(),
        edges: topo.edges.len(),
        allowed_h_milli: topo.allowed_h_milli.clone(),
        allowed_j_milli: topo.allowed_j_milli.clone(),
        allowed_spin_milli: topo.allowed_spin_milli.clone(),
    }
}

/// Top-level `manifest.json` document: every bucket keyed by topology hash.
#[derive(serde::Serialize)]
struct Manifest<'a> {
    generated_by: &'static str,
    buckets: BTreeMap<&'a String, ManifestEntry>,
}

/// Write `<out>/<hash>/instances.jsonl` per bucket plus a single
/// `<out>/manifest.json` describing every bucket (with its embedded topology
/// spec, so miners can redraw h/J offline).
///
/// # Errors
/// Returns an I/O error if any directory/file cannot be created or written.
pub fn write_dataset(
    out_dir: &Path,
    buckets: &BTreeMap<String, Vec<InstanceRecord>>,
    topo_by_hash: &BTreeMap<String, TopologyInputs>,
    default_hash_hex: Option<&str>,
) -> std::io::Result<()> {
    let mut manifest: BTreeMap<&String, ManifestEntry> = BTreeMap::new();
    for (hash, recs) in buckets {
        let Some(topo) = topo_by_hash.get(hash) else {
            continue;
        };
        let bucket_dir = out_dir.join(hash);
        std::fs::create_dir_all(&bucket_dir)?;
        let mut body = String::new();
        for rec in recs {
            let line = serde_json::to_string(rec)?;
            body.push_str(&line);
            body.push('\n');
        }
        std::fs::write(bucket_dir.join("instances.jsonl"), body)?;
        let _ = manifest.insert(hash, manifest_entry(hash, recs, topo, default_hash_hex));
    }
    let doc = Manifest {
        generated_by: "quip-coordinator download",
        buckets: manifest,
    };
    let text = serde_json::to_string_pretty(&doc)?;
    std::fs::write(out_dir.join("manifest.json"), text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download::record::{DifficultyJson, ProvenanceJson};

    fn rec(topo: &str, energy: i64, nonce: &str) -> InstanceRecord {
        InstanceRecord {
            nonce: nonce.to_string(),
            topology_hash: topo.to_string(),
            energy_milli: energy,
            salt_hex: "00".repeat(32),
            qblock_id: 1,
            miner_hex: "00".repeat(32),
            submitted_at: 0,
            device_access_time_us: 0,
            difficulty: DifficultyJson {
                max_energy_milli: 0,
                min_solutions: 1,
                min_diversity_milli: 0,
            },
            provenance: ProvenanceJson {
                source: "chain".into(),
                qblock_id: 1,
                submitted_at: 0,
            },
        }
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "test asserts on keys just inserted by bucket_and_rank"
    )]
    fn ranks_ascending_and_never_cross_pools() {
        let recs = vec![
            rec("aa", -100, "01"),
            rec("bb", -999, "02"),
            rec("aa", -500, "03"),
            rec("aa", -300, "04"),
        ];
        let b = bucket_and_rank(recs, 10_000);
        assert_eq!(b.len(), 2); // two topologies, no merge
        let aa: Vec<i64> = b["aa"].iter().map(|r| r.energy_milli).collect();
        assert_eq!(aa, vec![-500, -300, -100]); // most negative first
        assert_eq!(b["bb"].len(), 1);
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "test asserts on keys just inserted by bucket_and_rank"
    )]
    fn cap_truncates_hardest_first() {
        let recs = vec![
            rec("aa", -100, "01"),
            rec("aa", -900, "02"),
            rec("aa", -500, "03"),
        ];
        let b = bucket_and_rank(recs, 2);
        let aa: Vec<i64> = b["aa"].iter().map(|r| r.energy_milli).collect();
        assert_eq!(aa, vec![-900, -500]); // keeps the two hardest
    }

    #[test]
    fn write_dataset_lays_out_files_and_manifest() {
        let dir = std::env::temp_dir().join(format!("hm-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut buckets = BTreeMap::new();
        let _ = buckets.insert(
            "aa".to_string(),
            vec![rec("aa", -500, "01"), rec("aa", -100, "02")],
        );
        let mut topo = BTreeMap::new();
        let _ = topo.insert(
            "aa".to_string(),
            TopologyInputs {
                nodes: vec![0, 1],
                edges: vec![(0, 1)],
                allowed_h_milli: vec![-1000, 1000],
                allowed_j_milli: vec![-1000, 1000],
                allowed_spin_milli: vec![-1000, 1000],
            },
        );
        write_dataset(&dir, &buckets, &topo, Some("aa")).unwrap();
        let jsonl = std::fs::read_to_string(dir.join("aa/instances.jsonl")).unwrap();
        assert_eq!(jsonl.lines().count(), 2);
        let manifest = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
        assert!(manifest.contains("\"chain-default\"")); // alias for default_hash
        assert!(manifest.contains("energy_band_min_milli"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Guardrail: the emitted `instances.jsonl` must be byte-compatible with
    /// the drive harness's `--source list` loader. A schema drift here (field
    /// name, `0x` prefix) would silently break the whole download → drive
    /// pipeline.
    #[test]
    fn emitted_jsonl_loads_via_list_source() {
        use crate::chain::snapshot::MiningSnapshot;
        use crate::download::record::hex_plain;
        use crate::drive::{drain_all, ListSource};

        // A bucket whose topology matches the snapshot the drive harness will pass.
        let snap = MiningSnapshot {
            last_proof_block_hash: [0u8; 32],
            topology_hash: vec![9u8; 32],
            nodes: vec![0, 1, 2, 3],
            edges: vec![(0, 1), (1, 2), (2, 3), (0, 3)],
            allowed_h_milli: vec![-1000, 0, 1000],
            allowed_j_milli: vec![-1000, 1000],
            allowed_spin_milli: vec![-1000, 1000],
            min_solutions: 1,
            max_energy_milli: i64::MAX,
            min_diversity_milli: 0,
            block_number: 0,
        };
        let hash_hex = hex_plain(&[9u8; 32]);
        // Two instances with real 64-char nonces.
        let recs = vec![
            rec(&hash_hex, -500, &"11".repeat(32)),
            rec(&hash_hex, -100, &"22".repeat(32)),
        ];
        let buckets = bucket_and_rank(recs, 10_000);
        let mut topo_map = BTreeMap::new();
        let _ = topo_map.insert(
            hash_hex.clone(),
            TopologyInputs {
                nodes: snap.nodes.clone(),
                edges: snap.edges.clone(),
                allowed_h_milli: snap.allowed_h_milli.clone(),
                allowed_j_milli: snap.allowed_j_milli.clone(),
                allowed_spin_milli: snap.allowed_spin_milli.clone(),
            },
        );
        let dir = std::env::temp_dir().join(format!("hm-ls-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_dataset(&dir, &buckets, &topo_map, None).unwrap();
        let jsonl = dir.join(&hash_hex).join("instances.jsonl");
        // The drive harness load path: nonce-ref entries re-derived against the snapshot.
        let mut src = ListSource::load(&jsonl, Some(&snap), 0).unwrap();
        let jobs = drain_all(&mut src);
        assert_eq!(jobs.len(), 2);
        assert!(jobs.first().unwrap().provenance.as_ref().unwrap().is_pow); // recognized as nonce-ref
        let _ = std::fs::remove_dir_all(&dir);
    }
}

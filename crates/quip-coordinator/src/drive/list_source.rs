//! JSONL model replay list: one entry per line, auto-detected as either a
//! nonce-ref (re-derived via `draw_ising_milli` against a topology) or an
//! explicit problem (used verbatim).

use crate::chain::MiningSnapshot;
use crate::drive::JobSource;
use crate::producer::build_ising_job_from_nonce;
use quip_proto::v1::{ising_problem, EdgeList, IsingProblem, Job, JobKind, Provenance};
use quip_protocol::wire::encode_i32_le;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ListEntryJson {
    #[serde(default)]
    nonce: Option<String>,
    #[serde(default)]
    h_milli: Option<Vec<i32>>,
    #[serde(default)]
    j_milli: Option<Vec<i32>>,
    #[serde(default)]
    edges: Option<Vec<(u32, u32)>>,
    #[serde(default)]
    num_reads: Option<u32>,
}

/// Errors reading/parsing a drive-mode JSONL model list.
#[derive(Debug, PartialEq)]
pub enum ListSourceError {
    /// Filesystem read failure (message is the I/O error).
    Io(String),
    /// `reason` names the JSON decode failure; `line` is 1-based.
    Parse {
        /// 1-based line number in the JSONL file.
        line: usize,
        /// Decode or nonce-draw failure detail.
        reason: String,
    },
    /// Neither or both of `nonce` / `h_milli` present.
    AmbiguousEntry {
        /// 1-based line number of the ambiguous entry.
        line: usize,
    },
    /// A nonce-ref entry with no `--topology` spec supplied.
    MissingTopologyForNonceRef {
        /// 1-based line number of the nonce-ref entry.
        line: usize,
    },
}

impl std::fmt::Display for ListSourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "cannot read list file: {e}"),
            Self::Parse { line, reason } => {
                write!(f, "line {line}: {reason}")
            }
            Self::AmbiguousEntry { line } => write!(
                f,
                "line {line}: entry must have exactly one of `nonce` or `h_milli`"
            ),
            Self::MissingTopologyForNonceRef { line } => write!(
                f,
                "line {line}: nonce-ref entry requires --topology <spec.json>"
            ),
        }
    }
}

impl std::error::Error for ListSourceError {}

fn decode_nonce_hex(hex: &str) -> Result<[u8; 32], String> {
    let bytes = hex_to_bytes(hex)?;
    if bytes.len() != 32 {
        return Err(format!(
            "nonce must decode to 32 bytes, got {}",
            bytes.len()
        ));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, String> {
    let hex = hex.trim();
    if !hex.len().is_multiple_of(2) {
        return Err("odd-length hex string".into());
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

fn build_explicit_job(entry: &ListEntryJson, generation: u64, deadline_ms: u64) -> Job {
    let h = entry.h_milli.clone().unwrap_or_default();
    let j = entry.j_milli.clone().unwrap_or_default();
    let edges = entry.edges.clone().unwrap_or_default();
    let (u, v): (Vec<u32>, Vec<u32>) = edges.into_iter().unzip();
    Job {
        job_id: format!("explicit-{generation}").into_bytes(),
        kind: JobKind::IsingSample as i32,
        generation,
        deadline_ms,
        ising: Some(IsingProblem {
            graph: Some(ising_problem::Graph::Edges(EdgeList { u, v })),
            h_milli_le32: encode_i32_le(&h),
            j_milli_le32: encode_i32_le(&j),
            num_reads: entry.num_reads.unwrap_or(0),
            num_sweeps: 0,
            anneal_time_us: 0,
        }),
        provenance: Some(Provenance {
            is_pow: false,
            order_id: vec![],
        }),
    }
}

fn build_entry_job(
    entry: &ListEntryJson,
    line_no: usize,
    topology_snapshot: Option<&MiningSnapshot>,
    deadline_ms: u64,
) -> Result<Job, ListSourceError> {
    let has_nonce = entry.nonce.is_some();
    let has_explicit = entry.h_milli.is_some();
    if has_nonce == has_explicit {
        return Err(ListSourceError::AmbiguousEntry { line: line_no });
    }
    if has_nonce {
        let snap = topology_snapshot
            .ok_or(ListSourceError::MissingTopologyForNonceRef { line: line_no })?;
        let nonce_hex = entry.nonce.as_deref().unwrap_or_default();
        let nonce = decode_nonce_hex(nonce_hex).map_err(|reason| ListSourceError::Parse {
            line: line_no,
            reason,
        })?;
        build_ising_job_from_nonce(snap, nonce, line_no as u64, deadline_ms).map_err(|e| {
            ListSourceError::Parse {
                line: line_no,
                reason: format!("cannot draw model from nonce-ref: {e}"),
            }
        })
    } else {
        Ok(build_explicit_job(entry, line_no as u64, deadline_ms))
    }
}

/// Replays a JSONL file of models as jobs, one line per model.
#[derive(Debug)]
pub struct ListSource {
    jobs: std::vec::IntoIter<Job>,
}

impl ListSource {
    /// Load and parse `path`. `topology_snapshot` is required when the file
    /// contains any nonce-ref entry; explicit-only lists can pass `None`.
    ///
    /// # Errors
    ///
    /// Returns [`ListSourceError`] on I/O failure or when any line fails to
    /// parse or validate (ambiguous entry, missing topology for a nonce-ref).
    pub fn load(
        path: &std::path::Path,
        topology_snapshot: Option<&MiningSnapshot>,
        deadline_ms: u64,
    ) -> Result<Self, ListSourceError> {
        let text = std::fs::read_to_string(path).map_err(|e| ListSourceError::Io(e.to_string()))?;
        Self::parse(&text, topology_snapshot, deadline_ms)
    }

    /// Parse already-loaded JSONL text (used directly by tests).
    ///
    /// # Errors
    ///
    /// Returns [`ListSourceError`] when any line fails to parse or validate
    /// (ambiguous entry, missing topology for a nonce-ref, bad JSON).
    pub fn parse(
        text: &str,
        topology_snapshot: Option<&MiningSnapshot>,
        deadline_ms: u64,
    ) -> Result<Self, ListSourceError> {
        let mut jobs = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line_no = i + 1;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let entry: ListEntryJson =
                serde_json::from_str(trimmed).map_err(|e| ListSourceError::Parse {
                    line: line_no,
                    reason: e.to_string(),
                })?;
            jobs.push(build_entry_job(
                &entry,
                line_no,
                topology_snapshot,
                deadline_ms,
            )?);
        }
        Ok(Self {
            jobs: jobs.into_iter(),
        })
    }
}

impl JobSource for ListSource {
    fn next_job(&mut self) -> Option<Job> {
        self.jobs.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drive::drain_all;
    use quip_protocol::wire::decode_i32_le;

    fn snapshot() -> MiningSnapshot {
        MiningSnapshot {
            head_hash: [0u8; 32],
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
        }
    }

    #[test]
    fn parses_explicit_entry_verbatim() {
        let text = r#"{"h_milli":[1000,-1000],"j_milli":[500],"edges":[[0,1]]}"#;
        let mut src = ListSource::parse(text, None, 9_999_999).unwrap();
        let job = src.next_job().unwrap();
        assert!(!job.provenance.unwrap().is_pow);
        let ising = job.ising.unwrap();
        assert!(matches!(ising.graph, Some(ising_problem::Graph::Edges(_))));
        assert_eq!(
            decode_i32_le(&ising.h_milli_le32).unwrap(),
            vec![1000, -1000]
        );
    }

    #[test]
    fn parses_nonce_ref_entry_against_topology() {
        let nonce_hex = "11".repeat(32);
        let text = format!(r#"{{"nonce":"{nonce_hex}"}}"#);
        let snap = snapshot();
        let mut src = ListSource::parse(&text, Some(&snap), 9_999_999).unwrap();
        let job = src.next_job().unwrap();
        assert!(job.provenance.unwrap().is_pow);
        assert_eq!(job.job_id, vec![0x11u8; 32]);
        let ising = job.ising.unwrap();
        assert!(matches!(
            ising.graph,
            Some(ising_problem::Graph::TopologyHash(_))
        ));
    }

    #[test]
    fn nonce_ref_without_topology_errors() {
        let nonce_hex = "22".repeat(32);
        let text = format!(r#"{{"nonce":"{nonce_hex}"}}"#);
        let err = ListSource::parse(&text, None, 9_999_999).unwrap_err();
        assert_eq!(err, ListSourceError::MissingTopologyForNonceRef { line: 1 });
    }

    #[test]
    fn ambiguous_entry_names_the_line() {
        let text = "{}\n{\"h_milli\":[1],\"nonce\":\"aa\"}";
        let err = ListSource::parse(text, None, 9_999_999).unwrap_err();
        assert_eq!(err, ListSourceError::AmbiguousEntry { line: 1 });
    }

    #[test]
    fn malformed_json_names_the_line() {
        let text = "{\"h_milli\":[1]}\nnot json";
        let err = ListSource::parse(text, None, 9_999_999).unwrap_err();
        assert!(matches!(err, ListSourceError::Parse { line: 2, .. }));
    }

    #[test]
    fn blank_lines_are_skipped() {
        let text = "{\"h_milli\":[1]}\n\n{\"h_milli\":[2]}\n";
        let mut src = ListSource::parse(text, None, 9_999_999).unwrap();
        assert_eq!(drain_all(&mut src).len(), 2);
    }
}

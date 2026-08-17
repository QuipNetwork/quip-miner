//! Generic external-solver miner backend.
//!
//! Each job is serialized to a JSON model, handed to a configured external
//! solver (by file path or stdin), and the solver's JSON solutions are parsed
//! back into [`SamplerResult`]s. This lets a standalone executable act as a
//! quip miner without implementing the Rust [`Sampler`] trait natively — the
//! unbuilt "wrapper" seam from the v0.3 architecture (design point 4c).
//!
//! Contract with the external solver:
//! - Input: a JSON object `{h, j, edges, num_reads, num_sweeps,
//!   sweeps_per_beta, beta_range, seed}` — via the `{model}` file-path
//!   substitution in `--solver-cmd`, or on stdin with `--solver-stdin`.
//! - Output (stdout): a JSON array of `{spins, energy_milli}`, one per read.
//!   `spins` is length `num_nodes` with values in `{-1, +1}`.
//! - Exit 0 on success; nonzero, a timeout, or malformed output rejects the job.

use quip_miner_core::{IsingGraph, SampleParams, Sampler, SamplerResult};
use quip_proto::v1::RejectReason;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// JSON model handed to the external solver. Borrows the job's fields directly.
#[derive(Serialize)]
struct ModelJson<'a> {
    h: &'a [f64],
    j: &'a [f64],
    edges: &'a [(usize, usize)],
    num_reads: usize,
    num_sweeps: usize,
    sweeps_per_beta: usize,
    beta_range: Option<(f64, f64)>,
    seed: u64,
}

impl<'a> ModelJson<'a> {
    fn new(graph: &'a IsingGraph, params: &'a SampleParams) -> Self {
        Self {
            h: &graph.h,
            j: &graph.j,
            edges: &graph.edges,
            num_reads: params.num_reads,
            num_sweeps: params.num_sweeps,
            sweeps_per_beta: params.sweeps_per_beta,
            beta_range: params.beta_range,
            seed: params.seed,
        }
    }
}

/// One solution read from the solver's stdout.
#[derive(Deserialize)]
struct SolutionJson {
    spins: Vec<i8>,
    energy_milli: i64,
}

/// A generic external-solver [`Sampler`]: renders each job to JSON and execs a
/// configured command.
pub struct ExecSampler {
    /// Pre-split command template. Tokens may contain the `{model}` placeholder.
    argv_template: Vec<String>,
    timeout: Duration,
    /// Feed the model on stdin instead of a temp file + `{model}` substitution.
    use_stdin: bool,
}

impl ExecSampler {
    /// Build from the raw `--solver-cmd` template. Fails if the template is not
    /// a valid command, or (in file mode) lacks the `{model}` placeholder.
    ///
    /// # Errors
    ///
    /// Returns an error string when `solver_cmd` is empty/unparseable, or when
    /// file mode is requested without a `{model}` placeholder.
    pub fn new(solver_cmd: &str, timeout_ms: u64, use_stdin: bool) -> Result<Self, String> {
        let argv_template = shlex::split(solver_cmd)
            .ok_or_else(|| format!("--solver-cmd is not a valid command: {solver_cmd:?}"))?;
        if argv_template.is_empty() {
            return Err("--solver-cmd is empty".to_string());
        }
        if !use_stdin && !argv_template.iter().any(|t| t.contains("{model}")) {
            return Err(
                "--solver-cmd must contain the {model} placeholder (or pass --solver-stdin)"
                    .to_string(),
            );
        }
        Ok(Self {
            argv_template,
            timeout: Duration::from_millis(timeout_ms),
            use_stdin,
        })
    }
}

impl Sampler for ExecSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, RejectReason> {
        let model = serde_json::to_vec(&ModelJson::new(graph, params))
            .map_err(|_| RejectReason::Malformed)?;

        let stdout = if self.use_stdin {
            run_solver(&self.argv_template, Some(model), self.timeout)?
        } else {
            let tmp = TempModel::write(&model).map_err(|_| RejectReason::Overloaded)?;
            let argv = render_argv(&self.argv_template, tmp.path());
            run_solver(&argv, None, self.timeout)?
        };

        parse_solutions(&stdout, graph.num_nodes())
    }
}

/// Substitute the `{model}` placeholder in every template token with `path`.
fn render_argv(template: &[String], path: &Path) -> Vec<String> {
    let path = path.to_string_lossy();
    template
        .iter()
        .map(|t| t.replace("{model}", &path))
        .collect()
}

/// Parse the solver's stdout into scored reads. Rejects (`Malformed`) on invalid
/// JSON, a spin count that doesn't match the topology, or a spin outside
/// `{-1, +1}`.
fn parse_solutions(stdout: &[u8], num_nodes: usize) -> Result<Vec<SamplerResult>, RejectReason> {
    let solutions: Vec<SolutionJson> =
        serde_json::from_slice(stdout).map_err(|_| RejectReason::Malformed)?;
    let mut out = Vec::with_capacity(solutions.len());
    for s in solutions {
        if s.spins.len() != num_nodes || s.spins.iter().any(|&v| v != -1 && v != 1) {
            return Err(RejectReason::Malformed);
        }
        out.push(SamplerResult {
            spins: s.spins,
            energy_milli: s.energy_milli,
        });
    }
    Ok(out)
}

/// Spawn the solver, optionally feed `stdin_bytes`, drain stdout concurrently,
/// and wait up to `timeout`. Timeout, spawn failure, or an I/O fault map to
/// `Overloaded`; a nonzero exit maps to `Malformed`.
fn run_solver(
    argv: &[String],
    stdin_bytes: Option<Vec<u8>>,
    timeout: Duration,
) -> Result<Vec<u8>, RejectReason> {
    let (prog, args) = argv.split_first().ok_or(RejectReason::Malformed)?;
    let mut cmd = Command::new(prog);
    let _ = cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::null());
    let _ = cmd.stdin(if stdin_bytes.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    });

    let mut child = cmd.spawn().map_err(|_| RejectReason::Overloaded)?;

    // Feed stdin from a detached thread so a large model can't deadlock against
    // the solver filling its stdout pipe before it finishes reading stdin.
    if let Some(bytes) = stdin_bytes {
        if let Some(mut sin) = child.stdin.take() {
            let _ = std::thread::spawn(move || {
                let _ = sin.write_all(&bytes);
            });
        }
    }

    // Drain stdout concurrently for the same reason.
    let mut out = child.stdout.take().ok_or(RejectReason::Overloaded)?;
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = out.read_to_end(&mut buf);
        buf
    });

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    // Don't join the reader: a killed solver can leave a grand-
                    // child holding the stdout pipe open, which would block
                    // `read_to_end` (and thus `sample`) until it exits. Detach
                    // the reader instead so the timeout stays responsive.
                    drop(reader);
                    return Err(RejectReason::Overloaded);
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(_) => return Err(RejectReason::Overloaded),
        }
    };

    let stdout = reader.join().unwrap_or_default();
    if status.success() {
        Ok(stdout)
    } else {
        Err(RejectReason::Malformed)
    }
}

/// A model file that removes itself on drop. Named by pid + a process-local
/// counter, so concurrent `sample` calls never collide.
struct TempModel(PathBuf);

impl TempModel {
    fn write(bytes: &[u8]) -> std::io::Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("quip-exec-model-{}-{}.json", std::process::id(), n));
        std::fs::write(&path, bytes)?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempModel {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_empty_and_missing_placeholder() {
        assert!(ExecSampler::new("", 1000, false).is_err());
        // File mode needs {model}.
        assert!(ExecSampler::new("solver --in", 1000, false).is_err());
        // ...but stdin mode does not.
        assert!(ExecSampler::new("solver --in", 1000, true).is_ok());
        // File mode with the placeholder is fine.
        assert!(ExecSampler::new("solver --model {model}", 1000, false).is_ok());
    }

    #[test]
    fn render_argv_substitutes_every_model_token() {
        let tmpl: Vec<String> = ["solver", "--model={model}", "{model}", "--reads=8"]
            .map(str::to_owned)
            .to_vec();
        let got = render_argv(&tmpl, Path::new("/tmp/m.json"));
        assert_eq!(
            got,
            vec!["solver", "--model=/tmp/m.json", "/tmp/m.json", "--reads=8"]
        );
    }

    #[test]
    fn parse_solutions_maps_valid_reads() {
        let json =
            br#"[{"spins":[1,-1],"energy_milli":-14000},{"spins":[-1,-1],"energy_milli":-1}]"#;
        let got = parse_solutions(json, 2).expect("valid");
        assert_eq!(got.len(), 2);
        #[expect(
            clippy::indexing_slicing,
            reason = "fixture yields two solutions; indices 0 and 1 are in bounds"
        )]
        {
            assert_eq!(got[0].spins, vec![1i8, -1]);
            assert_eq!(got[0].energy_milli, -14000);
            assert_eq!(got[1].energy_milli, -1);
        }
    }

    #[test]
    fn parse_solutions_rejects_bad_shapes() {
        // Wrong spin count for the topology.
        assert_eq!(
            parse_solutions(br#"[{"spins":[1],"energy_milli":0}]"#, 2),
            Err(RejectReason::Malformed)
        );
        // Spin value outside {-1, +1}.
        assert_eq!(
            parse_solutions(br#"[{"spins":[1,0],"energy_milli":0}]"#, 2),
            Err(RejectReason::Malformed)
        );
        // Not JSON at all.
        assert_eq!(
            parse_solutions(b"not json", 2),
            Err(RejectReason::Malformed)
        );
    }

    #[test]
    fn parse_solutions_accepts_empty_set() {
        assert_eq!(parse_solutions(b"[]", 4), Ok(vec![]));
    }

    #[test]
    fn model_json_serializes_expected_fields() {
        let graph = IsingGraph::new(vec![1.0, -1.0], vec![1.0], vec![(0, 1)]);
        let params = SampleParams {
            num_reads: 8,
            num_sweeps: 64,
            sweeps_per_beta: 2,
            beta_range: Some((0.1, 3.0)),
            seed: 7,
        };
        let v: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&ModelJson::new(&graph, &params)).unwrap())
                .unwrap();
        #[expect(
            clippy::indexing_slicing,
            reason = "serialized model always includes these fixed schema keys"
        )]
        {
            assert_eq!(v["h"], serde_json::json!([1.0, -1.0]));
            assert_eq!(v["edges"], serde_json::json!([[0, 1]]));
            assert_eq!(v["num_reads"], 8);
            assert_eq!(v["beta_range"], serde_json::json!([0.1, 3.0]));
            assert_eq!(v["seed"], 7);
        }
    }
}

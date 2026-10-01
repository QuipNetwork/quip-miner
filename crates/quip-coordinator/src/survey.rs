//! One-shot host hardware survey for the V2 node descriptor.
//!
//! The coordinator files a `system_info` block so indexers can map an
//! `AccountId` to the hardware behind it. Three invariants hold this module
//! together, and each is enforced structurally rather than by convention:
//!
//! 1. [`probe`] takes no arguments. It reads no config file and no chain
//!    state, so an operator secret from a TOML backend table has no path into
//!    the payload for a redaction pass to remove. Two caveats, both real:
//!    `QUIP_DOCKER_IMAGE` is the single environment variable read, and it is
//!    operator-settable, so it goes through the same credential scan and byte
//!    bound as every probed string; and the probe helpers (`nvidia-smi`,
//!    `sysctl`, `system_profiler`, `python`) resolve through `PATH` and
//!    inherit the process environment like any child, so whoever controls
//!    `PATH` chooses whose stdout becomes on-chain data.
//! 2. [`sanitize`] is pure and total. Every output satisfies
//!    [`fits_pallet_bounds`] for every input, so a survey can never produce a
//!    payload the pallet refuses to decode.
//! 3. Probing runs on a detached thread under a wall-clock budget. A wedged
//!    GPU driver costs the survey, never the descriptor.
//!
//! The survey is best effort. Every failure degrades to `None` or to
//! `"unknown"`, and nothing here can stop the node from filing a descriptor.

use crate::chain::scale_types::{
    CpuInfoScale, GpuInfoScale, OsInfoScale, RuntimeInfoScale, SystemInfoScale, MAX_ARCH_BYTES,
    MAX_CPU_BRAND_BYTES, MAX_DOCKER_IMAGE_BYTES, MAX_GPUS, MAX_GPU_NAME_BYTES,
    MAX_GPU_VENDOR_BYTES, MAX_OS_STRING_BYTES, MAX_RUNTIME_VERSION_BYTES,
};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

/// Stand-in for a string the pallet requires to be non-empty.
const UNKNOWN: &str = "unknown";

/// Stand-in for a field whose probed value looked like a credential.
const REDACTED: &str = "redacted";

/// Reported `python` when no interpreter answers.
///
/// The pallet rejects an empty `python`, so the field needs a value even on a
/// host with no Python at all. A version-shaped sentinel keeps the field
/// parseable by anything that reads it as a version.
const PYTHON_UNKNOWN: &str = "v0.0.0";

/// Protocol version this coordinator speaks.
///
/// Mirrors the `protocol 1` the binary reports in `--version` (`main.rs`).
const PROTOCOL_VERSION: u32 = 1;

/// Wall-clock budget for the whole survey.
///
/// A healthy Linux host finishes in about 46 ms: file reads plus one
/// `nvidia-smi` call. macOS `system_profiler` runs 1-3 s and longer under
/// load, so the budget is generous. It is paid once, at startup, before the
/// round machine runs.
pub const SURVEY_BUDGET: Duration = Duration::from_secs(5);

/// One probed GPU, before sanitizing. Strings here are unbounded.
#[derive(Clone, Debug, Default)]
pub struct RawGpu {
    /// Short vendor token, such as `NVIDIA` or `Apple`.
    pub vendor: String,
    /// Product name.
    pub name: String,
    /// Device memory in MiB.
    pub memory_mb: Option<u32>,
    /// Utilization percentage as the driver reports it.
    pub utilization_pct: Option<u8>,
}

/// The probed host, before sanitizing. Strings here are unbounded.
#[derive(Clone, Debug, Default)]
pub struct RawSurvey {
    /// OS family, such as `Linux` or `Darwin`.
    pub os_system: String,
    /// Kernel release string.
    pub os_release: String,
    /// Machine architecture as the OS spells it.
    pub os_machine: String,
    /// CPU brand string.
    pub cpu_brand: String,
    /// Instruction-set architecture.
    pub cpu_arch: String,
    /// Schedulable core count.
    pub logical_cores: Option<u32>,
    /// Distinct physical cores.
    pub physical_cores: Option<u32>,
    /// Host memory in MiB.
    pub memory_mb: Option<u32>,
    /// Attached GPUs.
    pub gpus: Vec<RawGpu>,
    /// Python interpreter version, or [`PYTHON_UNKNOWN`] when none answers.
    pub python: String,
    /// Whether the process runs inside a container.
    pub in_docker: bool,
    /// Container image, when the operator set one.
    pub docker_image: Option<String>,
}

/// Both descriptor blocks the survey produces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostSurvey {
    /// Hardware block.
    pub system: SystemInfoScale,
    /// Node-software block.
    pub runtime: RuntimeInfoScale,
}

/// Survey the host, returning a payload already inside every pallet bound.
///
/// Returns `None` when the probe thread cannot start or misses `budget`. The
/// probe runs detached, so a hung driver leaks one thread rather than blocking
/// startup or process exit.
#[must_use]
pub fn collect(budget: Duration) -> Option<HostSurvey> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<RawSurvey>(1);
    let spawned = std::thread::Builder::new()
        .name("quip-survey".to_owned())
        .spawn(move || {
            let _ = tx.send(probe());
        });
    if let Err(error) = spawned {
        tracing::warn!(%error, "could not start the host survey; filing without one");
        return None;
    }
    match rx.recv_timeout(budget) {
        Ok(raw) => Some(sanitize(raw)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            tracing::warn!(
                budget_ms = budget.as_millis(),
                "host survey missed its budget; filing without one"
            );
            None
        }
        // The probe thread unwound. Reporting this as a budget miss would send
        // an operator chasing a slow driver over a panic that took microseconds.
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            tracing::warn!("host survey thread ended without a result; filing without one");
            None
        }
    }
}

/// Clamp a raw survey into the pallet's shape. Pure, total, and free of I/O.
#[must_use]
pub fn sanitize(raw: RawSurvey) -> HostSurvey {
    let runtime = RuntimeInfoScale {
        python: bounded_nonempty(&raw.python, MAX_RUNTIME_VERSION_BYTES, "runtime.python"),
        // Compile-time, so this is structurally non-empty and the pallet's
        // `EmptyQuipVersion` is unreachable.
        quip_version: bounded_nonempty(
            env!("CARGO_PKG_VERSION"),
            MAX_RUNTIME_VERSION_BYTES,
            "runtime.quip_version",
        ),
        protocol_version: PROTOCOL_VERSION,
        in_docker: raw.in_docker,
        // Dropping an empty value is load-bearing, not tidiness: both shipped
        // images declare `ARG QUIP_DOCKER_IMAGE=""`, so an image built without
        // `--build-arg` sets the variable to the empty string. Sending
        // `Some("")` trips `EmptyDockerImage`, and one rejection latches the
        // descriptor off for the whole process.
        docker_image: raw
            .docker_image
            .as_deref()
            .map(|image| bounded(image, MAX_DOCKER_IMAGE_BYTES, "runtime.docker_image"))
            .filter(|image| !image.is_empty()),
    };
    let system = SystemInfoScale {
        os: OsInfoScale {
            system: bounded_nonempty(&raw.os_system, MAX_OS_STRING_BYTES, "os.system"),
            release: bounded(&raw.os_release, MAX_OS_STRING_BYTES, "os.release"),
            machine: bounded(&raw.os_machine, MAX_OS_STRING_BYTES, "os.machine"),
        },
        cpu: CpuInfoScale {
            logical_cores: raw.logical_cores,
            // `logical_cores` honors cgroup quota and CPU affinity; the `/sys`
            // walk behind `physical_cores` sees the whole host regardless. A
            // 4-CPU container on a 16-core host would otherwise file the
            // self-contradictory pair 4 logical / 16 physical, and an indexer
            // computing threads-per-core would read 0.25.
            physical_cores: match (raw.physical_cores, raw.logical_cores) {
                (Some(physical), Some(logical)) => Some(physical.min(logical)),
                (physical, _) => physical,
            },
            brand: bounded_nonempty(&raw.cpu_brand, MAX_CPU_BRAND_BYTES, "cpu.brand"),
            arch: bounded_nonempty(&raw.cpu_arch, MAX_ARCH_BYTES, "cpu.arch"),
        },
        memory_mb: raw.memory_mb,
        gpus: raw
            .gpus
            .into_iter()
            .take(MAX_GPUS)
            .enumerate()
            .map(|(index, gpu)| sanitize_gpu(index, &gpu))
            .collect(),
    };
    HostSurvey { system, runtime }
}

/// Whether every field satisfies the pallet's `BoundedVec` limits.
///
/// An over-length field is a SCALE decode failure at the node rather than a
/// pallet error, so it is misclassified as transient and burns the
/// descriptor's only filing attempt. This is the last gate before the wire.
#[must_use]
pub fn fits_pallet_bounds(info: &SystemInfoScale) -> bool {
    let os_ok = !info.os.system.is_empty()
        && info.os.system.len() <= MAX_OS_STRING_BYTES
        && info.os.release.len() <= MAX_OS_STRING_BYTES
        && info.os.machine.len() <= MAX_OS_STRING_BYTES;
    let cpu_ok = !info.cpu.brand.is_empty()
        && info.cpu.brand.len() <= MAX_CPU_BRAND_BYTES
        && !info.cpu.arch.is_empty()
        && info.cpu.arch.len() <= MAX_ARCH_BYTES;
    let gpus_ok = info.gpus.len() <= MAX_GPUS
        && info.gpus.iter().all(|gpu| {
            !gpu.vendor.is_empty()
                && gpu.vendor.len() <= MAX_GPU_VENDOR_BYTES
                && !gpu.name.is_empty()
                && gpu.name.len() <= MAX_GPU_NAME_BYTES
                && gpu.utilization_pct.is_none_or(|pct| pct <= 100)
        });
    os_ok && cpu_ok && gpus_ok
}

/// Whether the runtime block satisfies the pallet's limits.
///
/// Checked separately from [`fits_pallet_bounds`] so a bad runtime block drops
/// only itself, rather than taking the hardware survey off the descriptor with
/// it.
#[must_use]
pub fn runtime_fits_pallet_bounds(info: &RuntimeInfoScale) -> bool {
    !info.python.is_empty()
        && info.python.len() <= MAX_RUNTIME_VERSION_BYTES
        && !info.quip_version.is_empty()
        && info.quip_version.len() <= MAX_RUNTIME_VERSION_BYTES
        && info
            .docker_image
            .as_ref()
            .is_none_or(|image| !image.is_empty() && image.len() <= MAX_DOCKER_IMAGE_BYTES)
}

/// Whether a probed value carries the shape of a credential.
///
/// Nothing the probes read should ever match. This guards values the host
/// hands back; it is not a filter on operator config, which cannot reach here.
#[must_use]
pub fn looks_like_credential(value: &str) -> bool {
    /// Secret names, with separators removed so `api_key` and `api-key` match.
    const NEEDLES: [&str; 5] = [
        "dwaveapikey",
        "dwaveapitoken",
        "awsaccesskey",
        "awssecretkey",
        "privatekey",
    ];
    /// Names that carry a secret when they are assigned a value.
    const KEYS: [&str; 4] = ["apikey", "token", "secret", "password"];

    let lowered = value.to_lowercase();
    let squashed: String = lowered.chars().filter(|c| *c != '-' && *c != '_').collect();
    if NEEDLES.iter().any(|needle| squashed.contains(needle)) {
        return true;
    }
    if KEYS.iter().any(|key| assigns_value(&lowered, key)) {
        return true;
    }
    has_prefixed_run(value, "AKIA", 16, |c| {
        c.is_ascii_uppercase() || c.is_ascii_digit()
    }) || has_prefixed_run(&lowered, "sk-", 20, is_token_char)
        || has_prefixed_run(&lowered, "bearer ", 20, is_token_char)
        || has_prefixed_run(value, "eyJ", 20, is_token_char)
}

/// Characters that make up an opaque token body.
fn is_token_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_'
}

/// Whether `prefix` is followed by at least `run` characters passing `ok`.
fn has_prefixed_run(value: &str, prefix: &str, run: usize, ok: fn(char) -> bool) -> bool {
    value.match_indices(prefix).any(|(idx, _)| {
        value
            .get(idx + prefix.len()..)
            .is_some_and(|rest| rest.chars().take_while(|c| ok(*c)).count() >= run)
    })
}

/// Whether `haystack` assigns a non-trivial value to `key`, as in `token=...`.
fn assigns_value(haystack: &str, key: &str) -> bool {
    haystack.match_indices(key).any(|(idx, _)| {
        let Some(rest) = haystack.get(idx + key.len()..) else {
            return false;
        };
        let after_key = rest.trim_start();
        let Some(sep) = after_key.chars().next() else {
            return false;
        };
        if sep != ':' && sep != '=' {
            return false;
        }
        after_key.get(sep.len_utf8()..).is_some_and(|tail| {
            tail.split_whitespace()
                .next()
                .is_some_and(|assigned| assigned.len() >= 8)
        })
    })
}

/// Truncate to at most `max` bytes without splitting a UTF-8 code point.
///
/// A `BoundedVec<u8>` accepts a mid-code-point split without complaint, so a
/// naive byte truncation would store mojibake on chain forever.
fn clamp_utf8(value: &str, max: usize) -> &str {
    let trimmed = value.trim();
    if trimmed.len() <= max {
        return trimmed;
    }
    let mut end = max;
    while end > 0 && !trimmed.is_char_boundary(end) {
        end -= 1;
    }
    trimmed.get(..end).unwrap_or_default()
}

/// Drop control characters and collapse whitespace runs.
///
/// `/proc` and `nvidia-smi` hand back NUL- and tab-padded strings. A raw NUL
/// renders as a hole in every dashboard, and an embedded newline would let a
/// probed value forge a log line.
fn scrub(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut gap = false;
    for ch in value.chars() {
        if ch.is_control() || ch.is_whitespace() || is_invisible(ch) {
            gap = true;
            continue;
        }
        if gap && !out.is_empty() {
            out.push(' ');
        }
        gap = false;
        out.push(ch);
    }
    out
}

/// Format and bidi code points that render invisibly or reorder text.
///
/// `char::is_control` covers only the Cc category, so these survive it. That
/// matters here because the payload is permanent public storage that
/// dashboards render beside the node name: an unfiltered `U+202E` reverses
/// everything after it, and the zero-width members let two on-chain-distinct
/// nodes display identically. A probe source an operator controls, such as a
/// bind-mounted `/proc/cpuinfo`, is enough to plant one.
fn is_invisible(ch: char) -> bool {
    matches!(ch,
        '\u{00ad}'
        | '\u{200b}'..='\u{200f}'
        | '\u{2028}'..='\u{202e}'
        | '\u{2060}'..='\u{2064}'
        | '\u{2066}'..='\u{2069}'
        | '\u{feff}'
        | '\u{fff9}'..='\u{fffb}')
}

/// Scrub, redact, and clamp a field the pallet allows to be empty.
fn bounded(value: &str, max: usize, field: &'static str) -> Vec<u8> {
    let scrubbed = scrub(value);
    if looks_like_credential(&scrubbed) {
        tracing::warn!(
            field,
            "host survey field looks like a credential; redacting it"
        );
        return REDACTED.as_bytes().to_vec();
    }
    let clamped = clamp_utf8(&scrubbed, max);
    if clamped.len() < scrubbed.len() {
        tracing::warn!(
            field,
            max,
            "host survey field truncated to its pallet bound"
        );
    }
    clamped.as_bytes().to_vec()
}

/// As [`bounded`], substituting `unknown` when the result would be empty.
///
/// Use for the five fields the pallet rejects when empty: `os.system`,
/// `cpu.brand`, `cpu.arch`, `gpu.vendor`, and `gpu.name`.
fn bounded_nonempty(value: &str, max: usize, field: &'static str) -> Vec<u8> {
    let out = bounded(value, max, field);
    if out.is_empty() {
        return UNKNOWN.as_bytes().to_vec();
    }
    out
}

/// Clamp one GPU, re-indexing it by position in the capped list.
fn sanitize_gpu(index: usize, raw: &RawGpu) -> GpuInfoScale {
    GpuInfoScale {
        index: u8::try_from(index).unwrap_or(u8::MAX),
        vendor: bounded_nonempty(&raw.vendor, MAX_GPU_VENDOR_BYTES, "gpu.vendor"),
        name: bounded_nonempty(&raw.name, MAX_GPU_NAME_BYTES, "gpu.name"),
        memory_mb: raw.memory_mb,
        utilization_pct: raw.utilization_pct.map(|pct| pct.min(100)),
    }
}

/// Read the host. Takes no arguments by design; see the module docs.
fn probe() -> RawSurvey {
    let sysctl = if cfg!(target_os = "macos") {
        sysctl_values()
    } else {
        Vec::new()
    };
    let cpu_brand = cpu_brand(&sysctl).unwrap_or_default();
    let mut gpus = nvidia_gpus();
    gpus.extend(macos_gpus(&cpu_brand));
    gpus.truncate(MAX_GPUS);
    RawSurvey {
        os_system: uname_system().to_owned(),
        os_release: os_release(&sysctl).unwrap_or_default(),
        os_machine: os_machine(&sysctl),
        cpu_brand,
        cpu_arch: std::env::consts::ARCH.to_owned(),
        logical_cores: logical_cores(),
        physical_cores: physical_cores(&sysctl),
        memory_mb: memory_mb(&sysctl),
        gpus,
        python: python_version(),
        in_docker: Path::new("/.dockerenv").exists(),
        docker_image: docker_image(),
    }
}

/// Python interpreter version, as `X.Y.Z`.
///
/// The D-Wave miner runs on the `quip-solver-core` wheel, so the interpreter
/// behind it is worth reporting. The bundled virtualenv comes first because a
/// containerized node has one, and a bare `python3` on `PATH` may be a
/// different interpreter than the one the miner uses. Returns
/// [`PYTHON_UNKNOWN`] when nothing answers, which is the common case on a
/// CPU-only or Metal host.
fn python_version() -> String {
    for program in ["/opt/venv/bin/python", "python3", "python"] {
        // Python 3.4 and newer print the version to stdout; older releases
        // used stderr, which `run_probe` discards. Those are long unsupported.
        let probed = run_probe(program, &["--version"]);
        if let Some(version) = probed.as_deref().and_then(parse_python_version) {
            return version;
        }
    }
    PYTHON_UNKNOWN.to_owned()
}

/// Pull `3.12.7` out of `Python 3.12.7`.
fn parse_python_version(text: &str) -> Option<String> {
    text.split_whitespace()
        .nth(1)
        .filter(|version| version.starts_with(|c: char| c.is_ascii_digit()))
        .map(str::to_owned)
}

/// Container image, when the operator set one.
///
/// The single environment read in this module. Trimmed and emptied-to-`None`
/// because both shipped images default the variable to the empty string.
fn docker_image() -> Option<String> {
    std::env::var("QUIP_DOCKER_IMAGE")
        .ok()
        .map(|image| image.trim().to_owned())
        .filter(|image| !image.is_empty())
}

/// The OS family token, matching what v0.2 descriptors carried.
fn uname_system() -> &'static str {
    match std::env::consts::OS {
        "linux" => "Linux",
        "macos" => "Darwin",
        "windows" => "Windows",
        other => other,
    }
}

/// Look one key up in the parsed `sysctl` output.
fn sysctl_get<'a>(values: &'a [(String, String)], key: &str) -> Option<&'a str> {
    values
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.as_str())
}

/// Read the macOS values in one spawn.
///
/// Deliberately without `-n`, so the output is self-describing `key: value`
/// and a missing key cannot shift the parse.
fn sysctl_values() -> Vec<(String, String)> {
    let Some(text) = run_probe(
        "sysctl",
        &[
            "kern.osrelease",
            "hw.machine",
            "hw.physicalcpu",
            "hw.memsize",
            "machdep.cpu.brand_string",
        ],
    ) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.trim().to_owned(), value.trim().to_owned()))
        .collect()
}

/// Kernel release string.
fn os_release(sysctl: &[(String, String)]) -> Option<String> {
    if cfg!(target_os = "macos") {
        return sysctl_get(sysctl, "kern.osrelease").map(str::to_owned);
    }
    read_trimmed(Path::new("/proc/sys/kernel/osrelease"))
}

/// Machine architecture as the OS spells it.
///
/// macOS says `arm64` where Rust says `aarch64`. The OS spelling is what the
/// v0.2 descriptors carried, so existing indexers keep matching.
fn os_machine(sysctl: &[(String, String)]) -> String {
    if cfg!(target_os = "macos") {
        if let Some(machine) = sysctl_get(sysctl, "hw.machine") {
            return machine.to_owned();
        }
    }
    std::env::consts::ARCH.to_owned()
}

/// Cores the process can actually schedule on.
///
/// Honors cgroup quota and CPU affinity, so a container advertises the
/// capacity it can use rather than the host's.
fn logical_cores() -> Option<u32> {
    if let Ok(count) = std::thread::available_parallelism() {
        return u32::try_from(count.get()).ok();
    }
    u32::try_from(cpu_dirs().len()).ok().filter(|n| *n > 0)
}

/// Distinct physical cores.
fn physical_cores(sysctl: &[(String, String)]) -> Option<u32> {
    if cfg!(target_os = "macos") {
        return sysctl_get(sysctl, "hw.physicalcpu").and_then(|value| value.parse().ok());
    }
    let mut groups: BTreeSet<String> = BTreeSet::new();
    for dir in cpu_dirs() {
        let siblings = read_trimmed(&dir.join("topology/core_cpus_list"))
            .or_else(|| read_trimmed(&dir.join("topology/thread_siblings_list")));
        if let Some(group) = siblings {
            let _ = groups.insert(group);
        }
    }
    u32::try_from(groups.len()).ok().filter(|n| *n > 0)
}

/// Per-CPU directories under `/sys/devices/system/cpu`.
fn cpu_dirs() -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu") else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|entry| {
            entry.file_name().to_str().is_some_and(|name| {
                name.strip_prefix("cpu").is_some_and(|rest| {
                    !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
                })
            })
        })
        .map(|entry| entry.path())
        .collect()
}

/// CPU brand string.
fn cpu_brand(sysctl: &[(String, String)]) -> Option<String> {
    if cfg!(target_os = "macos") {
        return sysctl_get(sysctl, "machdep.cpu.brand_string").map(str::to_owned);
    }
    let cpuinfo = read_trimmed(Path::new("/proc/cpuinfo"));
    let from_proc = cpuinfo.as_deref().and_then(|text| {
        proc_key(
            text,
            &["model name", "Model Name", "Hardware", "Model", "cpu model"],
        )
    });
    from_proc.or_else(device_tree_model)
}

/// Board model from the device tree.
///
/// `aarch64` hosts have no `model name` line in `/proc/cpuinfo`, so this is
/// the only brand source on the shipped `linux/arm64` image.
fn device_tree_model() -> Option<String> {
    read_trimmed(Path::new("/sys/firmware/devicetree/base/model"))
        .map(|model| model.trim_end_matches('\0').to_owned())
}

/// Host memory in MiB.
fn memory_mb(sysctl: &[(String, String)]) -> Option<u32> {
    if cfg!(target_os = "macos") {
        let bytes: u64 = sysctl_get(sysctl, "hw.memsize")?.parse().ok()?;
        return u32::try_from(bytes / (1024 * 1024)).ok();
    }
    let text = read_trimmed(Path::new("/proc/meminfo"))?;
    let line = text.lines().find(|line| line.starts_with("MemTotal:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    u32::try_from(kb / 1024).ok()
}

/// Read a file and trim it, returning `None` when it is missing or empty.
fn read_trimmed(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_owned())
}

/// First non-empty `key : value` match, tried in key order.
fn proc_key(text: &str, keys: &[&str]) -> Option<String> {
    for key in keys {
        for line in text.lines() {
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            if name.trim() != *key {
                continue;
            }
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_owned());
            }
        }
    }
    None
}

/// Run a probe binary and capture stdout.
///
/// `output` drains both pipes, so a chatty child cannot deadlock the probe
/// thread. There is no inner timeout: [`collect`] already bounds the whole
/// survey, and a timeout per probe would multiply the leaked threads.
fn run_probe(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        tracing::debug!(program, "host survey probe exited non-zero");
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

/// Enumerate NVIDIA GPUs.
///
/// An absent `nvidia-smi` is the normal case on the CPU image, so a failure
/// here is a debug line rather than a warning.
fn nvidia_gpus() -> Vec<RawGpu> {
    let Some(text) = run_probe(
        "nvidia-smi",
        &[
            "--query-gpu=index,name,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
    ) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for line in text.lines() {
        let mut fields = line.split(',').map(str::trim);
        let _reported_index = fields.next();
        let Some(name) = fields.next().filter(|name| !name.is_empty()) else {
            continue;
        };
        out.push(RawGpu {
            // The short token, never the PCI string "NVIDIA Corporation",
            // which is 18 bytes against a 16-byte vendor bound.
            vendor: "NVIDIA".to_owned(),
            name: name.to_owned(),
            // `nounits` yields MiB and a plain 0..=100 percentage. Parsing per
            // field means an `[N/A]` on MIG or vGPU gives None, never a wrong
            // number.
            memory_mb: fields.next().and_then(|value| value.parse().ok()),
            utilization_pct: fields.next().and_then(|value| value.parse().ok()),
        });
    }
    out
}

/// Enumerate Apple GPUs.
///
/// Falls back to the CPU brand on Apple Silicon, where CPU and GPU share a
/// die. On an Intel Mac with a failed probe this reports nothing rather than
/// inventing a device.
fn macos_gpus(cpu_brand: &str) -> Vec<RawGpu> {
    if !cfg!(target_os = "macos") {
        return Vec::new();
    }
    let parsed = run_probe("system_profiler", &["-json", "SPDisplaysDataType"])
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok());
    let entries = parsed
        .as_ref()
        .and_then(|value| value.get("SPDisplaysDataType"))
        .and_then(serde_json::Value::as_array);
    if let Some(entries) = entries {
        let out: Vec<RawGpu> = entries
            .iter()
            .filter_map(|entry| {
                let name = entry
                    .get("sppci_model")
                    .or_else(|| entry.get("_name"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default();
                if name.is_empty() {
                    return None;
                }
                let vendor = entry
                    .get("spdisplays_vendor")
                    .and_then(serde_json::Value::as_str)
                    .map_or_else(
                        || "Apple".to_owned(),
                        |value| value.trim_start_matches("sppci_vendor_").to_owned(),
                    );
                Some(RawGpu {
                    vendor,
                    name: name.to_owned(),
                    memory_mb: None,
                    utilization_pct: None,
                })
            })
            .collect();
        if !out.is_empty() {
            return out;
        }
    }
    if cpu_brand.starts_with("Apple") {
        return vec![RawGpu {
            vendor: "Apple".to_owned(),
            name: format!("{cpu_brand} GPU"),
            memory_mb: None,
            utilization_pct: None,
        }];
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::{
        clamp_utf8, collect, fits_pallet_bounds, looks_like_credential, sanitize, RawGpu,
        RawSurvey, SURVEY_BUDGET,
    };
    use std::time::Duration;

    /// Sanitize and keep the hardware block. Most tests assert on that half.
    fn sys(raw: RawSurvey) -> crate::chain::scale_types::SystemInfoScale {
        sanitize(raw).system
    }

    fn gpu(vendor: &str, name: &str) -> RawGpu {
        RawGpu {
            vendor: vendor.to_owned(),
            name: name.to_owned(),
            memory_mb: None,
            utilization_pct: None,
        }
    }

    #[test]
    fn bounds_hold_under_adversarial_input() {
        let cases = vec![
            RawSurvey {
                cpu_brand: "x".repeat(4096),
                ..RawSurvey::default()
            },
            RawSurvey {
                os_release: "\u{1f600}".repeat(200),
                ..RawSurvey::default()
            },
            RawSurvey::default(),
            RawSurvey {
                gpus: (0..64)
                    .map(|i| gpu("NVIDIA", &format!("card {i}")))
                    .collect(),
                ..RawSurvey::default()
            },
            RawSurvey {
                gpus: vec![RawGpu {
                    vendor: "Advanced Micro Devices, Inc. [AMD/ATI]".to_owned(),
                    name: "y".repeat(500),
                    memory_mb: Some(16376),
                    utilization_pct: Some(250),
                }],
                ..RawSurvey::default()
            },
            RawSurvey {
                os_system: " ".repeat(80),
                os_machine: "z".repeat(300),
                cpu_arch: "x86_64-unknown-linux-gnu".to_owned(),
                ..RawSurvey::default()
            },
        ];
        for raw in cases {
            let out = sys(raw);
            assert!(fits_pallet_bounds(&out), "sanitize left a bound violated");
        }
    }

    #[test]
    fn truncation_never_splits_a_codepoint() {
        assert_eq!(clamp_utf8("日本語", 4), "日");
        assert_eq!(clamp_utf8("héllo", 2), "h");
        let value = "日本語エミュ";
        for max in 0..value.len() + 2 {
            assert!(std::str::from_utf8(clamp_utf8(value, max).as_bytes()).is_ok());
        }
    }

    #[test]
    fn guarded_fields_are_never_empty() {
        let out = sys(RawSurvey {
            gpus: vec![RawGpu::default()],
            ..RawSurvey::default()
        });
        assert_eq!(out.os.system, b"unknown");
        assert_eq!(out.cpu.brand, b"unknown");
        assert_eq!(out.cpu.arch, b"unknown");
        let first = out.gpus.first().expect("one gpu");
        assert_eq!(first.vendor, b"unknown");
        assert_eq!(first.name, b"unknown");
        // The pallet does not check these two, so empty stays empty.
        assert!(out.os.release.is_empty());
        assert!(out.os.machine.is_empty());
    }

    #[test]
    fn gpu_list_is_capped_and_reindexed() {
        let out = sys(RawSurvey {
            gpus: (0..40)
                .map(|i| gpu("NVIDIA", &format!("card {i}")))
                .collect(),
            ..RawSurvey::default()
        });
        assert_eq!(out.gpus.len(), 16);
        let indices: Vec<u8> = out.gpus.iter().map(|g| g.index).collect();
        assert_eq!(indices, (0..16).collect::<Vec<u8>>());
    }

    #[test]
    fn utilization_is_clamped_never_dropped() {
        let out = sys(RawSurvey {
            gpus: vec![RawGpu {
                utilization_pct: Some(250),
                ..gpu("NVIDIA", "RTX A4000")
            }],
            ..RawSurvey::default()
        });
        assert_eq!(out.gpus.first().and_then(|g| g.utilization_pct), Some(100));
    }

    #[test]
    fn scrub_strips_control_characters() {
        let out = sys(RawSurvey {
            cpu_brand: "AMD Ryzen\t9\u{0}  5950X\nrogue".to_owned(),
            ..RawSurvey::default()
        });
        assert_eq!(out.cpu.brand, b"AMD Ryzen 9 5950X rogue");
    }

    #[test]
    fn credential_shapes_match_and_real_probe_output_does_not() {
        let positive = [
            "DWAVE_API_TOKEN",
            "dwave-api-key",
            "AWS_SECRET_KEY",
            "AKIAIOSFODNN7EXAMPLE",
            "sk-abcdefghijklmnopqrstuvwxyz",
            "Bearer abcdefghijklmnopqrstuvwx",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0",
            "token = hunter2hunter2",
        ];
        for value in positive {
            assert!(looks_like_credential(value), "missed: {value}");
        }
        let negative = [
            "AMD Ryzen 9 5950X 16-Core Processor",
            "NVIDIA RTX A4000",
            "Apple M2 Max",
            "6.1.0-52-amd64",
            "x86_64",
            "aarch64",
            "Linux",
            "Darwin",
            "unknown",
        ];
        for value in negative {
            assert!(!looks_like_credential(value), "false positive: {value}");
        }
    }

    #[test]
    fn redaction_replaces_the_field_not_the_survey() {
        let out = sys(RawSurvey {
            cpu_brand: "DWAVE_API_KEY".to_owned(),
            os_system: "Linux".to_owned(),
            cpu_arch: "x86_64".to_owned(),
            ..RawSurvey::default()
        });
        assert_eq!(out.cpu.brand, b"redacted");
        assert_eq!(out.os.system, b"Linux");
        assert!(fits_pallet_bounds(&out));
    }

    #[test]
    fn deadline_is_real() {
        assert!(collect(Duration::from_millis(0)).is_none());
    }

    #[test]
    fn live_probe_is_within_bounds() {
        // Deliberately not `expect`. A wedged GPU driver or a slow
        // `system_profiler` legitimately misses the budget, and that is the
        // degradation this module exists to perform. Requiring `Some` would
        // turn correct production behavior into a red build. Assert the
        // invariant the module actually promises: if a survey comes back, it
        // fits every pallet bound.
        if let Some(surveyed) = collect(SURVEY_BUDGET) {
            assert!(fits_pallet_bounds(&surveyed.system));
            assert!(super::runtime_fits_pallet_bounds(&surveyed.runtime));
        }
    }

    #[test]
    fn runtime_block_is_always_fileable() {
        // The pallet rejects an empty python, an empty quip_version, and
        // Some("") for docker_image. None of the three can be produced here.
        let bare = sanitize(RawSurvey::default()).runtime;
        assert_eq!(bare.python, b"unknown");
        assert!(!bare.quip_version.is_empty());
        assert_eq!(bare.protocol_version, 1);
        assert!(bare.docker_image.is_none());
        assert!(super::runtime_fits_pallet_bounds(&bare));

        // The shipped images default QUIP_DOCKER_IMAGE to "", which must not
        // reach the wire as Some("") — that trips EmptyDockerImage and latches
        // the descriptor off for the whole process.
        let empty_image = sanitize(RawSurvey {
            docker_image: Some("   ".to_owned()),
            ..RawSurvey::default()
        })
        .runtime;
        assert!(empty_image.docker_image.is_none());
        assert!(super::runtime_fits_pallet_bounds(&empty_image));

        // A probed interpreter survives; the sentinel stands in when none does.
        let probed = sanitize(RawSurvey {
            python: "3.12.7".to_owned(),
            docker_image: Some("registry.example/quip-miner:beta".to_owned()),
            in_docker: true,
            ..RawSurvey::default()
        })
        .runtime;
        assert_eq!(probed.python, b"3.12.7");
        assert!(probed.in_docker);
        assert_eq!(
            probed.docker_image.as_deref(),
            Some(b"registry.example/quip-miner:beta".as_slice())
        );
        assert!(super::runtime_fits_pallet_bounds(&probed));
    }

    #[test]
    fn runtime_bounds_hold_under_adversarial_input() {
        let out = sanitize(RawSurvey {
            python: "9".repeat(400),
            docker_image: Some("i".repeat(4096)),
            ..RawSurvey::default()
        })
        .runtime;
        assert!(super::runtime_fits_pallet_bounds(&out));
        assert!(out.python.len() <= 48);
        assert!(out.docker_image.is_none_or(|image| image.len() <= 256));
    }

    #[test]
    fn python_version_is_parsed_from_the_banner() {
        assert_eq!(
            super::parse_python_version("Python 3.12.7\n").as_deref(),
            Some("3.12.7")
        );
        assert_eq!(super::parse_python_version("Python").as_deref(), None);
        // A wrapper that prints something else must not become a version.
        assert_eq!(
            super::parse_python_version("bash: no python").as_deref(),
            None
        );
    }

    #[test]
    fn invisible_characters_never_reach_the_payload() {
        // U+202E reverses rendering from that point on; the zero-width members
        // let two distinct nodes render identically. Both land in permanent
        // public storage, so they must not survive scrubbing.
        let out = sys(RawSurvey {
            cpu_brand: "AMD EPYC 9654\u{202e}dexednu\u{200b} 96-Core".to_owned(),
            os_release: "6.1.0\u{feff}-52\u{00ad}-amd64".to_owned(),
            ..RawSurvey::default()
        });
        let brand = std::str::from_utf8(&out.cpu.brand).expect("utf8");
        let release = std::str::from_utf8(&out.os.release).expect("utf8");
        for field in [brand, release] {
            assert!(
                !field.chars().any(super::is_invisible),
                "an invisible code point survived: {field:?}"
            );
        }
        assert!(fits_pallet_bounds(&out));
    }

    #[test]
    fn physical_cores_never_exceed_logical_cores() {
        // The container case: `available_parallelism` sees the cgroup quota,
        // the /sys walk sees the whole host.
        let out = sys(RawSurvey {
            logical_cores: Some(4),
            physical_cores: Some(16),
            ..RawSurvey::default()
        });
        assert_eq!(out.cpu.logical_cores, Some(4));
        assert_eq!(out.cpu.physical_cores, Some(4));

        // Bare metal stays untouched.
        let bare = sys(RawSurvey {
            logical_cores: Some(32),
            physical_cores: Some(16),
            ..RawSurvey::default()
        });
        assert_eq!(bare.cpu.physical_cores, Some(16));
    }

    #[test]
    fn every_bound_condition_is_enforced() {
        use crate::chain::scale_types::{
            GpuInfoScale, SystemInfoScale, MAX_ARCH_BYTES, MAX_CPU_BRAND_BYTES, MAX_GPUS,
            MAX_GPU_NAME_BYTES, MAX_GPU_VENDOR_BYTES, MAX_OS_STRING_BYTES,
        };
        // One mutation that violates a single pallet bound. Declared before any
        // statement: clippy::items_after_statements.
        type Mutate = fn(&mut SystemInfoScale);

        let base = sys(RawSurvey {
            os_system: "Linux".into(),
            os_release: "6.1.0-52-amd64".into(),
            os_machine: "x86_64".into(),
            cpu_brand: "AMD Ryzen 9 5950X".into(),
            cpu_arch: "x86_64".into(),
            gpus: vec![RawGpu {
                vendor: "NVIDIA".into(),
                name: "RTX A4000".into(),
                memory_mb: Some(16_376),
                utilization_pct: Some(50),
            }],
            ..RawSurvey::default()
        });
        assert!(fits_pallet_bounds(&base), "the fixture must start valid");

        // Each mutation violates exactly one condition. Every one must be
        // rejected, or the gate that stops a SCALE decode failure has a hole.
        let cases: Vec<(&str, Mutate)> = vec![
            ("os.system over", |i| {
                i.os.system = vec![b'x'; MAX_OS_STRING_BYTES + 1];
            }),
            ("os.system empty", |i| i.os.system.clear()),
            ("os.release over", |i| {
                i.os.release = vec![b'x'; MAX_OS_STRING_BYTES + 1];
            }),
            ("os.machine over", |i| {
                i.os.machine = vec![b'x'; MAX_OS_STRING_BYTES + 1];
            }),
            ("cpu.brand over", |i| {
                i.cpu.brand = vec![b'x'; MAX_CPU_BRAND_BYTES + 1];
            }),
            ("cpu.brand empty", |i| i.cpu.brand.clear()),
            ("cpu.arch over", |i| {
                i.cpu.arch = vec![b'x'; MAX_ARCH_BYTES + 1];
            }),
            ("cpu.arch empty", |i| i.cpu.arch.clear()),
            ("too many gpus", |i| {
                i.gpus = (0..=MAX_GPUS)
                    .map(|_| GpuInfoScale {
                        index: 0,
                        vendor: b"NVIDIA".to_vec(),
                        name: b"card".to_vec(),
                        memory_mb: None,
                        utilization_pct: None,
                    })
                    .collect();
            }),
            ("gpu.vendor over", |i| {
                if let Some(gpu) = i.gpus.first_mut() {
                    gpu.vendor = vec![b'x'; MAX_GPU_VENDOR_BYTES + 1];
                }
            }),
            ("gpu.vendor empty", |i| {
                if let Some(gpu) = i.gpus.first_mut() {
                    gpu.vendor.clear();
                }
            }),
            ("gpu.name over", |i| {
                if let Some(gpu) = i.gpus.first_mut() {
                    gpu.name = vec![b'x'; MAX_GPU_NAME_BYTES + 1];
                }
            }),
            ("gpu.name empty", |i| {
                if let Some(gpu) = i.gpus.first_mut() {
                    gpu.name.clear();
                }
            }),
            ("utilization over 100", |i| {
                if let Some(gpu) = i.gpus.first_mut() {
                    gpu.utilization_pct = Some(101);
                }
            }),
        ];
        for (label, mutate) in cases {
            let mut info = base.clone();
            mutate(&mut info);
            assert!(
                !fits_pallet_bounds(&info),
                "fits_pallet_bounds accepted an invalid payload: {label}"
            );
        }
    }
}

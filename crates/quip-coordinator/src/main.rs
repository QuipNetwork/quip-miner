//! quip-coordinator binary: CLI, runtime wiring, graceful shutdown.

use clap::{Parser, Subcommand, ValueEnum};
use quip_coordinator::chain::extrinsic::{
    hex_encode, load_hybrid_pair, miner_identity_bytes, signer_account_bytes,
};
use quip_coordinator::chain::scale_types::DifficultyConfig;
use quip_coordinator::chain::{
    seed_chain, RealChainClient, SeedParams, SeedReport, SeedTopology, DEFAULT_SEED_DIFFICULTY,
};
use quip_coordinator::config::{parse_config, CoordinatorConfig, LaunchEntry};
use quip_coordinator::drive::{
    aggregate, drain_all, parse_topology_spec, print_table, run_drive, write_jsonl,
    DriveManyParams, ListSource, RandomSource,
};
use quip_coordinator::logging::LogLevel;
use quip_coordinator::presets::preset_spec;
use quip_coordinator::runtime::{run_runtime, RuntimeParams};
use quip_coordinator::session::{gen_session_token, CoordinatorState};
use quip_coordinator::supervisor::BackoffPolicy;
use quip_coordinator::topology::Topology;
use quip_proto::v1::{Configure, Job};
use quip_protocol::session::ExitCode;
use sp_core::crypto::Ss58Codec;
use std::path::PathBuf;
use std::process::ExitCode as StdExitCode;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Parser, Debug)]
#[command(
    name = "quip-coordinator",
    version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"),
    about = "QuIP v0.3 coordinator: chain access, routing, miner supervision"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Path to coordinator config.toml (ignored when a subcommand is given)
    #[arg(long)]
    config: Option<PathBuf>,

    /// Log verbosity. Defaults to `info`. Takes precedence over `RUST_LOG`;
    /// when omitted, `RUST_LOG` is honored if set. Logs go to stderr.
    #[arg(long, value_enum, global = true)]
    log_level: Option<LogLevel>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Drive a spawned miner with synthetic work; no chain, no submit.
    Drive(DriveArgs),
    /// Create a signer keystore at --out. Refuses to overwrite.
    Keygen(KeygenArgs),
    /// Seed a fresh chain: register the default topology and set its difficulty.
    SeedChain(SeedChainArgs),
}

#[derive(clap::Args, Debug)]
struct KeygenArgs {
    /// Destination path for the keystore JSON.
    #[arg(long)]
    out: PathBuf,
}

#[derive(clap::Args, Debug)]
struct SeedChainArgs {
    /// Validator RPC endpoint.
    #[arg(long, default_value = "ws://quip-validator:9944")]
    validator: String,
    /// Sudo signer: a dev URI (//Alice), a BIP39 mnemonic, a 32-byte hex master
    /// seed, or a keystore path. Mutually exclusive with --mnemonic-file.
    #[arg(long, conflicts_with = "mnemonic_file")]
    sudo_key: Option<String>,
    /// Path to a file holding a BIP39 mnemonic phrase. Mount it read-only.
    #[arg(long)]
    mnemonic_file: Option<PathBuf>,
    /// Built-in topology by name. Mutually exclusive with --topology.
    #[arg(
        long,
        default_value = "advantage2-system1",
        conflicts_with = "topology"
    )]
    topology_preset: String,
    /// Topology spec JSON path.
    #[arg(long)]
    topology: Option<PathBuf>,
    /// Minimum valid solutions a proof must carry.
    #[arg(long, default_value_t = DEFAULT_SEED_DIFFICULTY.min_solutions)]
    min_solutions: u32,
    /// Energy ceiling in milli units; solutions must be strictly below it.
    #[arg(long, default_value_t = DEFAULT_SEED_DIFFICULTY.max_energy_milli)]
    max_energy_milli: i64,
    /// Minimum solution-set diversity in milli units.
    #[arg(long, default_value_t = DEFAULT_SEED_DIFFICULTY.min_diversity_milli)]
    min_diversity_milli: u32,
}

#[derive(ValueEnum, Clone, Debug)]
enum DriveSourceKind {
    Random,
    List,
}

#[derive(clap::Args, Debug)]
struct DriveArgs {
    /// Miner binary to spawn.
    #[arg(long)]
    miner: PathBuf,
    /// Job source: golden-draw random problems, or a JSONL replay list.
    #[arg(long, value_enum)]
    source: DriveSourceKind,
    /// Topology spec JSON. For `--source random`, optional — defaults to the
    /// `advantage2-system1` preset if neither this nor `--topology-preset` is
    /// given. For `--source list`, needed only if the list has a nonce-ref entry.
    #[arg(long)]
    topology: Option<PathBuf>,
    /// Built-in topology by name (`advantage2-system1`, `smoke`), embedded in
    /// the binary. Mutually exclusive with `--topology`.
    #[arg(long)]
    topology_preset: Option<String>,
    /// Number of problems to draw (`--source random`).
    #[arg(long, default_value_t = 10)]
    count: u32,
    /// Draw seed: same seed + topology draws the same jobs (`--source random`).
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// JSONL model list (`--source list`).
    #[arg(long)]
    list: Option<PathBuf>,
    /// Difficulty target energy (the adapt target, in energy units). Overrides
    /// the spec's gate; the miner adapts its reads/sweeps from this.
    #[arg(long)]
    target_energy: Option<f64>,
    /// Minimum unique solutions gate. Overrides the spec.
    #[arg(long)]
    min_solutions: Option<u32>,
    /// Pin `num_reads` via the `SetTarget` control-plane override (bypasses adapt).
    /// Useful for the dwave/QPU path, which does not adapt yet.
    #[arg(long)]
    num_reads: Option<u32>,
    /// Pin `num_sweeps` via the `SetTarget` override (bypasses adapt). Pairs with
    /// `--num-reads` for controlled, matched-condition throughput/parity runs.
    #[arg(long)]
    num_sweeps: Option<u32>,
    /// Per-job deadline, milliseconds from now. 0 = no deadline (the default);
    /// a deadline is only meaningful for real mempool/chain jobs.
    #[arg(long, default_value_t = 0)]
    deadline_ms: u64,
    /// Optional path to write a per-job + aggregate JSONL report.
    #[arg(long)]
    report: Option<PathBuf>,
    /// GPU utilization ceiling 1–100 forwarded to the spawned miner's
    /// `--utilization` (cuda/metal only; other backends reject it).
    #[arg(long)]
    utilization: Option<u32>,
    /// Forward `--yielding` to the spawned miner (cuda/metal only).
    #[arg(long, default_value_t = false)]
    yielding: bool,
    /// CUDA device ordinal for the spawned miner. A config run takes this from
    /// the `[cuda.N]` section key, which drive has no equivalent of, so a
    /// multi-GPU host needs it stated here or the miner lands on GPU 0.
    /// Omit for backends whose CLI has no `--device`.
    #[arg(long)]
    device: Option<usize>,
}

#[expect(
    clippy::print_stderr,
    reason = "the log subscriber is what failed; stderr is the only channel left"
)]
fn main() -> StdExitCode {
    let cli = Cli::parse();
    // Before anything else: without this, every `tracing` call in the process
    // is a silent no-op and `RUST_LOG` has no effect.
    if let Err(e) = quip_coordinator::logging::init(cli.log_level) {
        eprintln!("error: {e}");
        return StdExitCode::from(ExitCode::ConfigInvalid as u8);
    }
    match cli.command {
        Some(Command::Drive(args)) => run_drive_cli(args, cli.log_level.unwrap_or(LogLevel::Info)),
        Some(Command::Keygen(args)) => run_keygen_cli(&args),
        Some(Command::SeedChain(args)) => run_seed_chain_cli(args),
        // When --log-level is omitted the coordinator default is Info (unless
        // RUST_LOG overrides the coordinator filter alone). Forward that same
        // default to miner children so their verbosity matches the CLI default.
        None => run_config_path(cli.config, cli.log_level.unwrap_or(LogLevel::Info)),
    }
}

#[expect(
    clippy::print_stderr,
    clippy::print_stdout,
    reason = "CLI binary reports keystore path to stdout and errors to stderr"
)]
fn run_keygen_cli(args: &KeygenArgs) -> StdExitCode {
    match quip_coordinator::keygen::write_keystore(&args.out) {
        Ok(()) => {
            println!("wrote keystore to {}", args.out.display());
            StdExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            StdExitCode::from(ExitCode::ConfigInvalid as u8)
        }
    }
}

#[expect(
    clippy::print_stderr,
    clippy::print_stdout,
    reason = "CLI binary reports seed-chain errors to stderr"
)]
fn run_seed_chain_cli(args: SeedChainArgs) -> StdExitCode {
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("seed-chain: tokio runtime: {e}");
            return StdExitCode::FAILURE;
        }
    };
    match rt.block_on(seed_chain_inner(args)) {
        Ok(report) => {
            println!(
                "seeded topology {} ({} nodes, {} edges)",
                hex_encode(&report.topology_hash),
                report.nodes,
                report.edges
            );
            println!("  register_topology included in {}", report.register_block);
            println!(
                "  set_difficulty    included in {}",
                report.difficulty_block
            );
            StdExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("seed-chain: {e}");
            StdExitCode::FAILURE
        }
    }
}

async fn seed_chain_inner(args: SeedChainArgs) -> Result<SeedReport, String> {
    let sudo_key = match (&args.sudo_key, &args.mnemonic_file) {
        (Some(k), None) => k.clone(),
        (None, Some(p)) => {
            let phrase = std::fs::read_to_string(p)
                .map_err(|e| format!("read mnemonic file {}: {e}", p.display()))?;
            let phrase = phrase.trim().to_string();
            if phrase.is_empty() {
                return Err(format!("mnemonic file is empty: {}", p.display()));
            }
            phrase
        }
        _ => return Err("give exactly one of --sudo-key or --mnemonic-file".into()),
    };

    let text = match &args.topology {
        Some(p) => {
            std::fs::read_to_string(p).map_err(|e| format!("read topology {}: {e}", p.display()))?
        }
        None => preset_spec(&args.topology_preset)?.to_string(),
    };
    let spec = parse_topology_spec(&text).map_err(|e| format!("{e:?}"))?;

    seed_chain(SeedParams {
        validator: args.validator,
        sudo_key,
        topology: SeedTopology::from_spec(&spec),
        difficulty: DifficultyConfig {
            min_solutions: args.min_solutions,
            max_energy_milli: args.max_energy_milli,
            min_diversity_milli: args.min_diversity_milli,
        },
    })
    .await
    .map_err(|e| e.to_string())
}

fn load_coordinator_config(
    config: Option<PathBuf>,
) -> Result<(PathBuf, CoordinatorConfig), StdExitCode> {
    let Some(config_path) = config else {
        tracing::error!("--config <path> is required");
        return Err(StdExitCode::from(ExitCode::ConfigInvalid as u8));
    };

    let text = match std::fs::read_to_string(&config_path) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!(path = %config_path.display(), error = %e, "cannot read config");
            return Err(StdExitCode::from(ExitCode::ConfigInvalid as u8));
        }
    };

    match parse_config(&text) {
        Ok(c) => Ok((config_path, c)),
        Err(e) => {
            tracing::error!(path = %config_path.display(), error = %e, "invalid config");
            Err(StdExitCode::from(ExitCode::ConfigInvalid as u8))
        }
    }
}

fn run_config_path(config: Option<PathBuf>, log_level: LogLevel) -> StdExitCode {
    // --help is handled by clap (exit 0). Missing/invalid config → exit 64.
    let (config_path, cfg) = match load_coordinator_config(config) {
        Ok(v) => v,
        Err(code) => return code,
    };

    // Identify the process before doing anything that can warn or fail, so the
    // first line of any captured log says what this is and what it is talking to.
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        protocol = 1,
        config = %config_path.display(),
        "quip-coordinator starting"
    );
    tracing::info!(validators = ?cfg.validators, "chain validators configured");

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            tracing::error!(error = %e, "cannot start tokio runtime");
            return StdExitCode::from(ExitCode::InternalFatal as u8);
        }
    };

    let chain = Arc::new(RealChainClient::new(
        cfg.validators.clone(),
        cfg.signer_key.clone(),
        quip_coordinator::config::participate_kind(&cfg.launch),
    ));

    if let Some(code) = run_preflight_gate(&rt, chain.as_ref()) {
        return code;
    }

    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    // Two distinct 32-byte values come out of one signer key, and mixing them
    // up funds an address that never signs anything. A live mining coordinator
    // needs its signer key; without a usable one, warn and fall back to zeros —
    // it still serves and feeds, but its proofs won't verify on-chain.
    let MinerKeys {
        identity: miner_identity,
        account: miner_account,
        ss58: miner_ss58,
    } = miner_keys_from_key(&cfg.signer_key);

    let identity = identity_from_keys(miner_ss58, miner_account, cfg.node_id.as_deref());

    let funding = quip_coordinator::funding::FundingParams {
        faucet_url: cfg.faucet_url.clone(),
        min_balance: cfg.min_balance_plancks,
        top_up: cfg.faucet_top_up_plancks,
        timeout: std::time::Duration::from_secs(cfg.funding_timeout_s),
    };
    let mut descriptor = quip_coordinator::config::DescriptorParams::from_config(&cfg);
    // Survey the host once, here. `build_descriptor_payload` is a synchronous
    // fn reached from async with no `spawn_blocking`, so probing there would
    // stall a tokio worker. Once per process also matches the semantics of the
    // `descriptor_filed` latch.
    if let Some(surveyed) =
        quip_coordinator::survey::collect(quip_coordinator::survey::SURVEY_BUDGET)
    {
        descriptor.system_info = Some(surveyed.system);
        descriptor.runtime = Some(surveyed.runtime);
    }
    let descriptor_filed = Arc::new(AtomicBool::new(false));
    let miner_registered = Arc::new(AtomicBool::new(false));
    if let Some(code) = run_startup_prepare(
        &rt,
        chain.as_ref(),
        miner_account,
        &funding,
        &descriptor,
        &quip_coordinator::readiness::ProcessLatches {
            descriptor_filed: descriptor_filed.as_ref(),
            miner_registered: miner_registered.as_ref(),
        },
    ) {
        return code;
    }

    let params = RuntimeParams {
        sock_path: format!("/tmp/quip-coordinator-{}.sock", std::process::id()),
        max_submit_attempts: cfg.max_submit_attempts,
        grace_ms: 2000,
        backoff: BackoffPolicy::default(),
        miner_identity,
        miner_account,
        // Generous floor: keep every miner well-fed from the first poll, before
        // its drain-rate EMA ramps. The adaptive window grows above this for
        // fast, many-core backends; slow ones simply sit at the floor.
        buffer_depth: 256,
        poll_interval_ms: 1000,
        dashboard: cfg
            .dashboard
            .as_ref()
            .map(|d| (d.listen.clone(), PathBuf::from(&d.data_dir))),
        log_level,
        funding,
        descriptor,
        descriptor_filed,
        miner_registered,
        identity,
    };
    tracing::info!(
        miners = cfg.launch.len(),
        ids = ?cfg.launch.iter().map(|e| e.miner_id.as_str()).collect::<Vec<_>>(),
        socket = %params.sock_path,
        "serving miner session socket"
    );

    let result = rt.block_on(async move {
        run_runtime(cfg.launch, chain, state, params, shutdown_signal()).await
    });
    match result {
        Ok(()) => {
            tracing::info!("quip-coordinator stopped cleanly");
            StdExitCode::from(ExitCode::Clean as u8)
        }
        Err(e) => {
            tracing::error!(error = %e, "runtime failed");
            StdExitCode::from(ExitCode::InternalFatal as u8)
        }
    }
}

/// Compatibility gate run before anything else touches the chain.
///
/// The coordinator drives the chain through mirrored SCALE types and a pinned
/// runtime API; against a validator that predates them every read fails one
/// poll at a time, deep in the feeder. Decide it here instead.
///
/// A *skewed* validator is fatal (exit 64 — operator error, do not respawn).
/// An *unreachable* one is not: the node manager starts the coordinator and its
/// validator together, so exiting because the node is still booting would just
/// crash-loop. The feeder retries, and reachability transitions are logged.
fn run_preflight_gate(
    rt: &tokio::runtime::Runtime,
    chain: &RealChainClient,
) -> Option<StdExitCode> {
    match rt.block_on(chain.preflight()) {
        Ok(_) => None,
        Err(quip_coordinator::chain::preflight::PreflightError::Unreachable(e)) => {
            tracing::warn!(
                error = %e,
                "cannot reach a validator to verify compatibility; starting anyway and \
                 will retry (set --log-level debug to watch the retries)"
            );
            None
        }
        Err(e) => {
            tracing::error!(error = %e, "incompatible validator; refusing to start");
            Some(StdExitCode::from(ExitCode::ConfigInvalid as u8))
        }
    }
}

/// The two unrelated 32-byte values one signer key produces.
struct MinerKeys {
    /// `PoW` nonce input. Holds no balance and keys no storage map.
    identity: [u8; 32],
    /// Signing `AccountId32`: pays fees, holds the balance, keys the maps.
    account: [u8; 32],
    /// SS58 form of `account`, or `None` when no usable signer key loaded.
    ss58: Option<String>,
}

fn miner_keys_from_key(signer_key: &str) -> MinerKeys {
    match load_hybrid_pair(signer_key) {
        Ok(pair) => {
            let account = signer_account_bytes(&pair);
            MinerKeys {
                identity: miner_identity_bytes(&pair),
                account,
                ss58: Some(sp_core::crypto::AccountId32::from(account).to_ss58check()),
            }
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "no usable signer key; PoW proofs will not verify on-chain"
            );
            MinerKeys {
                identity: [0u8; 32],
                account: [0u8; 32],
                ss58: None,
            }
        }
    }
}

/// Build the dashboard identity from the loaded signer keys.
///
/// An unkeyed coordinator advertises no identity at all. The dashboard reads
/// an empty `ss58_address` as an absent identity and skips its whole identity
/// block, which is the right answer for a process that cannot sign.
fn identity_from_keys(
    ss58: Option<String>,
    account: [u8; 32],
    node_id: Option<&str>,
) -> quip_coordinator::metrics::Identity {
    ss58.map_or_else(
        quip_coordinator::metrics::Identity::default,
        |ss58_address| quip_coordinator::metrics::Identity {
            ss58_address,
            account_id_hex: hex_encode(&account),
            node_id: node_id.map_or_else(
                || quip_coordinator::readiness::node_id_from_account(&account),
                str::to_string,
            ),
        },
    )
}

/// Same readiness walk the feeder re-runs on every later round.
///
/// Funding failure is fatal at startup (exit 64). A missing snapshot is not:
/// the feeder retries once miners are connected.
fn run_startup_prepare(
    rt: &tokio::runtime::Runtime,
    chain: &RealChainClient,
    miner_account: [u8; 32],
    funding: &quip_coordinator::funding::FundingParams,
    descriptor: &quip_coordinator::config::DescriptorParams,
    latches: &quip_coordinator::readiness::ProcessLatches<'_>,
) -> Option<StdExitCode> {
    let faucet = quip_coordinator::readiness::build_faucet(funding.faucet_url.as_deref());
    match rt.block_on(quip_coordinator::readiness::prepare_round(
        chain,
        faucet.as_ref(),
        miner_account,
        funding,
        tokio::time::sleep,
        descriptor,
        latches,
    )) {
        Ok(_) => None,
        Err(quip_coordinator::readiness::ReadinessError::Funding(e)) => {
            tracing::error!(error = %e, "miner account is not funded; refusing to start");
            Some(StdExitCode::from(ExitCode::ConfigInvalid as u8))
        }
        Err(quip_coordinator::readiness::ReadinessError::Snapshot(e)) => {
            tracing::warn!(
                error = %e,
                "no mining snapshot at startup; feeder will retry"
            );
            None
        }
    }
}

/// Resolve when the process receives SIGINT or SIGTERM (SIGINT-only off-unix).
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

fn now_unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    #[expect(
        clippy::cast_possible_truncation,
        reason = "unix millis fit u64 for practical coordinator lifetimes"
    )]
    {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

type BuiltJobs = (
    Vec<Job>,
    Option<Topology>,
    Option<quip_proto::v1::SetTarget>,
);

/// Convert a CLI energy value to milli-units.
#[expect(
    clippy::cast_possible_truncation,
    reason = "CLI target energy is a bounded value; milli fits i64"
)]
fn energy_to_milli(e: f64) -> i64 {
    (e * 1000.0) as i64
}

/// Build the `SetTarget` a drive run advertises, from the spec's gates, with
/// optional CLI overrides (`--target-energy`, `--min-solutions`).
fn set_target_from_spec(
    spec: &quip_coordinator::drive::TopologySpec,
    args: &DriveArgs,
) -> quip_proto::v1::SetTarget {
    let max_energy_milli = args
        .target_energy
        .map_or(spec.max_energy_milli, energy_to_milli);
    quip_proto::v1::SetTarget {
        max_energy_milli,
        min_solutions: args.min_solutions.unwrap_or(spec.min_solutions),
        min_diversity_milli: spec.min_diversity_milli,
        num_reads: args.num_reads.unwrap_or(0),
        num_sweeps: args.num_sweeps.unwrap_or(0),
        anneal_time_us: 0,
    }
}

/// Load the requested job source, returning its jobs plus the wire `Topology`
/// message (if any). Any synthetic `MiningSnapshot` needed to re-derive
/// nonce-ref list entries is consumed inside `ListSource::load` and is not
/// returned: drive mode has no chain to wire it into.
fn build_jobs(args: &DriveArgs, deadline_ms: u64) -> Result<BuiltJobs, String> {
    let topo_text = resolve_topology_text(args)?;
    match args.source {
        DriveSourceKind::Random => {
            let text =
                topo_text.ok_or("--source random requires --topology or --topology-preset")?;
            let spec = parse_topology_spec(&text).map_err(|e| e.to_string())?;
            let miner_account = [0u8; 32];
            let mut src =
                RandomSource::new(&spec, miner_account, args.seed, args.count, deadline_ms);
            let jobs = drain_all(&mut src);
            let target = set_target_from_spec(&spec, args);
            Ok((jobs, Some(spec.topology), Some(target)))
        }
        DriveSourceKind::List => {
            let list_path = args
                .list
                .as_ref()
                .ok_or("--source list requires --list <models.jsonl>")?;
            let (topology, snapshot, target) = match &topo_text {
                Some(text) => {
                    let spec = parse_topology_spec(text).map_err(|e| e.to_string())?;
                    let target = set_target_from_spec(&spec, args);
                    (
                        Some(spec.topology.clone()),
                        Some(spec.to_snapshot()),
                        Some(target),
                    )
                }
                None => (None, None, None),
            };
            let mut src = ListSource::load(list_path, snapshot.as_ref(), deadline_ms)
                .map_err(|e| e.to_string())?;
            let jobs = drain_all(&mut src);
            Ok((jobs, topology, target))
        }
    }
}

/// Default preset used by `--source random` when no topology is specified.
const DEFAULT_PRESET: &str = "advantage2-system1";

/// Resolve the topology spec text from `--topology` / `--topology-preset`.
/// `--source random` falls back to [`DEFAULT_PRESET`]; `--source list` returns
/// `None` (a nonce-ref list supplies its own topology or needs none).
fn resolve_topology_text(args: &DriveArgs) -> Result<Option<String>, String> {
    if args.topology.is_some() && args.topology_preset.is_some() {
        return Err("--topology and --topology-preset are mutually exclusive".into());
    }
    if let Some(p) = &args.topology {
        return std::fs::read_to_string(p)
            .map(Some)
            .map_err(|e| format!("read topology {}: {e}", p.display()));
    }
    if let Some(name) = &args.topology_preset {
        return preset_spec(name).map(|s| Some(s.to_string()));
    }
    match args.source {
        DriveSourceKind::Random => preset_spec(DEFAULT_PRESET).map(|s| Some(s.to_string())),
        DriveSourceKind::List => Ok(None),
    }
}

#[expect(
    clippy::print_stderr,
    reason = "CLI binary reports drive runtime errors to stderr"
)]
fn run_drive_cli(args: DriveArgs, log_level: LogLevel) -> StdExitCode {
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: runtime: {e}");
            return StdExitCode::from(ExitCode::InternalFatal as u8);
        }
    };
    rt.block_on(drive_main(args, log_level))
}

/// The launch entry `drive` spawns its miner from.
///
/// Split out of `drive_main` so the one field an operator can get wrong is
/// testable without standing up a session: `--device` is the only way a drive
/// run reaches a GPU other than 0, because drive has no `[cuda.N]` section to
/// take an ordinal from.
fn drive_entry(args: &DriveArgs) -> LaunchEntry {
    LaunchEntry {
        miner_id: "drive-0".into(),
        binary: args.miner.to_string_lossy().into_owned(),
        backend: "cpu".into(),
        device: args.device,
        configure: Configure {
            queue_depth: 3,
            idle_timeout_s: 30,
            heartbeat_s: 15,
            reconnect_window_s: 60,
            backend_toml: String::new(),
        },
    }
}

#[expect(
    clippy::print_stderr,
    reason = "CLI binary reports drive run failures to stderr"
)]
async fn drive_main(args: DriveArgs, log_level: LogLevel) -> StdExitCode {
    // 0 => no deadline (the sentinel the miner honors); otherwise an absolute
    // wall-clock deadline `now + args.deadline_ms`.
    let deadline_ms = if args.deadline_ms == 0 {
        0
    } else {
        now_unix_ms() + args.deadline_ms
    };
    let (jobs, topology, target) = match build_jobs(&args, deadline_ms) {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("error: {msg}");
            return StdExitCode::from(ExitCode::ConfigInvalid as u8);
        }
    };

    let entry = drive_entry(&args);
    let sock = format!("/tmp/quip-coordinator-drive-{}.sock", std::process::id());
    let token = gen_session_token();
    let report = run_drive(DriveManyParams {
        miner_bin: &entry.binary,
        sock_path: &sock,
        miner_id: &entry.miner_id,
        token: &token,
        entry: &entry,
        topology,
        target,
        jobs,
        utilization: args.utilization,
        yielding: args.yielding,
        log_level,
    })
    .await;

    if !report.handshake_ok {
        eprintln!(
            "error: miner handshake failed (exit code {})",
            report.miner_exit_code
        );
        return StdExitCode::from(ExitCode::InternalFatal as u8);
    }

    let agg = aggregate(&report.rows, report.run_wall_ms);
    print_table(&report.rows, &agg);
    if let Some(path) = &args.report {
        if let Err(e) = write_jsonl(path, &report.rows, &agg) {
            eprintln!("error: cannot write report {}: {e}", path.display());
            return StdExitCode::from(ExitCode::InternalFatal as u8);
        }
    }
    // A truncated run (miner crash / dropped Result / timeout) leaves fewer rows
    // than jobs. That is a run failure, not per-job data, so it must not exit 0.
    if report.rows.len() < report.total {
        eprintln!(
            "error: incomplete run — {} of {} jobs produced no result",
            report.total - report.rows.len(),
            report.total
        );
        return StdExitCode::from(ExitCode::InternalFatal as u8);
    }
    StdExitCode::from(ExitCode::Clean as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The drive counterpart of the `[cuda.N]` bug: without `--device`, a
    /// drive run on a multi-GPU host silently lands on GPU 0.
    #[test]
    fn drive_device_flag_reaches_the_launch_entry() {
        let mut args = drive_args(DriveSourceKind::Random);
        args.device = Some(1);
        assert_eq!(drive_entry(&args).device, Some(1));
    }

    /// Backends whose CLI has no `--device` must not be sent one, so the
    /// absent flag has to stay absent rather than defaulting to 0.
    #[test]
    fn drive_without_the_flag_forwards_no_device() {
        let args = drive_args(DriveSourceKind::Random);
        assert_eq!(drive_entry(&args).device, None);
    }

    fn drive_args(source: DriveSourceKind) -> DriveArgs {
        DriveArgs {
            miner: PathBuf::from("miner"),
            source,
            topology: None,
            topology_preset: None,
            count: 10,
            seed: 0,
            list: None,
            target_energy: None,
            min_solutions: None,
            num_reads: None,
            num_sweeps: None,
            deadline_ms: 1000,
            device: None,
            report: None,
            utilization: None,
            yielding: false,
        }
    }

    #[test]
    fn topology_and_preset_together_is_error() {
        let mut a = drive_args(DriveSourceKind::Random);
        a.topology = Some(PathBuf::from("t.json"));
        a.topology_preset = Some("smoke".into());
        assert!(resolve_topology_text(&a).is_err());
    }

    #[test]
    fn random_source_defaults_to_the_advantage2_preset() {
        let a = drive_args(DriveSourceKind::Random);
        let text = resolve_topology_text(&a).unwrap().unwrap();
        let spec = parse_topology_spec(&text).unwrap();
        assert_eq!(spec.topology.nodes.len(), 4577);
    }

    #[test]
    fn list_defaults_to_no_topology() {
        let a = drive_args(DriveSourceKind::List);
        assert!(resolve_topology_text(&a).unwrap().is_none());
    }

    #[test]
    fn named_preset_resolves_to_its_embedded_spec() {
        let mut a = drive_args(DriveSourceKind::Random);
        a.topology_preset = Some("smoke".into());
        let text = resolve_topology_text(&a).unwrap().unwrap();
        assert!(parse_topology_spec(&text).is_ok());
    }
}

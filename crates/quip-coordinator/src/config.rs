//! Parse `config.toml` into a per-miner launch plan.

use crate::chain::{MinerKind, MinerSpecScale, NodeLogLevel};
use quip_proto::v1::Configure;

/// Errors raised while parsing a coordinator TOML config.
#[derive(Debug, PartialEq)]
pub enum ConfigError {
    /// No `[miner]` section present.
    MissingMiner,
    /// TOML parse failure (message from the parser).
    BadToml(String),
    /// No backend section, so there is nothing to launch. A coordinator with an
    /// empty launch plan binds its socket, follows the chain, and mines nothing
    /// — the failure this variant exists to make loud instead of silent.
    NoBackends {
        /// True when the config carries `[miner]` keys that only ever existed
        /// in the v0.2 `quip-miner` format, which is the usual reason a v0.3
        /// coordinator finds no backends.
        looks_like_v0_2: bool,
    },
    /// A required `[miner]` key is missing or unusable (blank host, port 0).
    MissingMinerKey {
        /// Key name, e.g. `public_host`.
        key: &'static str,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingMiner => write!(f, "missing [miner] section"),
            Self::BadToml(e) => write!(f, "bad toml: {e}"),
            Self::MissingMinerKey { key } => write!(f, "missing [miner].{key}"),
            Self::NoBackends { looks_like_v0_2 } => {
                write!(
                    f,
                    "no miner backend section: add at least one of [cpu], [cuda.N], \
                     [metal], [dwave]/[qpu]"
                )?;
                if *looks_like_v0_2 {
                    write!(
                        f,
                        ". This config carries v0.2 quip-miner keys (faucet_url / \
                         rest_host / rest_port); the v0.3 coordinator uses a different \
                         format, where each backend gets its own section (see \
                         docker/config.toml)"
                    )?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// Fully parsed coordinator config: chain endpoints, signer, and launch plan.
pub struct CoordinatorConfig {
    /// Validator WebSocket endpoints.
    pub validators: Vec<String>,
    /// Signer key URI or keystore path.
    pub signer_key: String,
    /// Faucet base URL for auto-funding the miner account. `None` disables it,
    /// and an underfunded account then fails startup with no way to recover.
    pub faucet_url: Option<String>,
    /// Balance floor in plancks, below which the coordinator funds or refuses.
    pub min_balance_plancks: u128,
    /// Amount requested per faucet attempt, in plancks.
    pub faucet_top_up_plancks: u128,
    /// How long to keep trying the faucet before giving up.
    pub funding_timeout_s: u64,
    /// Consecutive failed submissions of one proof, inside one quantum block,
    /// before the coordinator stops retrying it.
    pub max_submit_attempts: u32,
    /// One entry per supervised miner subprocess.
    pub launch: Vec<LaunchEntry>,
    /// Optional mining-attempt dashboard (`[dashboard]` section). `None`
    /// disables recording + the REST endpoint.
    pub dashboard: Option<DashboardConfig>,
    /// Optional node id for `set_descriptor`. When absent the coordinator
    /// derives one from the miner account.
    pub node_id: Option<String>,
    /// Display name for `set_descriptor`. Required to file a descriptor.
    pub node_name: Option<String>,
    /// Public host advertised in the descriptor. Required at parse.
    pub public_host: String,
    /// Public port advertised in the descriptor. Required at parse. Non-zero.
    pub public_port: u16,
    /// Whether the node advertises auto-mine. Defaults to `true`.
    pub auto_mine: bool,
    /// Log level advertised in the descriptor. Defaults to `Info`.
    pub node_log_level: NodeLogLevel,
}

/// `[dashboard]` config: where to serve the attempt logs and where they live.
pub struct DashboardConfig {
    /// HTTP listen address, e.g. `0.0.0.0:20100` (Caddy `/api/v1/*` proxy).
    pub listen: String,
    /// Root directory for `<qblock_id>/attempts.jsonl` files.
    pub data_dir: String,
}

/// One supervised miner: identity, binary path, and handshake `Configure`.
pub struct LaunchEntry {
    /// Miner id used on the session wire (`cpu-0`, `cuda-1`, …).
    pub miner_id: String,
    /// Executable name or path for the miner subprocess.
    pub binary: String,
    /// Handshake configure payload (queue depths, heartbeat, backend TOML).
    pub configure: Configure,
    /// Backend section that produced this entry (`cpu`, `cuda`, `metal`, `dwave`).
    pub backend: String,
}

impl LaunchEntry {
    /// Pallet kind for this backend section.
    #[must_use]
    pub fn miner_kind(&self) -> MinerKind {
        miner_kind_from_backend(&self.backend)
    }

    /// Descriptor miner spec for this launch entry.
    #[must_use]
    pub fn miner_spec(&self) -> MinerSpecScale {
        MinerSpecScale {
            kind: self.miner_kind(),
            label: Some(self.miner_id.as_bytes().to_vec()),
            backend: Some(self.backend.as_bytes().to_vec()),
            device_id: None,
        }
    }
}

/// Map a backend section name to the pallet `MinerKind`.
#[must_use]
pub fn miner_kind_from_backend(backend: &str) -> MinerKind {
    match backend {
        "cuda" => MinerKind::Gpu,
        "metal" => MinerKind::Metal,
        "dwave" => MinerKind::QpuDwave,
        _ => MinerKind::Cpu,
    }
}

/// Rank used to pick one kind for `participate` from a mixed fleet.
fn kind_rank(kind: MinerKind) -> u8 {
    match kind {
        MinerKind::Cpu => 0,
        MinerKind::Gpu => 1,
        MinerKind::Metal => 2,
        MinerKind::Asic => 3,
        MinerKind::QpuPasqal => 4,
        MinerKind::QpuIonq => 5,
        MinerKind::QpuIbm => 6,
        MinerKind::QpuDwave => 7,
    }
}

/// Kind a mixed fleet declares on `participate`: the highest-capability miner.
///
/// The pallet takes one kind. A node that runs CPU and Metal must not look
/// like a CPU-only node. Order: QPU, then ASIC, then Metal, then GPU, then CPU.
#[must_use]
pub fn participate_kind(launch: &[LaunchEntry]) -> MinerKind {
    launch
        .iter()
        .map(LaunchEntry::miner_kind)
        .max_by_key(|k| kind_rank(*k))
        .unwrap_or(MinerKind::Cpu)
}

/// Values the round machine needs to file a node descriptor.
#[derive(Clone, Debug)]
pub struct DescriptorParams {
    /// Optional configured node id. When absent the account hex is used.
    pub node_id: Option<String>,
    /// Display name. Required to file.
    pub node_name: Option<String>,
    /// Optional public host.
    pub public_host: Option<String>,
    /// Optional public port.
    pub public_port: Option<u16>,
    /// Advertised auto-mine flag.
    pub auto_mine: bool,
    /// Advertised log level.
    pub log_level: NodeLogLevel,
    /// RPC endpoints, from `[miner].validators`.
    pub rpc_endpoints: Vec<String>,
    /// Miner specs derived from the launch plan.
    pub miners: Vec<MinerSpecScale>,
}

impl Default for DescriptorParams {
    fn default() -> Self {
        Self {
            node_id: None,
            node_name: None,
            public_host: None,
            public_port: None,
            auto_mine: true,
            log_level: NodeLogLevel::Info,
            rpc_endpoints: Vec::new(),
            miners: Vec::new(),
        }
    }
}

impl DescriptorParams {
    /// Build descriptor inputs from a parsed config.
    #[must_use]
    pub fn from_config(cfg: &CoordinatorConfig) -> Self {
        Self {
            node_id: cfg.node_id.clone(),
            node_name: cfg.node_name.clone(),
            public_host: Some(cfg.public_host.clone()),
            public_port: Some(cfg.public_port),
            auto_mine: cfg.auto_mine,
            log_level: cfg.node_log_level,
            rpc_endpoints: cfg.validators.clone(),
            miners: cfg.launch.iter().map(LaunchEntry::miner_spec).collect(),
        }
    }
}

/// Validators used when `[miner].validators` is absent: the container-network
/// name first, then a node on this host.
///
/// The v0.2 `quip-miner` applied exactly this fallback, and its shipped config
/// template ships the key commented out on the strength of it. Without the
/// fallback such a config yields an empty validator list, and every chain read
/// fails with "no validators configured" — a coordinator that runs and never
/// mines. An explicit empty list is still honored as an explicit choice.
pub const DEFAULT_VALIDATORS: [&str; 2] = ["ws://quip-validator:9944", "ws://127.0.0.1:9944"];

fn make_configure(table: &toml::Table) -> Configure {
    let u32_of = |k: &str, d: u32| {
        table
            .get(k)
            .and_then(toml::Value::as_integer)
            .map_or(d, |i| {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "config u32 knobs are small non-negative integers"
                )]
                {
                    i as u32
                }
            })
    };
    // Strip coordinator-owned keys; the rest passes through verbatim.
    let mut passthrough = table.clone();
    for k in [
        "binary",
        "queue_depth",
        "idle_timeout_s",
        "heartbeat_s",
        "reconnect_window_s",
    ] {
        let _ = passthrough.remove(k);
    }
    Configure {
        queue_depth: u32_of("queue_depth", 3),
        idle_timeout_s: u32_of("idle_timeout_s", 300),
        heartbeat_s: u32_of("heartbeat_s", 15),
        reconnect_window_s: u32_of("reconnect_window_s", 60),
        backend_toml: toml::to_string(&passthrough).unwrap_or_default(),
    }
}

fn default_binary(backend: &str) -> String {
    match backend {
        "dwave" => "quip-dwave-qa".into(),
        other => format!("quip-{other}-sa"),
    }
}

fn entry(miner_id: &str, backend: &str, table: &toml::Table) -> LaunchEntry {
    let binary = table
        .get("binary")
        .and_then(|v| v.as_str())
        .map_or_else(|| default_binary(backend), String::from);
    LaunchEntry {
        miner_id: miner_id.into(),
        binary,
        configure: make_configure(table),
        backend: backend.into(),
    }
}

/// Identity keys from `[miner]` that feed `set_descriptor`.
struct MinerIdentity {
    node_id: Option<String>,
    node_name: Option<String>,
    public_host: String,
    public_port: u16,
    auto_mine: bool,
    node_log_level: NodeLogLevel,
}

fn optional_str(table: &toml::Table, key: &str) -> Option<String> {
    table
        .get(key)
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
}

fn parse_node_log_level(raw: Option<&str>) -> NodeLogLevel {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("debug" | "trace") => NodeLogLevel::Debug,
        Some("warn" | "warning") => NodeLogLevel::Warning,
        Some("error") => NodeLogLevel::Error,
        _ => NodeLogLevel::Info,
    }
}

fn parse_launch(root: &toml::Table) -> Vec<LaunchEntry> {
    let mut launch = Vec::new();
    if let Some(t) = root.get("cpu").and_then(|v| v.as_table()) {
        launch.push(entry("cpu-0", "cpu", t));
    }
    if let Some(cuda) = root.get("cuda").and_then(|v| v.as_table()) {
        let mut idxs: Vec<&String> = cuda.keys().collect();
        idxs.sort();
        for k in idxs {
            if let Some(t) = cuda.get(k).and_then(|v| v.as_table()) {
                launch.push(entry(&format!("cuda-{k}"), "cuda", t));
            }
        }
    }
    if let Some(t) = root.get("metal").and_then(|v| v.as_table()) {
        launch.push(entry("metal-0", "metal", t));
    }
    // Prefer [dwave], fall back to [qpu].
    if let Some(t) = root
        .get("dwave")
        .or_else(|| root.get("qpu"))
        .and_then(|v| v.as_table())
    {
        launch.push(entry("qpu-0", "dwave", t));
    }
    launch
}

fn parse_dashboard(root: &toml::Table) -> Option<DashboardConfig> {
    root.get("dashboard")
        .and_then(|v| v.as_table())
        .and_then(|t| {
            let listen = t.get("listen").and_then(|v| v.as_str())?;
            let data_dir = t.get("data_dir").and_then(|v| v.as_str())?;
            Some(DashboardConfig {
                listen: listen.to_string(),
                data_dir: data_dir.to_string(),
            })
        })
}

fn parse_miner_identity(miner: &toml::Table) -> Result<MinerIdentity, ConfigError> {
    let public_host = optional_str(miner, "public_host")
        .ok_or(ConfigError::MissingMinerKey { key: "public_host" })?;
    let public_port = miner
        .get("public_port")
        .and_then(toml::Value::as_integer)
        .and_then(|i| u16::try_from(i).ok().filter(|&p| p > 0))
        .ok_or(ConfigError::MissingMinerKey { key: "public_port" })?;
    let auto_mine = miner
        .get("auto_mine")
        .and_then(toml::Value::as_bool)
        .unwrap_or(true);
    Ok(MinerIdentity {
        node_id: optional_str(miner, "node_id"),
        node_name: optional_str(miner, "node_name"),
        public_host,
        public_port,
        auto_mine,
        node_log_level: parse_node_log_level(miner.get("log_level").and_then(toml::Value::as_str)),
    })
}

/// Parse a coordinator TOML config into validators, signer, and launch plan.
///
/// Section mapping: `[cpu]`→`cpu-0`, `[cuda.N]`→`cuda-N`, `[metal]`→`metal-0`,
/// `[dwave]`/`[qpu]`→`qpu-0`.
///
/// # Errors
/// Returns [`ConfigError::MissingMiner`] when the `[miner]` section is absent,
/// [`ConfigError::MissingMinerKey`] when a required `[miner]` identity key is
/// missing or unusable, or [`ConfigError::BadToml`] when the input is not
/// valid TOML.
pub fn parse_config(toml_text: &str) -> Result<CoordinatorConfig, ConfigError> {
    let root: toml::Table =
        toml::from_str(toml_text).map_err(|e| ConfigError::BadToml(e.to_string()))?;
    let miner = root
        .get("miner")
        .and_then(|v| v.as_table())
        .ok_or(ConfigError::MissingMiner)?;
    // Absent key → the v0.2 fallback pair. A present-but-empty array is an
    // explicit "no validators" and is left alone.
    let validators: Vec<String> = miner
        .get("validators")
        .and_then(|v| v.as_array())
        .map_or_else(
            || {
                DEFAULT_VALIDATORS
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect()
            },
            |a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            },
        );
    let signer_key = miner
        .get("signer_key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Auto-funding. A blank `faucet_url` reads as "disabled" so an operator can
    // switch it off by emptying the value instead of deleting the line.
    let faucet_url = miner
        .get("faucet_url")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from);
    let plancks_of = |key: &str, default: u128| -> u128 {
        miner
            .get(key)
            .and_then(toml::Value::as_integer)
            .and_then(|i| u128::try_from(i).ok())
            .unwrap_or(default)
    };
    let min_balance_plancks = plancks_of(
        "min_balance_plancks",
        crate::funding::DEFAULT_MIN_BALANCE_PLANCKS,
    );
    let faucet_top_up_plancks = plancks_of(
        "faucet_top_up_plancks",
        crate::funding::DEFAULT_TOP_UP_PLANCKS,
    );
    let funding_timeout_s = miner
        .get("funding_timeout_s")
        .and_then(toml::Value::as_integer)
        .and_then(|i| u64::try_from(i).ok())
        .unwrap_or(crate::funding::DEFAULT_FUNDING_TIMEOUT.as_secs());
    let max_submit_attempts = miner
        .get("max_submit_attempts")
        .and_then(toml::Value::as_integer)
        .and_then(|i| u32::try_from(i).ok())
        .unwrap_or(5);

    let launch = parse_launch(&root);
    let dashboard = parse_dashboard(&root);

    if launch.is_empty() {
        // Keys that only ever existed in the v0.2 `quip-miner` config. The v0.3
        // coordinator reads a different file, and silently ignoring the rest of
        // it would leave an operator with a running process that never mines.
        let looks_like_v0_2 = ["faucet_url", "rest_host", "rest_port"]
            .iter()
            .any(|k| miner.contains_key(*k));
        return Err(ConfigError::NoBackends { looks_like_v0_2 });
    }

    let identity = parse_miner_identity(miner)?;

    Ok(CoordinatorConfig {
        validators,
        signer_key,
        faucet_url,
        min_balance_plancks,
        faucet_top_up_plancks,
        funding_timeout_s,
        max_submit_attempts,
        launch,
        dashboard,
        node_id: identity.node_id,
        node_name: identity.node_name,
        public_host: identity.public_host,
        public_port: identity.public_port,
        auto_mine: identity.auto_mine,
        node_log_level: identity.node_log_level,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
[miner]
validators = ["ws://127.0.0.1:9944"]
signer_key = "//Alice"
public_host = "203.0.113.10"
public_port = 20050

[cpu]
binary = "quip-cpu-gibbs"
num_cpus = 8
queue_depth = 4
idle_timeout_s = 120

[cuda.0]
device_index = 0

[cuda.1]
device_index = 1

[dwave]
daily_budget = "30s"
min_block_budget = "90s"
budget_cap = "5m"
"#;

    #[test]
    fn maps_sections_to_launch_entries() {
        let c = parse_config(SAMPLE).unwrap();
        assert_eq!(c.validators, vec!["ws://127.0.0.1:9944"]);
        let ids: Vec<&str> = c.launch.iter().map(|e| e.miner_id.as_str()).collect();
        assert_eq!(ids, vec!["cpu-0", "cuda-0", "cuda-1", "qpu-0"]);
    }

    #[test]
    fn binary_override_and_default() {
        let c = parse_config(SAMPLE).unwrap();
        let cpu = c.launch.iter().find(|e| e.miner_id == "cpu-0").unwrap();
        assert_eq!(cpu.binary, "quip-cpu-gibbs");
        assert_eq!(cpu.configure.queue_depth, 4);
        assert_eq!(cpu.configure.idle_timeout_s, 120);
        let cuda = c.launch.iter().find(|e| e.miner_id == "cuda-0").unwrap();
        assert_eq!(cuda.binary, "quip-cuda-sa");
    }

    #[test]
    fn backend_toml_carries_unknown_keys() {
        // The dwave miner consumes its budget knobs from `backend_toml`; the
        // coordinator forwards every non-coordinator key verbatim, so all three
        // budget dimensions reach the miner.
        let c = parse_config(SAMPLE).unwrap();
        let dwave = c.launch.iter().find(|e| e.miner_id == "qpu-0").unwrap();
        let backend_toml = &dwave.configure.backend_toml;
        assert!(backend_toml.contains("daily_budget"));
        assert!(backend_toml.contains("min_block_budget"));
        assert!(backend_toml.contains("budget_cap"));
    }

    #[test]
    fn dwave_default_binary_is_qa() {
        let c = parse_config(SAMPLE).unwrap();
        let dwave = c.launch.iter().find(|e| e.miner_id == "qpu-0").unwrap();
        assert_eq!(dwave.binary, "quip-dwave-qa");
    }

    #[test]
    fn dashboard_section_parsed_when_present() {
        let c = parse_config(SAMPLE).unwrap();
        assert!(c.dashboard.is_none());

        let with = format!(
            "{SAMPLE}\n[dashboard]\nlisten = \"127.0.0.1:9090\"\ndata_dir = \"/data/attempts\"\n"
        );
        let d = parse_config(&with).unwrap().dashboard.unwrap();
        assert_eq!(d.listen, "127.0.0.1:9090");
        assert_eq!(d.data_dir, "/data/attempts");
    }

    #[test]
    fn dashboard_ignored_when_missing_keys() {
        // A [dashboard] section missing a required key disables it rather than
        // erroring, so a partial config still boots.
        let partial = format!("{SAMPLE}\n[dashboard]\nlisten = \"127.0.0.1:9090\"\n");
        assert!(parse_config(&partial).unwrap().dashboard.is_none());
    }

    #[test]
    fn config_without_any_backend_is_rejected() {
        let no_backend =
            "[miner]\nvalidators = [\"ws://127.0.0.1:9944\"]\nsigner_key = \"//Alice\"\n";
        assert!(matches!(
            parse_config(no_backend),
            Err(ConfigError::NoBackends {
                looks_like_v0_2: false
            })
        ));
    }

    /// The exact failure seen on a live node-manager stack: a v0.2.1 `quip-miner`
    /// config handed to the v0.3 coordinator. It parses as valid TOML and has a
    /// `[miner]` section, so nothing rejected it — it just launched no miners.
    #[test]
    fn v0_2_miner_config_is_rejected_with_a_format_hint() {
        let v0_2 = r#"
[miner]
signer_key = "/data/keystore.json"
faucet_url = "https://faucet.testnet.quip.network"
rest_host = "0.0.0.0"
rest_port = 8086
"#;
        let Err(err) = parse_config(v0_2) else {
            panic!("a v0.2 miner config must not parse as a v0.3 coordinator config");
        };
        assert!(matches!(
            err,
            ConfigError::NoBackends {
                looks_like_v0_2: true
            }
        ));
        let msg = err.to_string();
        assert!(msg.contains("[cpu]"), "{msg}");
        assert!(msg.contains("v0.2"), "{msg}");
    }

    /// v0.2 parity: a config that omits `validators` must still reach a chain.
    /// The node-manager template ships the key commented out, so without the
    /// fallback the coordinator starts with nowhere to connect and never mines.
    #[test]
    fn omitted_validators_fall_back_to_the_v0_2_pair() {
        let cfg = parse_config(
            "[miner]\nsigner_key = \"//Alice\"\n\
             public_host = \"203.0.113.10\"\npublic_port = 20050\n\n[cpu]\n",
        )
        .unwrap();
        assert_eq!(cfg.validators, DEFAULT_VALIDATORS.to_vec());
    }

    #[test]
    fn explicit_validators_win_over_the_fallback() {
        let cfg = parse_config(
            "[miner]\nvalidators = [\"ws://example:9944\"]\nsigner_key = \"//Alice\"\n\
             public_host = \"203.0.113.10\"\npublic_port = 20050\n\n[cpu]\n",
        )
        .unwrap();
        assert_eq!(cfg.validators, vec!["ws://example:9944".to_string()]);
    }

    #[test]
    fn explicit_empty_validators_stay_empty() {
        // An explicit `[]` is a deliberate choice, not an omission.
        let cfg = parse_config(
            "[miner]\nvalidators = []\nsigner_key = \"//Alice\"\n\
             public_host = \"203.0.113.10\"\npublic_port = 20050\n\n[cpu]\n",
        )
        .unwrap();
        assert!(cfg.validators.is_empty());
    }

    #[test]
    fn funding_defaults_apply_when_keys_are_absent() {
        let cfg = parse_config(SAMPLE).unwrap();
        assert_eq!(cfg.faucet_url, None);
        assert_eq!(
            cfg.min_balance_plancks,
            crate::funding::DEFAULT_MIN_BALANCE_PLANCKS
        );
        assert_eq!(
            cfg.faucet_top_up_plancks,
            crate::funding::DEFAULT_TOP_UP_PLANCKS
        );
        assert_eq!(cfg.funding_timeout_s, 600);
    }

    #[test]
    fn funding_keys_are_parsed_and_overridable() {
        let text = "[miner]\nsigner_key = \"//Alice\"\n\
                    public_host = \"203.0.113.10\"\npublic_port = 20050\n\
                    faucet_url = \"https://f.example\"\n\
                    min_balance_plancks = 5\nfaucet_top_up_plancks = 50\n\
                    funding_timeout_s = 30\n\n[cpu]\n";
        let cfg = parse_config(text).unwrap();
        assert_eq!(cfg.faucet_url.as_deref(), Some("https://f.example"));
        assert_eq!(cfg.min_balance_plancks, 5);
        assert_eq!(cfg.faucet_top_up_plancks, 50);
        assert_eq!(cfg.funding_timeout_s, 30);
    }

    #[test]
    fn max_submit_attempts_defaults_and_overrides() {
        let cfg = parse_config(SAMPLE).unwrap();
        assert_eq!(cfg.max_submit_attempts, 5);
        let cfg = parse_config(
            "[miner]\nvalidators = [\"ws://x\"]\nsigner_key = \"//Alice\"\n\
             public_host = \"203.0.113.10\"\npublic_port = 20050\n\
             max_submit_attempts = 2\n\n[cpu]\n",
        )
        .unwrap();
        assert_eq!(cfg.max_submit_attempts, 2);
    }

    /// Emptying the value is how an operator opts out without deleting the key.
    #[test]
    fn blank_faucet_url_disables_auto_funding() {
        for blank in ["\"\"", "\"   \""] {
            let text = format!(
                "[miner]\nsigner_key = \"//Alice\"\n\
                 public_host = \"203.0.113.10\"\npublic_port = 20050\n\
                 faucet_url = {blank}\n\n[cpu]\n"
            );
            assert_eq!(parse_config(&text).unwrap().faucet_url, None, "{blank}");
        }
    }

    /// The operator's live config: descriptor keys live under `[miner]`.
    const LIVE: &str = r#"
[miner]
validators = ["ws://127.0.0.1:9944"]
signer_key = "/Users/carback1/quip-data-3/keystore.json"
faucet_url = "https://faucet.testnet.quip.network"
rest_host = "127.0.0.1"
rest_port = 20100
node_name = "Tesla"
public_host = "96.233.112.201"
public_port = 20050
log_level = "info"

[cpu]
binary = "/Users/carback1/quip-data-3/bin/quip-cpu-sa"
num_cpus = 6

[metal]
binary = "/Users/carback1/quip-data-3/bin/quip-metal-sa"
utilization = 80
yielding = false
active_util = 50
idle_after_s = 600
"#;

    #[test]
    fn live_config_reads_descriptor_keys_from_miner() {
        let c = parse_config(LIVE).unwrap();
        assert_eq!(c.node_name.as_deref(), Some("Tesla"));
        assert_eq!(c.public_host, "96.233.112.201");
        assert_eq!(c.node_log_level, NodeLogLevel::Info);
        assert_eq!(c.node_id, None);
        assert_eq!(c.public_port, 20050);
        assert!(c.auto_mine);
        let ids: Vec<&str> = c.launch.iter().map(|e| e.miner_id.as_str()).collect();
        assert_eq!(ids, vec!["cpu-0", "metal-0"]);
        let backends: Vec<&str> = c.launch.iter().map(|e| e.backend.as_str()).collect();
        assert_eq!(backends, vec!["cpu", "metal"]);
    }

    #[test]
    fn rest_port_is_not_the_descriptor_public_port() {
        let c = parse_config(LIVE).unwrap();
        assert_eq!(c.public_port, 20050);
        assert_ne!(c.public_port, 20100);
    }

    #[test]
    fn metal_section_is_metal_kind() {
        let c = parse_config(
            "[miner]\nsigner_key = \"//Alice\"\n\
             public_host = \"203.0.113.10\"\npublic_port = 20050\n\n[metal]\n",
        )
        .unwrap();
        let metal = c.launch.first().expect("one metal entry");
        assert_eq!(metal.backend, "metal");
        assert_eq!(metal.miner_kind(), MinerKind::Metal);
        assert_eq!(participate_kind(&c.launch), MinerKind::Metal);
    }

    #[test]
    fn mixed_cpu_metal_declares_metal_on_participate() {
        let c = parse_config(LIVE).unwrap();
        assert_eq!(
            c.launch
                .iter()
                .map(LaunchEntry::miner_kind)
                .collect::<Vec<_>>(),
            vec![MinerKind::Cpu, MinerKind::Metal]
        );
        assert_eq!(participate_kind(&c.launch), MinerKind::Metal);
    }

    #[test]
    fn descriptor_optional_keys_parse_when_present() {
        let text = "[miner]\nsigner_key = \"//Alice\"\n\
                    node_id = \"tesla-1\"\nnode_name = \"Tesla\"\n\
                    public_host = \"203.0.113.10\"\n\
                    public_port = 20050\nauto_mine = false\nlog_level = \"debug\"\n\n[cpu]\n";
        let c = parse_config(text).unwrap();
        assert_eq!(c.node_id.as_deref(), Some("tesla-1"));
        assert_eq!(c.node_name.as_deref(), Some("Tesla"));
        assert_eq!(c.public_host, "203.0.113.10");
        assert_eq!(c.public_port, 20050);
        assert!(!c.auto_mine);
        assert_eq!(c.node_log_level, NodeLogLevel::Debug);
    }

    #[test]
    fn missing_descriptor_keys_take_defaults() {
        let c = parse_config(SAMPLE).unwrap();
        assert_eq!(c.node_id, None);
        assert_eq!(c.node_name, None);
        assert_eq!(c.public_host, "203.0.113.10");
        assert_eq!(c.public_port, 20050);
        assert!(c.auto_mine);
        assert_eq!(c.node_log_level, NodeLogLevel::Info);
    }

    fn miner_cpu(extra: &str) -> String {
        format!("[miner]\nsigner_key = \"//Alice\"\n{extra}\n[cpu]\n")
    }

    #[test]
    fn public_host_and_public_port_parse_when_both_present() {
        let c = parse_config(&miner_cpu(
            "public_host = \"203.0.113.10\"\npublic_port = 20050\n",
        ))
        .unwrap();
        assert_eq!(c.public_host, "203.0.113.10");
        assert_eq!(c.public_port, 20050);
    }

    #[test]
    fn missing_public_host_is_a_config_error_naming_the_key() {
        let Err(err) = parse_config(&miner_cpu("public_port = 20050\n")) else {
            panic!("missing public_host must not parse");
        };
        assert!(matches!(
            err,
            ConfigError::MissingMinerKey { key: "public_host" }
        ));
        let msg = err.to_string();
        assert!(msg.contains("public_host"), "{msg}");
        assert!(msg.contains("[miner]"), "{msg}");
    }

    #[test]
    fn missing_public_port_is_a_config_error_naming_the_key() {
        let Err(err) = parse_config(&miner_cpu("public_host = \"203.0.113.10\"\n")) else {
            panic!("missing public_port must not parse");
        };
        assert!(matches!(
            err,
            ConfigError::MissingMinerKey { key: "public_port" }
        ));
        let msg = err.to_string();
        assert!(msg.contains("public_port"), "{msg}");
        assert!(msg.contains("[miner]"), "{msg}");
    }

    #[test]
    fn blank_public_host_is_rejected_as_missing() {
        for blank in ["\"\"", "\"   \""] {
            let text = miner_cpu(&format!("public_host = {blank}\npublic_port = 20050\n"));
            let Err(err) = parse_config(&text) else {
                panic!("blank public_host {blank} must not parse");
            };
            assert!(
                matches!(err, ConfigError::MissingMinerKey { key: "public_host" }),
                "{blank}: {err}"
            );
            let msg = err.to_string();
            assert!(msg.contains("public_host"), "{blank}: {msg}");
            assert!(msg.contains("[miner]"), "{blank}: {msg}");
        }
    }

    #[test]
    fn zero_public_port_is_rejected() {
        let Err(err) = parse_config(&miner_cpu(
            "public_host = \"203.0.113.10\"\npublic_port = 0\n",
        )) else {
            panic!("public_port = 0 must not parse");
        };
        assert!(matches!(
            err,
            ConfigError::MissingMinerKey { key: "public_port" }
        ));
        let msg = err.to_string();
        assert!(msg.contains("public_port"), "{msg}");
        assert!(msg.contains("[miner]"), "{msg}");
    }

    /// The shipped v0.3 template must survive its own parser.
    #[test]
    fn shipped_docker_template_parses() {
        let text = include_str!("../../../docker/config.toml");
        let cfg = parse_config(text).expect("docker/config.toml must parse");
        assert!(
            !cfg.launch.is_empty(),
            "shipped template must declare a backend"
        );
        // The image must self-fund out of the box: a bare `docker run` has no
        // way to hand the account money otherwise.
        assert!(
            cfg.faucet_url.is_some(),
            "shipped template must configure a faucet"
        );
        assert!(
            !cfg.signer_key.is_empty(),
            "shipped template must point at a keystore path"
        );
    }
}

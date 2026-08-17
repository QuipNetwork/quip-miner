//! Faucet auto-funding for the miner account.
//!
//! A coordinator whose account cannot pay transaction fees mines normally and
//! then fails at every submit, so an unfunded wallet looks like a mining bug.
//! [`ensure_funded`] settles that at startup: read the balance, top up through
//! the configured faucet when it is short, and report clearly when it cannot.
//!
//! The chain and the faucet both sit behind traits so the retry and backoff
//! logic is testable without a validator or a live faucet.

use async_trait::async_trait;
use std::time::Duration;

/// Balance floor for a usable miner account, in plancks (2 UNIT). Matches the
/// v0.2 `DEFAULT_MIN_BALANCE_PLANCKS`.
pub const DEFAULT_MIN_BALANCE_PLANCKS: u128 = 2_000_000_000_000;

/// Amount requested per faucet call, in plancks (10 UNIT). Matches the v0.2
/// `DEFAULT_FAUCET_TOP_UP_PLANCKS`.
pub const DEFAULT_TOP_UP_PLANCKS: u128 = 10_000_000_000_000;

/// How long to keep trying before giving up.
pub const DEFAULT_FUNDING_TIMEOUT: Duration = Duration::from_mins(10);

/// First retry gap. Doubles up to [`BACKOFF_MAX`].
const BACKOFF_START: Duration = Duration::from_secs(2);

/// Ceiling on the retry gap.
const BACKOFF_MAX: Duration = Duration::from_secs(30);

/// Reads the on-chain free balance of an account.
#[async_trait]
pub trait BalanceSource: Send + Sync {
    /// Free balance of `account` in plancks.
    ///
    /// # Errors
    /// Returns a message when the chain cannot be read.
    async fn free_balance(&self, account: [u8; 32]) -> Result<u128, String>;
}

/// Outcome of one faucet request.
#[derive(Debug, PartialEq, Eq)]
pub enum FaucetError {
    /// The faucet rejected the request itself (HTTP 400). Retrying cannot help.
    Permanent(String),
    /// The account is already above the faucet's cap (HTTP 403). Benign: the
    /// balance read decides whether the floor is actually met.
    AlreadyFunded(String),
    /// Rate limited, server error, or transport failure. Worth another try.
    Transient(String),
}

impl std::fmt::Display for FaucetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Permanent(s) | Self::AlreadyFunded(s) | Self::Transient(s) => f.write_str(s),
        }
    }
}

/// Requests funds for an account.
#[async_trait]
pub trait Faucet: Send + Sync {
    /// Ask for `amount` plancks to be sent to `dest`.
    ///
    /// # Errors
    /// Returns the classified failure so the caller can decide to retry.
    async fn request(&self, dest: [u8; 32], amount: u128) -> Result<(), FaucetError>;
}

/// Why the account could not be brought up to the floor.
#[derive(Debug)]
pub enum FundingError {
    /// Balance is below the floor and no faucet is configured.
    Underfunded {
        /// Free balance actually observed.
        balance: u128,
        /// Floor required to mine.
        threshold: u128,
    },
    /// A faucet is configured, but the balance never reached the floor within
    /// the timeout.
    FaucetExhausted {
        /// Last balance observed.
        balance: u128,
        /// Floor required to mine.
        threshold: u128,
        /// Last thing the faucet said.
        last_note: String,
    },
    /// The faucet rejected the request in a way retrying cannot fix.
    FaucetRejected(String),
    /// The chain could not be read at all.
    Chain(String),
}

impl std::fmt::Display for FundingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Underfunded { balance, threshold } => write!(
                f,
                "miner account is underfunded: balance {balance} plancks is below the \
                 {threshold} needed to pay submit fees, and no faucet_url is configured"
            ),
            Self::FaucetExhausted {
                balance,
                threshold,
                last_note,
            } => write!(
                f,
                "faucet did not fund the miner account in time: balance {balance} plancks \
                 is still below {threshold} (last faucet response: {last_note})"
            ),
            Self::FaucetRejected(s) => write!(f, "faucet rejected the request: {s}"),
            Self::Chain(s) => write!(f, "cannot read miner account balance: {s}"),
        }
    }
}

impl std::error::Error for FundingError {}

/// Funding knobs, resolved from `[miner]` config.
pub struct FundingParams {
    /// Faucet base URL. `None` disables auto-funding.
    pub faucet_url: Option<String>,
    /// Balance floor in plancks.
    pub min_balance: u128,
    /// Amount to request per attempt, in plancks.
    pub top_up: u128,
    /// Total time to keep trying before giving up.
    pub timeout: Duration,
}

impl Default for FundingParams {
    fn default() -> Self {
        Self {
            faucet_url: None,
            min_balance: DEFAULT_MIN_BALANCE_PLANCKS,
            top_up: DEFAULT_TOP_UP_PLANCKS,
            timeout: DEFAULT_FUNDING_TIMEOUT,
        }
    }
}

/// How long a single faucet POST may take before it counts as transient.
const FAUCET_HTTP_TIMEOUT: Duration = Duration::from_secs(15);

/// Live faucet client over HTTPS.
///
/// Speaks the contract served by `gitlab.com/quip.network/faucet`, the same one
/// the v0.2 miner used: `POST {url}/request` with `{"dest": "0x…", "amount": n}`.
pub struct HttpFaucet {
    base_url: String,
    client: reqwest::Client,
}

impl HttpFaucet {
    /// Build a faucet client for `base_url`.
    ///
    /// # Errors
    /// Returns a message when the HTTP client cannot be constructed.
    pub fn new(base_url: &str) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(FAUCET_HTTP_TIMEOUT)
            .build()
            .map_err(|e| format!("faucet http client: {e}"))?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client,
        })
    }
}

/// Classify a faucet HTTP status.
///
/// Retrying is the default: only a malformed request (400) is permanent, and
/// 403 means the faucet considers the account already capped, which the balance
/// read reconciles.
fn classify_status(status: u16, detail: &str) -> Result<(), FaucetError> {
    let msg = format!("faucet returned {status}: {detail}");
    match status {
        200..=299 => Ok(()),
        400 => Err(FaucetError::Permanent(msg)),
        403 => Err(FaucetError::AlreadyFunded(msg)),
        _ => Err(FaucetError::Transient(msg)),
    }
}

#[async_trait]
impl Faucet for HttpFaucet {
    async fn request(&self, dest: [u8; 32], amount: u128) -> Result<(), FaucetError> {
        let dest_hex = format!("0x{}", hex_lower(&dest));
        let url = format!("{}/request", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({ "dest": dest_hex, "amount": amount }))
            .send()
            .await
            .map_err(|e| FaucetError::Transient(format!("faucet request to {url} failed: {e}")))?;
        let status = resp.status().as_u16();
        // The success body is discarded: the on-chain balance is the source of
        // truth, so a 2xx with a non-JSON body (proxy splash page, truncated
        // response) must not break the retry loop.
        let detail = resp.text().await.unwrap_or_default();
        classify_status(status, &detail)
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Next backoff gap, doubling from `current` up to [`BACKOFF_MAX`].
fn next_backoff(current: Duration) -> Duration {
    current.saturating_mul(2).min(BACKOFF_MAX)
}

/// Headline for the retry warning after one funding pass.
///
/// A missing balance is a chain-read failure, not a funding state.
fn retry_warning_message(last_balance: Option<u128>) -> &'static str {
    match last_balance {
        None => "cannot read miner account balance from the chain; retrying",
        Some(_) => "miner account is not funded yet; retrying",
    }
}

/// Bring `account` up to `params.min_balance`, topping up through the faucet if
/// one is configured. Returns the balance once it clears the floor.
///
/// The on-chain balance is the source of truth throughout: a faucet that
/// answers 403 (already funded) or rate-limits is not a failure by itself, only
/// a balance that never arrives is. `sleep` is injected so tests do not wait.
///
/// # Errors
/// Returns [`FundingError`] when the balance cannot be read, the faucet rejects
/// the request permanently, or the floor is not reached within the timeout.
pub async fn ensure_funded<B, F, S, Fut>(
    chain: &B,
    faucet: Option<&F>,
    account: [u8; 32],
    params: &FundingParams,
    mut sleep: S,
) -> Result<u128, FundingError>
where
    B: BalanceSource + ?Sized,
    F: Faucet + ?Sized,
    S: FnMut(Duration) -> Fut + Send,
    Fut: std::future::Future<Output = ()> + Send,
{
    let mut remaining = params.timeout;
    let mut backoff = BACKOFF_START;
    // Every path through the loop body assigns this before the budget check
    // reads it, so it needs no seed value.
    let mut last_note;
    // Last balance actually read. `None` while the chain has never answered,
    // which is distinct from "read zero".
    let mut last_balance: Option<u128> = None;
    let mut attempt: u32 = 0;
    let mut announced = false;

    loop {
        attempt = attempt.saturating_add(1);

        // The on-chain balance is authoritative and is re-read every pass: an
        // earlier mint may have settled, and a validator that was still booting
        // may have come up.
        match chain.free_balance(account).await {
            Ok(balance) => {
                last_balance = Some(balance);
                if balance >= params.min_balance {
                    if attempt == 1 {
                        tracing::trace!(balance, "miner account already funded");
                    } else {
                        tracing::info!(balance, attempt, "miner account is now funded");
                    }
                    return Ok(balance);
                }
                // Short, and nothing to top up from. Retrying cannot change
                // that, so fail now rather than burn the whole budget.
                let Some(faucet) = faucet else {
                    return Err(FundingError::Underfunded {
                        balance,
                        threshold: params.min_balance,
                    });
                };
                if !announced {
                    tracing::warn!(
                        balance,
                        threshold = params.min_balance,
                        top_up = params.top_up,
                        timeout_s = params.timeout.as_secs(),
                        "miner account is underfunded; requesting funds from the faucet"
                    );
                    announced = true;
                }
                match faucet.request(account, params.top_up).await {
                    Ok(()) => last_note = "requested".into(),
                    // Retrying a malformed or banned request cannot help.
                    Err(FaucetError::Permanent(e)) => return Err(FundingError::FaucetRejected(e)),
                    Err(e @ (FaucetError::AlreadyFunded(_) | FaucetError::Transient(_))) => {
                        last_note = e.to_string();
                    }
                }
            }
            // A chain that cannot be read yet is the co-start case: the node
            // manager launches the coordinator and its validator together. Keep
            // retrying inside the budget instead of exiting, or the container
            // crash-loops while its own validator boots.
            Err(e) => last_note = format!("chain read failed: {e}"),
        }

        if remaining <= backoff {
            return Err(match last_balance {
                Some(balance) => FundingError::FaucetExhausted {
                    balance,
                    threshold: params.min_balance,
                    last_note,
                },
                None => FundingError::Chain(last_note),
            });
        }
        let message = retry_warning_message(last_balance);
        match last_balance {
            None => tracing::warn!(
                attempt,
                retry_in_s = backoff.as_secs(),
                note = %last_note,
                "{}",
                message
            ),
            Some(balance) => tracing::warn!(
                attempt,
                balance,
                threshold = params.min_balance,
                retry_in_s = backoff.as_secs(),
                note = %last_note,
                "{}",
                message
            ),
        }
        sleep(backoff).await;
        remaining = remaining.saturating_sub(backoff);
        backoff = next_backoff(backoff);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Balance source that walks a scripted sequence, repeating the last value.
    struct ScriptedChain {
        values: Mutex<Vec<u128>>,
        calls: Mutex<usize>,
    }

    impl ScriptedChain {
        fn new(values: Vec<u128>) -> Self {
            Self {
                values: Mutex::new(values),
                calls: Mutex::new(0),
            }
        }
        fn calls(&self) -> usize {
            *self.calls.lock().unwrap()
        }
    }

    #[async_trait]
    impl BalanceSource for ScriptedChain {
        async fn free_balance(&self, _a: [u8; 32]) -> Result<u128, String> {
            let mut n = self.calls.lock().unwrap();
            let vals = self.values.lock().unwrap();
            let idx = (*n).min(vals.len().saturating_sub(1));
            *n += 1;
            vals.get(idx).copied().ok_or_else(|| "empty".to_string())
        }
    }

    struct FailingChain;
    #[async_trait]
    impl BalanceSource for FailingChain {
        async fn free_balance(&self, _a: [u8; 32]) -> Result<u128, String> {
            Err("rpc down".into())
        }
    }

    /// Faucet that returns a scripted outcome each call, repeating the last.
    struct ScriptedFaucet {
        outcomes: Mutex<Vec<Result<(), FaucetError>>>,
        calls: Mutex<usize>,
    }

    impl ScriptedFaucet {
        fn new(outcomes: Vec<Result<(), FaucetError>>) -> Self {
            Self {
                outcomes: Mutex::new(outcomes),
                calls: Mutex::new(0),
            }
        }
        fn calls(&self) -> usize {
            *self.calls.lock().unwrap()
        }
    }

    #[async_trait]
    impl Faucet for ScriptedFaucet {
        async fn request(&self, _d: [u8; 32], _a: u128) -> Result<(), FaucetError> {
            let mut n = self.calls.lock().unwrap();
            let outs = self.outcomes.lock().unwrap();
            let idx = (*n).min(outs.len().saturating_sub(1));
            *n += 1;
            match outs.get(idx) {
                Some(Ok(())) | None => Ok(()),
                Some(Err(FaucetError::Permanent(s))) => Err(FaucetError::Permanent(s.clone())),
                Some(Err(FaucetError::AlreadyFunded(s))) => {
                    Err(FaucetError::AlreadyFunded(s.clone()))
                }
                Some(Err(FaucetError::Transient(s))) => Err(FaucetError::Transient(s.clone())),
            }
        }
    }

    fn params(url: Option<&str>) -> FundingParams {
        FundingParams {
            faucet_url: url.map(String::from),
            min_balance: 100,
            top_up: 1000,
            timeout: Duration::from_mins(1),
        }
    }

    /// No sleeping in tests; the budget still advances via the injected gaps.
    async fn no_sleep(_d: Duration) {}

    #[tokio::test]
    async fn already_funded_never_calls_the_faucet() {
        let chain = ScriptedChain::new(vec![500]);
        let faucet = ScriptedFaucet::new(vec![Ok(())]);
        let got = ensure_funded(
            &chain,
            Some(&faucet),
            [0u8; 32],
            &params(Some("u")),
            no_sleep,
        )
        .await
        .unwrap();
        assert_eq!(got, 500);
        assert_eq!(faucet.calls(), 0, "must not hit the faucet when funded");
    }

    #[tokio::test]
    async fn underfunded_without_faucet_is_an_error() {
        let chain = ScriptedChain::new(vec![5]);
        let err = ensure_funded(
            &chain,
            None::<&ScriptedFaucet>,
            [0u8; 32],
            &params(None),
            no_sleep,
        )
        .await
        .unwrap_err();
        assert!(matches!(
            err,
            FundingError::Underfunded {
                balance: 5,
                threshold: 100
            }
        ));
        assert!(err.to_string().contains("no faucet_url"), "{err}");
    }

    #[tokio::test]
    async fn funds_on_the_first_attempt() {
        // Short, then funded after the request settles.
        let chain = ScriptedChain::new(vec![5, 1000]);
        let faucet = ScriptedFaucet::new(vec![Ok(())]);
        let got = ensure_funded(
            &chain,
            Some(&faucet),
            [0u8; 32],
            &params(Some("u")),
            no_sleep,
        )
        .await
        .unwrap();
        assert_eq!(got, 1000);
        assert_eq!(faucet.calls(), 1);
    }

    #[tokio::test]
    async fn retries_through_transient_failures() {
        // Two transient failures, then the mint lands.
        let chain = ScriptedChain::new(vec![5, 5, 5, 1000]);
        let faucet = ScriptedFaucet::new(vec![
            Err(FaucetError::Transient("429".into())),
            Err(FaucetError::Transient("502".into())),
            Ok(()),
        ]);
        let got = ensure_funded(
            &chain,
            Some(&faucet),
            [0u8; 32],
            &params(Some("u")),
            no_sleep,
        )
        .await
        .unwrap();
        assert_eq!(got, 1000);
        assert_eq!(faucet.calls(), 3);
    }

    /// 403 means the faucet thinks the account is capped. That is not fatal —
    /// only the on-chain balance decides.
    #[tokio::test]
    async fn already_funded_response_still_defers_to_the_balance() {
        let chain = ScriptedChain::new(vec![5, 5, 1000]);
        let faucet = ScriptedFaucet::new(vec![Err(FaucetError::AlreadyFunded("403".into()))]);
        let got = ensure_funded(
            &chain,
            Some(&faucet),
            [0u8; 32],
            &params(Some("u")),
            no_sleep,
        )
        .await
        .unwrap();
        assert_eq!(got, 1000);
    }

    #[tokio::test]
    async fn permanent_rejection_fails_without_retrying() {
        let chain = ScriptedChain::new(vec![5]);
        let faucet = ScriptedFaucet::new(vec![Err(FaucetError::Permanent("400 bad dest".into()))]);
        let err = ensure_funded(
            &chain,
            Some(&faucet),
            [0u8; 32],
            &params(Some("u")),
            no_sleep,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, FundingError::FaucetRejected(_)), "{err:?}");
        assert_eq!(faucet.calls(), 1, "a 400 must not be retried");
    }

    #[tokio::test]
    async fn gives_up_after_the_timeout_budget() {
        let chain = ScriptedChain::new(vec![5]); // never funds
        let faucet = ScriptedFaucet::new(vec![Ok(())]);
        let err = ensure_funded(
            &chain,
            Some(&faucet),
            [0u8; 32],
            &params(Some("u")),
            no_sleep,
        )
        .await
        .unwrap_err();
        match err {
            FundingError::FaucetExhausted {
                balance, threshold, ..
            } => {
                assert_eq!((balance, threshold), (5, 100));
            }
            other => panic!("expected exhaustion, got {other:?}"),
        }
        // 60s budget against 2,4,8,16,30... backoff must terminate, not spin.
        assert!(
            faucet.calls() >= 2 && faucet.calls() < 20,
            "calls={}",
            faucet.calls()
        );
    }

    #[tokio::test]
    async fn chain_read_failure_is_reported_not_retried_forever() {
        let err = ensure_funded(
            &FailingChain,
            None::<&ScriptedFaucet>,
            [0u8; 32],
            &params(None),
            no_sleep,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, FundingError::Chain(_)), "{err:?}");
    }

    /// Chain that fails its first `n` reads, then answers. This is the
    /// co-start case: the coordinator and its validator boot together.
    struct SlowStartChain {
        fail_first: usize,
        calls: Mutex<usize>,
    }

    #[async_trait]
    impl BalanceSource for SlowStartChain {
        async fn free_balance(&self, _a: [u8; 32]) -> Result<u128, String> {
            let mut n = self.calls.lock().unwrap();
            *n += 1;
            if *n <= self.fail_first {
                Err("connection refused".into())
            } else {
                Ok(1000)
            }
        }
    }

    /// A validator that is still booting must not kill startup. Exiting here
    /// would crash-loop the container against its own validator.
    #[tokio::test]
    async fn unreachable_chain_is_retried_within_the_budget() {
        let chain = SlowStartChain {
            fail_first: 3,
            calls: Mutex::new(0),
        };
        let faucet = ScriptedFaucet::new(vec![Ok(())]);
        let got = ensure_funded(
            &chain,
            Some(&faucet),
            [0u8; 32],
            &params(Some("u")),
            no_sleep,
        )
        .await
        .unwrap();
        assert_eq!(got, 1000);
        // The faucet is never asked while the balance is unknown.
        assert_eq!(faucet.calls(), 0);
    }

    /// Underfunded with no faucet must fail immediately, not burn the budget:
    /// retrying cannot conjure funds.
    #[tokio::test]
    async fn underfunded_without_faucet_does_not_retry() {
        let chain = ScriptedChain::new(vec![5]);
        let _ = ensure_funded(
            &chain,
            None::<&ScriptedFaucet>,
            [0u8; 32],
            &params(None),
            no_sleep,
        )
        .await
        .unwrap_err();
        assert_eq!(chain.calls(), 1, "must not retry a hopeless case");
    }

    #[test]
    fn status_classification_matches_the_v0_2_contract() {
        assert!(classify_status(200, "").is_ok());
        assert!(classify_status(204, "").is_ok());
        // 400 is the only permanent rejection; everything else is worth a retry.
        assert!(matches!(
            classify_status(400, "bad dest"),
            Err(FaucetError::Permanent(_))
        ));
        assert!(matches!(
            classify_status(403, "capped"),
            Err(FaucetError::AlreadyFunded(_))
        ));
        for s in [429u16, 500, 502, 503, 418] {
            assert!(
                matches!(classify_status(s, ""), Err(FaucetError::Transient(_))),
                "status {s} must be transient"
            );
        }
    }

    #[test]
    fn dest_is_lowercase_hex() {
        assert_eq!(hex_lower(&[0x0a, 0xff]), "0aff");
        assert_eq!(hex_lower(&[0u8; 32]).len(), 64);
    }

    #[test]
    fn faucet_base_url_trailing_slash_is_normalised() {
        let f = HttpFaucet::new("https://faucet.example/").unwrap();
        assert_eq!(f.base_url, "https://faucet.example");
    }

    #[test]
    fn retry_warning_names_a_chain_read_failure_when_balance_is_unknown() {
        assert_eq!(
            retry_warning_message(None),
            "cannot read miner account balance from the chain; retrying"
        );
        assert_eq!(
            retry_warning_message(Some(5)),
            "miner account is not funded yet; retrying"
        );
    }

    #[test]
    fn backoff_doubles_then_saturates() {
        assert_eq!(next_backoff(BACKOFF_START), Duration::from_secs(4));
        assert_eq!(
            next_backoff(Duration::from_secs(16)),
            Duration::from_secs(30)
        );
        assert_eq!(next_backoff(BACKOFF_MAX), BACKOFF_MAX);
    }

    #[tokio::test]
    async fn balance_is_read_once_when_already_funded() {
        let chain = ScriptedChain::new(vec![1000]);
        let _ = ensure_funded(
            &chain,
            None::<&ScriptedFaucet>,
            [0u8; 32],
            &params(None),
            no_sleep,
        )
        .await
        .unwrap();
        assert_eq!(chain.calls(), 1);
    }
}

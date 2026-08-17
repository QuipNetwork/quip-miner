//! Wall-clock timing for anticipatory proof submission (agm.2.3).
//!
//! Tracks the chain's block interval and our network lag from on-chain block
//! timestamps, and converts a target block into a monotonic-clock fire deadline
//! — so the win-time loop can submit slightly ahead and have the extrinsic land
//! at the intended block. Pure + deterministic (clocks are passed in), mirroring
//! the v0.2 `substrate/decay_timing.py`. No chain or async dependency.

/// EMA tracker for block interval + network lag, and block→deadline math. All
/// times in seconds.
#[derive(Debug, Clone)]
pub struct TimingTracker {
    lag_min_s: f64,
    lag_max_s: f64,
    ema_alpha: f64,

    /// Smoothed seconds-per-block (seeded with the fallback interval).
    interval_s: f64,
    /// Smoothed, clamped gap between a block's on-chain timestamp and our local
    /// observation of it — used as the tx-lead.
    lag_s: f64,

    anchor_block: Option<u64>,
    anchor_monotonic_s: f64,
    prev_block: Option<u64>,
    prev_chain_ts_s: f64,
    have_interval: bool,
}

impl TimingTracker {
    /// `fallback_interval_s` seeds `interval_s` until two heads give a real
    /// observation; `lag_min_s`/`lag_max_s` clamp the measured lag; `ema_alpha`
    /// weights each new observation.
    #[must_use]
    pub fn new(fallback_interval_s: f64, lag_min_s: f64, lag_max_s: f64, ema_alpha: f64) -> Self {
        Self {
            lag_min_s,
            lag_max_s,
            ema_alpha,
            interval_s: fallback_interval_s,
            lag_s: 0.0,
            anchor_block: None,
            anchor_monotonic_s: 0.0,
            prev_block: None,
            prev_chain_ts_s: 0.0,
            have_interval: false,
        }
    }

    /// Defaults matching v0.2: 6s fallback interval, lag clamped to [0, 12]s,
    /// EMA alpha 0.3.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(6.0, 0.0, 12.0, 0.3)
    }

    /// Fold one timestamped head into the interval + lag EMAs and the anchor.
    /// `chain_ts_s` is the block's on-chain timestamp; `monotonic_now` /
    /// `wallclock_now` are our local clocks when we observed it.
    pub fn observe_head(
        &mut self,
        block_number: u64,
        chain_ts_s: f64,
        monotonic_now: f64,
        wallclock_now: f64,
    ) {
        if let Some(prev) = self.prev_block {
            if block_number > prev && chain_ts_s > self.prev_chain_ts_s {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "block delta is small; used only as f64 interval divisor"
                )]
                let observed = (chain_ts_s - self.prev_chain_ts_s) / (block_number - prev) as f64;
                if self.have_interval {
                    self.interval_s =
                        self.ema_alpha * observed + (1.0 - self.ema_alpha) * self.interval_s;
                } else {
                    self.interval_s = observed;
                    self.have_interval = true;
                }
            }
        }
        self.prev_block = Some(block_number);
        self.prev_chain_ts_s = chain_ts_s;

        let clamped = (wallclock_now - chain_ts_s).clamp(self.lag_min_s, self.lag_max_s);
        if self.anchor_block.is_none() {
            self.lag_s = clamped;
        } else {
            self.lag_s = self.ema_alpha * clamped + (1.0 - self.ema_alpha) * self.lag_s;
        }

        self.anchor_block = Some(block_number);
        self.anchor_monotonic_s = monotonic_now;
    }

    /// Estimate the current chain block from the monotonic anchor + interval:
    /// `anchor_block + floor((now - anchor_monotonic) / interval)`. `None` until
    /// a head has been observed; never below the anchor block.
    #[must_use]
    pub fn estimate_block(&self, now_monotonic: f64) -> Option<u64> {
        let anchor = self.anchor_block?;
        if self.interval_s <= 0.0 {
            return None;
        }
        let elapsed = (now_monotonic - self.anchor_monotonic_s).max(0.0);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "elapsed/interval is non-negative; block estimate truncates toward zero"
        )]
        {
            Some(anchor + (elapsed / self.interval_s) as u64)
        }
    }

    /// Monotonic deadline to fire for target block `b_star`:
    /// `anchor_monotonic + (b_star - anchor_block) * interval - lag`. `None`
    /// until a head has been observed; may be ≤ now (fire immediately).
    #[must_use]
    pub fn fire_deadline_monotonic(&self, b_star: u64) -> Option<f64> {
        let anchor = self.anchor_block?;
        #[expect(
            clippy::cast_precision_loss,
            reason = "block numbers used as f64 magnitudes for deadline math"
        )]
        let chain_delta = (b_star as f64 - anchor as f64) * self.interval_s;
        Some(self.anchor_monotonic_s + chain_delta - self.lag_s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn no_head_yields_none() {
        let t = TimingTracker::with_defaults();
        assert_eq!(t.estimate_block(100.0), None);
        assert_eq!(t.fire_deadline_monotonic(5), None);
    }

    #[test]
    fn single_head_uses_fallback_interval_and_lag() {
        let mut t = TimingTracker::with_defaults(); // interval 6, lag [0,12], alpha 0.3
                                                    // wallclock - chain_ts = 3 → lag 3 (within clamp).
        t.observe_head(100, 1000.0, 500.0, 1003.0);
        // estimate at monotonic 512 → 100 + floor((512-500)/6) = 102.
        assert_eq!(t.estimate_block(512.0), Some(102));
        // never below the anchor block.
        assert_eq!(t.estimate_block(499.0), Some(100));
        // fire for block 103 → 500 + (103-100)*6 - 3 = 515.
        assert!(approx(t.fire_deadline_monotonic(103).unwrap(), 515.0));
    }

    #[test]
    fn lag_is_clamped_both_ends() {
        let mut hi = TimingTracker::with_defaults();
        hi.observe_head(1, 1000.0, 0.0, 1020.0); // raw lag 20 → clamp 12
        assert!(approx(hi.fire_deadline_monotonic(1).unwrap(), -12.0));
        let mut lo = TimingTracker::with_defaults();
        lo.observe_head(1, 1000.0, 0.0, 995.0); // raw lag -5 → clamp 0
        assert!(approx(lo.fire_deadline_monotonic(1).unwrap(), 0.0));
    }

    #[test]
    fn interval_ema_updates_across_heads() {
        let mut t = TimingTracker::with_defaults();
        t.observe_head(0, 0.0, 0.0, 0.0); // seeds prev; no interval observation yet
        t.observe_head(10, 50.0, 50.0, 50.0); // observed 50/10=5 → interval 5 (first)
        assert!(approx(t.fire_deadline_monotonic(11).unwrap(), 55.0)); // 50 + 1*5 - 0
        t.observe_head(20, 110.0, 110.0, 110.0); // observed 6 → EMA 0.3*6 + 0.7*5 = 5.3
        assert!(approx(t.fire_deadline_monotonic(21).unwrap(), 115.3)); // 110 + 1*5.3 - 0
    }
}

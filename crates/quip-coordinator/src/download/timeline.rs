//! Mining-era timeline: which topology was actually being mined at a given
//! block.
//!
//! The stored per-qblock `topology_hash` is unreliable — the chain's v5
//! `backfill_from_pre_topology` migration overwrote it with the then-current
//! default topology, so qblocks mined against an earlier topology (e.g. a
//! ternary-`h` puzzle) now carry the later h0 hash. The energies prove it: a
//! ternary topology reaches deeper (adds a per-node field term) than an h0
//! topology can, yet those deep qblocks are tagged h0.
//!
//! To re-attribute correctly the `download` command rebuilds the timeline from
//! `RegisteredTopologies` (authoritative) ordered by `registered_at`, and
//! assigns each qblock the era covering its `submitted_at`. A transition-zone
//! guard then reclassifies any qblock whose energy is below its assigned
//! topology's achievable floor into the previous (deeper) era — this catches
//! in-flight proofs for the old topology that land a few blocks after the
//! default switched.

use crate::chain::qblock::{RegisteredTopology, TopologyInputs};

/// One contiguous mining era: `topology_hash` was the default mined topology
/// from `from_block` (inclusive) until the next era's `from_block`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Era {
    /// First block (inclusive) this topology became the mined default. Derived
    /// from the topology's `registered_at`.
    pub from_block: u32,
    /// Consensus topology hash for this era.
    pub topology_hash: [u8; 32],
    /// Redraw inputs (nodes/edges/allowed sets) for this era's topology.
    pub inputs: TopologyInputs,
}

/// Ordered mining-era timeline. Eras are sorted ascending by `from_block`; the
/// era at index `i` covers `[eras[i].from_block, eras[i+1].from_block)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Timeline {
    eras: Vec<Era>,
}

impl Timeline {
    /// Build the timeline from the registered-topology set, ordering eras by
    /// `registered_at`. Ties (two topologies registered in the same block) are
    /// broken by hash for determinism, though the chain does not permit it.
    #[must_use]
    pub fn from_registered(mut topos: Vec<RegisteredTopology>) -> Self {
        topos.sort_by(|a, b| {
            a.registered_at
                .cmp(&b.registered_at)
                .then(a.topology_hash.cmp(&b.topology_hash))
        });
        let eras = topos
            .into_iter()
            .map(|t| Era {
                from_block: t.registered_at,
                topology_hash: t.topology_hash,
                inputs: t.inputs,
            })
            .collect();
        Self { eras }
    }

    /// Number of eras.
    #[must_use]
    pub fn len(&self) -> usize {
        self.eras.len()
    }

    /// True when the timeline has no eras.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.eras.is_empty()
    }

    /// Every era, ascending by `from_block`.
    #[must_use]
    pub fn eras(&self) -> &[Era] {
        &self.eras
    }

    /// Index of the era covering `block`: the last era whose `from_block <=
    /// block`. `None` if `block` precedes the earliest era.
    #[must_use]
    pub fn era_index_at(&self, block: u32) -> Option<usize> {
        let mut found = None;
        for (i, era) in self.eras.iter().enumerate() {
            if era.from_block <= block {
                found = Some(i);
            } else {
                break;
            }
        }
        found
    }

    /// Resolve the era for a qblock, applying the transition-zone energy-floor
    /// guard.
    ///
    /// A qblock is first bound to the era covering `submitted_at`. If that era
    /// has a predecessor and `energy_milli` is strictly below `energy_floor`
    /// (the shallower era's achievable floor), the qblock is reassigned to the
    /// immediately-preceding (deeper) era: only a deeper topology can reach that
    /// energy, so it must be an in-flight proof for the previous topology.
    /// Returns `(era_index, reclassified)`.
    #[must_use]
    pub fn resolve_era(
        &self,
        submitted_at: u32,
        energy_milli: i64,
        energy_floor: i64,
    ) -> Option<(usize, bool)> {
        let i = self.era_index_at(submitted_at)?;
        if i > 0 && energy_milli < energy_floor {
            Some((i - 1, true))
        } else {
            Some((i, false))
        }
    }

    /// Borrow the era at `index`.
    #[must_use]
    pub fn era(&self, index: usize) -> Option<&Era> {
        self.eras.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(nodes: u32) -> TopologyInputs {
        TopologyInputs {
            nodes: (0..nodes).collect(),
            edges: vec![(0, 1)],
            allowed_h_milli: vec![-1000, 0, 1000],
            allowed_j_milli: vec![-1000, 1000],
            allowed_spin_milli: vec![-1000, 1000],
        }
    }

    fn reg(hash: u8, registered_at: u32, nodes: u32) -> RegisteredTopology {
        RegisteredTopology {
            topology_hash: [hash; 32],
            inputs: inputs(nodes),
            registered_at,
        }
    }

    // Two eras: ternary [100, 200), h0 [200, ..). Built out of registration order.
    fn two_era() -> Timeline {
        Timeline::from_registered(vec![reg(0xe6, 200, 4577), reg(0x6e, 100, 4578)])
    }

    #[test]
    fn eras_sorted_by_registration_block() {
        let t = two_era();
        assert_eq!(t.len(), 2);
        let e0 = t.era(0).unwrap();
        let e1 = t.era(1).unwrap();
        assert_eq!(e0.from_block, 100);
        assert_eq!(e0.topology_hash, [0x6e; 32]); // ternary first
        assert_eq!(e1.from_block, 200);
        assert_eq!(e1.topology_hash, [0xe6; 32]); // h0 second
    }

    #[test]
    fn era_index_covers_block_ranges() {
        let t = two_era();
        assert_eq!(t.era_index_at(99), None); // before earliest era
        assert_eq!(t.era_index_at(100), Some(0)); // ternary start (inclusive)
        assert_eq!(t.era_index_at(150), Some(0));
        assert_eq!(t.era_index_at(199), Some(0));
        assert_eq!(t.era_index_at(200), Some(1)); // h0 start (inclusive)
        assert_eq!(t.era_index_at(9999), Some(1));
    }

    #[test]
    fn ternary_era_qblock_keeps_its_era_regardless_of_energy() {
        let t = two_era();
        // Deep energy in the ternary era: earliest era, never reclassified.
        assert_eq!(
            t.resolve_era(150, -15_060_000, -14_650_000),
            Some((0, false))
        );
    }

    #[test]
    fn h0_era_shallow_qblock_stays_in_h0() {
        let t = two_era();
        // Above the floor → genuine h0, no reclassification.
        assert_eq!(
            t.resolve_era(300, -14_540_000, -14_650_000),
            Some((1, false))
        );
    }

    #[test]
    fn h0_era_deep_qblock_reclassified_to_ternary() {
        let t = two_era();
        // In-flight ternary proof that landed after the switch: below the h0
        // floor → bumped to the previous (ternary) era.
        assert_eq!(
            t.resolve_era(205, -14_984_000, -14_650_000),
            Some((0, true))
        );
    }

    #[test]
    fn floor_boundary_is_strict() {
        let t = two_era();
        // Exactly at the floor is NOT below it → stays h0.
        assert_eq!(
            t.resolve_era(300, -14_650_000, -14_650_000),
            Some((1, false))
        );
        // One milli deeper → reclassified.
        assert_eq!(
            t.resolve_era(300, -14_650_001, -14_650_000),
            Some((0, true))
        );
    }

    #[test]
    fn block_before_any_era_is_unassigned() {
        let t = two_era();
        assert_eq!(t.resolve_era(50, -14_000_000, -14_650_000), None);
    }

    #[test]
    fn single_era_never_reclassifies() {
        let t = Timeline::from_registered(vec![reg(0x6e, 100, 4578)]);
        // Deep energy but no predecessor era → stays put.
        assert_eq!(
            t.resolve_era(150, -15_060_000, -14_650_000),
            Some((0, false))
        );
    }
}

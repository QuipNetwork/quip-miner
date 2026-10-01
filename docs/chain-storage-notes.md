# Chain storage notes for win confirmation

<!-- The next paragraph is the plan's Decision B wording, copied in full. -->
<!-- vale off -->
**Decision B:** No per-account win record exists. Task A6 confirms a win by comparing `QuantumPow::LastProofBlock` to the number of the block our extrinsic landed in, and accepts the same-block collision limitation.
<!-- vale on -->

The live chain and the pallet source agree on every item below. Trust both.

## LastProofBlock value type

Task A8 assumes `LastProofBlock` decodes as `u32`. That assumption is correct on the production runtime.

| Source | Finding |
|---|---|
| Pallet | `StorageValue<_, BlockNumberFor<T>, ValueQuery>` in `quantum-pow/src/lib.rs` at line 259 |
| Runtime | `pub type BlockNumber = u32` in `quip-validator/runtime/src/lib.rs` at line 211 |
| Live chain | `state_getStorage` returns four bytes. One read was `0x79371100` (little-endian `u32` 1128313). |
| Coordinator | `real.rs` already decodes the same key as `u32` |

The pallet mock runtime uses a `u64` block number. Do not copy that width. Decode four little-endian bytes.

## QuantumPow storage items

Runtime aliases used in the value column: `AccountId` is `AccountId32`, `BlockNumber` is `u32`, `Balance` is `u128`, `Hash` is `H256`.

Hasher for every map is `Blake2_128Concat`.

| Name | Kind | Key type | Value type | Identifies the account that won |
|---|---|---|---|---|
| `RegisteredTopologies` | StorageMap | `H256` (topology hash) | `TopologyMetaOf` | No |
| `DefaultTopology` | StorageValue | — | `H256` | No |
| `Difficulties` | StorageMap | `H256` (topology hash) | `DifficultyConfig` | No |
| `TopologyCurveC` | StorageMap | `H256` (topology hash) | `CurveC` (`easy_milli`, `knee_milli`, `hard_milli`: `u32`) | No |
| `MineableTopologies` | StorageMap | `H256` (topology hash) | `()` | No |
| `Miners` | StorageMap | `T::AccountId` | `MinerInfo` (`registered_at`, `deposit`, `proofs_submitted`, `proofs_won`, `rewards_earned`) | No. Keyed by account. `proofs_won` is a cumulative count, not a proof record. |
| `BlockBestProof` | StorageValue | — | `ProofRecord` (includes `miner: AccountId`) | Not after the block. `on_finalize` takes this value, so a read at the inclusion hash is empty. |
| `WinnerStreak` | StorageValue | — | `WinnerStreak` (`miner: AccountId`, `count: u32`) | No. Names the current streak holder. The value persists through blocks that have no win. |
| `LastProofBlock` | StorageValue | — | `BlockNumberFor<T>` = `u32` | No. Block number only. |
| `LastProofBlockHash` | StorageValue | — | `H256` | No |
| `BlockProofCount` | StorageValue | — | `u32` | No |
| `QBlocks` | StorageMap | `BlockNumberFor<T>` = `u32` | `QBlock` (first field `miner: AccountId`) | Yes, for that block number. Not keyed by account, so this is not Decision A. |
| `QBlockCount` | StorageValue | — | `u64` | No |
| `QBlockBlockById` | StorageMap | `u64` (qblock id) | `BlockNumberFor<T>` = `u32` | No |
| `QBlockIdByBlock` | StorageMap | `BlockNumberFor<T>` = `u32` | `u64` | No |

`Miners` is the only QuantumPow map keyed by account. Its value is miner stats. It does not store the winning proof.

`QBlocks[block_number].miner` is the account that won that block. `on_finalize` writes it next to the `LastProofBlock` update. A later task can read that field to remove the same-block collision. The Decision A form requires an item keyed by account, so this notes file still records Decision B.

## QuantumComputeMempool storage items

| Name | Kind | Key type | Value type | Identifies the account that won |
|---|---|---|---|---|
| `JobSpecs` | StorageMap | `T::Hash` (`H256`) | `JobSpecOf` | No |
| `JobOrders` | StorageMap | `u64` | `JobOrderOf` | No |
| `NextOrderId` | StorageValue | — | `u64` | No |
| `OpenOrders` | StorageMap | `u64` | `()` | No |
| `OrderSolutions` | StorageDoubleMap | `u64`, then `T::AccountId` | `JobSolutionOf` | No. Job solver, not the PoW winner. |
| `OrderFrontRunner` | StorageMap | `u64` | `FrontRunner` (`solver`, `energy_milli`) | No |
| `OrderTopSolvers` | StorageMap | `u64` | `BoundedVec<RankedSolver, 32>` | No |
| `Solvers` | StorageMap | `T::AccountId` | `SolverInfoOf` | No |
| `ProposerOrders` | StorageMap | `T::AccountId` | `BoundedVec<u64>` | No |
| `OrderResults` | StorageMap | `u64` | `StoredResultOf` | No |

Hasher for every map and both double-map sides is `Blake2_128Concat`.

**Task A4 check: it holds.** `JobOrders` is `StorageMap<_, Blake2_128Concat, u64, JobOrderOf<T>>` at `quantum-compute-mempool/src/lib.rs` line 266. Live keys are 56 bytes: 32-byte map prefix, 16-byte Blake2_128, then eight little-endian bytes for the `u64` id. Sample tails decode as 19, 27, and 26.

## Pallet error enums

Declaration order is the SCALE index. Task A7 must use this order, not the guessed tables in the plan.

### QuantumPow (`quantum-pow/src/lib.rs` lines 355–409)

| Index | Variant |
|---:|---|
| 0 | `MinerAlreadyRegistered` |
| 1 | `MinerNotRegistered` |
| 2 | `TopologyAlreadyRegistered` |
| 3 | `TopologyNotRegistered` |
| 4 | `InvalidCurve` |
| 5 | `GraphTooSmall` |
| 6 | `InvalidTopology` |
| 7 | `ProofLimitReached` |
| 8 | `InvalidNonce` |
| 9 | `NoSolutionsSubmitted` |
| 10 | `InvalidSpinValues` |
| 11 | `SolutionLengthMismatch` |
| 12 | `InsufficientEnergy` |
| 13 | `InsufficientDiversity` |
| 14 | `InsufficientSolutions` |
| 15 | `ArithmeticOverflow` |
| 16 | `EmptyAllowedValues` |
| 17 | `EncodingTooWide` |
| 18 | `PackedSolutionLengthMismatch` |
| 19 | `InvalidEncodedSpin` |
| 20 | `PackedSolutionTooLarge` |
| 21 | `TopologyNotMineable` |
| 22 | `TopologyIsDefault` |
| 23 | `MineableTopologyConflict` |
| 24 | `InvalidDiversityConfig` |

### MinerRegistry (`miner-registry/src/lib.rs` lines 494–518)

| Index | Variant |
|---:|---|
| 0 | `EmptyNodeId` |
| 1 | `EmptyNodeName` |
| 2 | `EmptyPublicHost` |
| 3 | `EmptyRpcEndpoint` |
| 4 | `EmptyMinerLabel` |
| 5 | `EmptyMinerBackend` |
| 6 | `EmptyMinerDeviceId` |
| 7 | `EmptyOsSystem` |
| 8 | `EmptyCpuBrand` |
| 9 | `EmptyCpuArch` |
| 10 | `EmptyGpuVendor` |
| 11 | `EmptyGpuName` |
| 12 | `InvalidGpuUtilization` |
| 13 | `EmptyPythonVersion` |
| 14 | `EmptyQuipVersion` |
| 15 | `EmptyDockerImage` |
| 16 | `NoMiners` |
| 17 | `InvalidPort` |
| 18 | `DescriptorNotFound` |
| 19 | `DescriptorRequired` |
| 20 | `InvalidQBlockId` |
| 21 | `DuplicateParticipation` |

### Names already matched in `classify_receipt` and `classify_participation`

`crates/quip-coordinator/src/chain/submit.rs` matches these strings today.

`classify_receipt` (`submit.rs` lines 126–134):

| String | Pallet variant |
|---|---|
| `InsufficientEnergy` | QuantumPow index 12. Same name. |
| `ProofLimitReached` | QuantumPow index 7. Same name. |
| `InvalidNonce` | QuantumPow index 8. Same name. |
| `TopologyNotRegistered` | QuantumPow index 3. Same name. |
| `InvalidTopology` | QuantumPow index 6. Same name. |
| `InsufficientSolutions` | QuantumPow index 14. Same name. |
| `InsufficientDiversity` | QuantumPow index 13. Same name. |
| `MinerNotRegistered` | QuantumPow index 1. Same name. Not a MinerRegistry variant. |
| `BadSignature` | No such variant on QuantumPow or MinerRegistry. |
| `BadProof` | No such variant on QuantumPow or MinerRegistry. |

`classify_participation` (`submit.rs` lines 109–117):

| String | Pallet variant |
|---|---|
| `DuplicateParticipation` | MinerRegistry index 21. Same name. |
| `InvalidQBlockId` | MinerRegistry index 20. Same name. |
| `DescriptorRequired` | MinerRegistry index 19. Same name. |

The plan's A7 guess tables are wrong. They start at `InsufficientEnergy` and `DuplicateParticipation`, and they invent `BadProof` and `BadSignature`. Replace those tables with the two lists in this section.

## Live chain check

Pallet prefixes (`twox128` of the pallet name):

| Pallet | Prefix |
|---|---|
| `QuantumPow` | `0x9b2c4dbe49d7a1aed7ce99e4b8c072e8` |
| `QuantumComputeMempool` | `0xcbfd51865888632eb84e8a6a17f30b4a` |

`state_getKeysPaged` on the QuantumPow pallet prefix returns `QBlocks` keys first. Each key is 52 bytes: 32-byte item prefix `0x3655cc34…a2d4`, 16-byte Blake2_128, then four little-endian bytes for the `u32` block number. That matches `QBlocks` keyed by `BlockNumberFor<T>`.

A `StorageValue` key is exactly the 32-byte item prefix. `state_getKeysPaged` treats `startKey` as exclusive, so a page that starts at that prefix does not return the value key. Read StorageValue items with `state_getStorage` on the 32-byte key.

The test miner `0xb4e65b8ce157ce9ec3aa818920e7b81b04a23fdce38cf2374eee037d4320da7a` is a `Miners` key (80-byte Blake2_128Concat account key). That account is not the tail of any other QuantumPow map. `QBlocks[LastProofBlock].miner` on the live head was a different account, which matches `WinnerStreak.miner`.

No live key disagreed with the pallet source.

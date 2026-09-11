# Versioning & release-tag standard

This is the canonical versioning standard for every Quip repository that
publishes release tags, container images, or native binaries consumed by
`quip-node-manager` — currently **quip-miner**, **quip-validator**, and
**dashboard.quip.network**. Apply it identically in each.

## The rule

| Artifact | Format | Example | Why |
|----------|--------|---------|-----|
| **Git release tag** | SemVer, **hyphenated** pre-release: `vMAJOR.MINOR.PATCH-rcN` | `v0.2.1-rc17` | quip-node-manager parses this |
| Stable git tag | `vMAJOR.MINOR.PATCH` | `v0.2.1` | — |

This repository ships no Python package. The `quip-solver-core` wheel versions
from its own repository.

## Why the hyphen matters (don't drop it)

`quip-node-manager` decides whether to offer a native-binary update by parsing
the **git release tag** with a SemVer comparator
(`src-tauri/src/update.rs::parse_semver`). That parser splits the pre-release on
`-`:

- `v0.2.1-rc17` → `(0, 2, 1, 17)` ✅ — ordered correctly; `rc17 > rc5`.
- `v0.2.1rc17`  → `(0, 2, 0, MAX)` ❌ — the `.1` patch and the `rc17` are
  lost, so *every* no-hyphen rc collapses to the same value. rc5 == rc16 == a
  final 0.2.0, and the updater never offers a newer rc. (This is exactly the bug
  that froze deployed Macs on an old rc.)

We standardize the **tag** to the form node-manager already understands rather
than changing the comparator, so any other SemVer consumer also works.

## CI tag rules

Each repo's `.gitlab-ci.yml` `.shared_image_rules` matches the hyphenated
pre-release first; the legacy no-hyphen form still builds (so old tags don't
break) but is deprecated:

```yaml
# STANDARD: SemVer hyphenated pre-release — node-manager-parseable.
- if: $CI_COMMIT_TAG =~ /^v[0-9]+\.[0-9]+\.[0-9]+-(rc|alpha|beta)[0-9]*$/
  variables: { MUTABLE_TAGS: "v0.2 $CI_COMMIT_TAG" }
# LEGACY no-hyphen — builds but node-manager can't parse it. Do not use.
- if: $CI_COMMIT_TAG =~ /^v[0-9]+\.[0-9]+\.[0-9]+(rc|alpha|beta)[0-9]*$/
  variables: { MUTABLE_TAGS: "v0.2 $CI_COMMIT_TAG" }
```

Pre-release and `v0.2`-branch builds roll the `:v0.2` rolling image tag and pin
`:<tag>`; they **never** move `:latest` (only `main` / a stable `vX.Y.Z` tag
does). `quip-node-manager` tracks `:v0.2` for images and the hyphenated git tag
for the native binary.

## Cutting a release

1. Bump the crate versions in `crates/*/Cargo.toml` and commit the bump.
2. Tag with the **hyphenated** form and push:
   ```bash
   git tag -a v0.2.1-rc17 -m "v0.2.1-rc17"
   git push origin v0.2 && git push origin v0.2.1-rc17
   ```
3. CI builds the image (`:v0.2` + `:v0.2.1-rc17`) and, for quip-miner, the
   `quip-miner-*` native binaries attached to the GitLab release. The release
   marker node-manager records is the git tag, so the hyphenated form flows
   through to the update check.

> Legacy `v0.2.1rcN` (no hyphen) tags already published remain valid images but
> are invisible to node-manager's binary updater. The first hyphenated tag a
> node-manager instance can see (numerically greater than its installed rc) is
> what unsticks it.

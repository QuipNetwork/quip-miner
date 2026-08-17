#!/usr/bin/env bash
# Fetch the miner binaries a host (or image) needs from the standalone miner
# repos' GitLab package registries, into a local bin dir the coordinator's
# config.toml points at. Miners now live in their own repos and publish
# per-platform binaries as generic-package assets; the coordinator no longer
# builds them.
#
# Single source of truth: the container images (docker/Dockerfile.quip-miner*)
# and the native/host install both call this script instead of duplicating the
# fetch logic.
#
# Usage: fetch-miners.sh [DEST_DIR]           (default: ./miners)
#        fetch-miners.sh --print-tag MINER    (cpu|cuda|metal|dwave)
# Env:
#   CPU_MINERS_TAG, CUDA_MINERS_TAG, METAL_MINERS_TAG, DWAVE_MINERS_TAG
#               pin one miner repo to a release tag instead of taking its
#               newest. One variable per repo because the miners version
#               independently. Leave them unset for the normal case.
#   MINER_SET   which miners to fetch:
#                 auto      (default) derive from this host's OS/arch —
#                           Linux -> cpu (+cuda if amd64, optional);
#                           Darwin/arm64 -> metal + cpu.
#                 cpu       cpu miners only (any arch).
#                 cpu-cuda  cpu + cuda miners (cuda required; amd64 only).
#   DRY_RUN     if set, print the assets that would be fetched and exit 0
#               without downloading (used by the CI verify job).
#   TARGETARCH  explicit target arch (amd64|arm64) for docker builds; defaults
#               to `uname -m` for the native/host install.
#   TARGETOS    explicit target OS (linux|darwin); defaults to `uname -s`.
#
# Assets are published as "<name>-<os>-<arch>" but saved locally as "<name>"
# (no suffix) to match config.toml's `binary = "quip-cpu-sa"` references. The
# OS is part of the name because arch alone is ambiguous: the cpu miner ships
# both a linux-arm64 and a darwin-arm64 build.
#
# NOTE: every run resolves tags from the GitLab API, DRY_RUN included, because
# the asset URL it prints contains the tag.
set -euo pipefail

API="https://gitlab.com/api/v4"
GROUP="quip.network"

# Newest release tag for one miner repo, or empty when it has never cut one.
# `per_page=1` asks for the most recent release, and the API orders newest
# first.
latest_tag() {
  curl --fail --silent --show-error \
    "${API}/projects/${GROUP}%2F${1}/releases?per_page=1" |
    sed -n 's/.*"tag_name":"\([^"]*\)".*/\1/p' | head -n 1
}

# Tag for one miner: an explicit env pin when set, otherwise that repo's newest
# release.
#
# Each repo resolves on its own. Never derive these from the coordinator's own
# release tag: quip-miner v0.3.0-rc4 shipped a coordinator fix while every miner
# stood still, and a build that assumed one shared tag looked for miner releases
# that were never cut. Taking each repo's newest keeps that independence and
# stops the pins going stale, which had left the images on cpu v0.3.0-rc3 long
# after that repo had moved on.
miner_tag() {
  local repo var override tag
  case "${1:-}" in
  cpu) repo=quip-miner-cpu var=CPU_MINERS_TAG override="${CPU_MINERS_TAG:-}" ;;
  cuda) repo=quip-miner-cuda var=CUDA_MINERS_TAG override="${CUDA_MINERS_TAG:-}" ;;
  metal) repo=quip-miner-metal var=METAL_MINERS_TAG override="${METAL_MINERS_TAG:-}" ;;
  dwave) repo=quip-miner-dwave var=DWAVE_MINERS_TAG override="${DWAVE_MINERS_TAG:-}" ;;
  *)
    echo "needs a miner: cpu|cuda|metal|dwave" >&2
    return 1
    ;;
  esac
  if [ -n "$override" ]; then
    printf '%s\n' "$override"
    return 0
  fi
  tag="$(latest_tag "$repo")" || true
  if [ -z "$tag" ]; then
    echo "no release found for ${repo}; set ${var} to pin one" >&2
    return 1
  fi
  printf '%s\n' "$tag"
}

# `--print-tag <miner>` reports one tag and exits, so a Dockerfile can install
# the dwave miner at the same version this script would fetch.
if [ "${1:-}" = "--print-tag" ]; then
  miner_tag "${2:-}"
  exit 0
fi

DEST="${1:-./miners}"
MINER_SET="${MINER_SET:-auto}"

# Arch: honor an explicit TARGETARCH (the docker cross-arch build signal the CI
# passes per image) so a build is never mis-detected; fall back to `uname -m`
# for the native/host install.
raw_arch="${TARGETARCH:-$(uname -m)}"
case "$raw_arch" in
x86_64 | amd64) arch=amd64 ;;
arm64 | aarch64) arch=arm64 ;;
*)
  echo "unsupported arch: $raw_arch" >&2
  exit 1
  ;;
esac
# OS: same treatment as the arch above. `uname -s` is already correct inside a
# docker build, but TARGETOS is honoured for symmetry so a caller preparing a
# build for another platform is never mis-detected.
raw_os="${TARGETOS:-$(uname -s)}"
case "$raw_os" in
Linux | linux) os_tag=linux ;;
Darwin | darwin) os_tag=darwin ;;
*)
  echo "unsupported OS: $raw_os" >&2
  exit 1
  ;;
esac

mkdir -p "$DEST"

# fetch TAG PROJ NAME...  — download "<NAME>-<os>-<arch>" from PROJ's generic
# package registry at TAG, saving it as "<NAME>" (suffix stripped) in DEST.
# PROJ is the URL-encoded "group/repo". TAG is per call because the miners
# version independently. DRY_RUN prints the plan and skips the download.
fetch() {
  local tag="$1" proj="$2"
  shift 2
  local name asset url
  for name in "$@"; do
    asset="${name}-${os_tag}-${arch}"
    url="${API}/projects/${proj}/packages/generic/${proj##*%2F}/${tag}/${asset}"
    if [ -n "${DRY_RUN:-}" ]; then
      echo "would fetch: ${asset} -> ${DEST}/${name}  <- ${url}"
      continue
    fi
    echo "fetch ${asset} -> ${DEST}/${name}"
    curl --fail --location --silent --show-error --output "${DEST}/${name}" "$url"
    chmod +x "${DEST}/${name}" || true
  done
}

# Decide which backends this run wants.
want_cpu=false
want_cuda=false
want_metal=false
cuda_required=false
case "$MINER_SET" in
cpu) want_cpu=true ;;
cpu-cuda)
  want_cpu=true
  want_cuda=true
  cuda_required=true
  ;;
auto)
  # `os_tag` is already validated above, so there is no catch-all arm here.
  case "$os_tag" in
  linux)
    want_cpu=true
    [ "$arch" = amd64 ] && want_cuda=true
    ;;
  darwin)
    want_metal=true
    want_cpu=true
    ;;
  esac
  ;;
*)
  echo "unknown MINER_SET '$MINER_SET' (want: auto|cpu|cpu-cuda)" >&2
  exit 1
  ;;
esac

# Resolve a tag only for a miner this run actually wants, so a repo that is not
# needed can never fail the run.
if [ "$want_metal" = true ]; then
  METAL_TAG="$(miner_tag metal)"
  fetch "$METAL_TAG" "quip.network%2Fquip-miner-metal" "quip-metal-sa" "quip-metal-gibbs"
fi
if [ "$want_cpu" = true ]; then
  CPU_TAG="$(miner_tag cpu)"
  fetch "$CPU_TAG" "quip.network%2Fquip-miner-cpu" "quip-cpu-sa" "quip-cpu-gibbs"
fi
if [ "$want_cuda" = true ]; then
  if [ "$arch" != amd64 ]; then
    echo "cuda miners are amd64-only; skipping on ${arch}" >&2
    if [ "$cuda_required" = true ]; then
      echo "MINER_SET=cpu-cuda requires amd64" >&2
      exit 1
    fi
  elif [ "$cuda_required" = true ]; then
    CUDA_TAG="$(miner_tag cuda)"
    fetch "$CUDA_TAG" "quip.network%2Fquip-miner-cuda" "quip-cuda-sa" "quip-cuda-gibbs"
  else
    CUDA_TAG="$(miner_tag cuda)"
    fetch "$CUDA_TAG" "quip.network%2Fquip-miner-cuda" "quip-cuda-sa" "quip-cuda-gibbs" ||
      echo "note: cuda binaries optional (no GPU host)"
  fi
fi

# The dwave miner ships as a frozen executable on macOS and as a Python package
# everywhere else. The coordinator spawns it as `binary = "quip-dwave-qa"`
# either way, so on darwin it is fetched like any other miner. On linux there
# is no published binary yet, so the container images keep installing from git.
DWAVE_TAG="$(miner_tag dwave)"
if [ "$os_tag" = darwin ]; then
  fetch "$DWAVE_TAG" "quip.network%2Fquip-miner-dwave" "quip-dwave-qa"
else
  echo "dwave: pip install 'quip-miner-dwave @ git+https://gitlab.com/${GROUP}/quip-miner-dwave.git@${DWAVE_TAG}'"
fi
echo "done -> $DEST"

#!/usr/bin/env bash
# Entrypoint for the quip-miner v0.3 image: run the quip-coordinator, which
# connects to the chain and supervises the bundled miner binaries per config.
set -euo pipefail

DATA_DIR="/data"
CONFIG_FILE="${DATA_DIR}/config.toml"
TEMPLATE_FILE="/app/config.toml"
KEYSTORE_FILE="${QUIP_SIGNER_KEY:-${DATA_DIR}/keystore.json}"

mkdir -p "$DATA_DIR"

# Seed the persistent config from the image template on first run; after that
# the mounted /data/config.toml wins.
if [ ! -f "$CONFIG_FILE" ]; then
    cp "$TEMPLATE_FILE" "$CONFIG_FILE"
    echo "entrypoint: seeded $CONFIG_FILE from image template"
fi

# First run: generate the signing keystore, as the v0.2 entrypoint did. Without
# a usable signer the coordinator still mines, but signs with a zero account and
# every proof it submits fails verification on chain. Mount your own keystore at
# $QUIP_SIGNER_KEY to manage the key yourself; this step then does nothing.
if [ ! -f "$KEYSTORE_FILE" ]; then
    echo "entrypoint: first run, generating hybrid keystore at $KEYSTORE_FILE"
    quip-coordinator keygen --out "$KEYSTORE_FILE"
else
    echo "entrypoint: using existing keystore at $KEYSTORE_FILE"
fi

# Ownership for the drop-to-quip below.
chown -R quip:quip "$DATA_DIR" || true
# keygen writes 0600, but the chown above ran as root; re-assert the mode so the
# private key is never group- or world-readable on a remounted volume.
[ -f "$KEYSTORE_FILE" ] && chmod 600 "$KEYSTORE_FILE"

echo "entrypoint: exec quip-coordinator --config $CONFIG_FILE"
# gosu drops root -> quip; tini (PID 1) reaps children the coordinator spawns.
exec gosu quip:quip quip-coordinator --config "$CONFIG_FILE" "$@"

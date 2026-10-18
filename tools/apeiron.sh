#!/bin/bash
# Builds a frozen Apeiron release as an external opponent (GPL-3.0; never linked or copied into
# our code) and installs it as ~/engines/apeiron-<tag>/sprt with its sha256. Run inside WSL.
# Usage: bash tools/apeiron.sh v7.10.0
set -e
TAG=${1:?usage: apeiron.sh <tag>}
SRC=~/src/apeiron-$TAG
DST=~/engines/apeiron-$TAG
source ~/.cargo/env

if [ ! -d "$SRC" ]; then
  git clone -q https://github.com/FirePlank/infinite-chess-engine.git "$SRC"
  git -C "$SRC" checkout -q "$TAG"
fi
# The repo's rust-toolchain.toml pins the nightly; rustup installs it on first use.
(cd "$SRC" && cargo build --release --features sprt --bin sprt && cargo build --release --features data_gen --bin data_gen)

mkdir -p "$DST"
cp "$SRC/target/release/sprt" "$SRC/target/release/data_gen" "$DST/"
git -C "$SRC" rev-parse HEAD > "$DST/COMMIT"
(cd "$DST" && sha256sum sprt data_gen | tee SHA256)

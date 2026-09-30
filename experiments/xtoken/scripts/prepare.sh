#!/usr/bin/env bash
set -euo pipefail
: "${XTOKEN_REPO:?use xtoken.py}"
test ! -e "$XTOKEN_REPO"
git init "$XTOKEN_REPO"
git -C "$XTOKEN_REPO" remote add origin "$1"
git -C "$XTOKEN_REPO" fetch --depth 1 origin "$2"
git -C "$XTOKEN_REPO" checkout --detach FETCH_HEAD
git -C "$XTOKEN_REPO" submodule update --init --recursive --depth 1

#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace${PYTHONPATH:+:${PYTHONPATH}}"
export QUAKE_AI_DEVICE="${QUAKE_AI_DEVICE:-gpu}"
export QUAKE_AI_ARTIFACT_ROOT="${QUAKE_AI_ARTIFACT_ROOT:-/artifacts}"

home_dir="${HOME:-/home/trainer}"
cache_root="${home_dir}/.cache"
config_root="${home_dir}/.config"
if ! mkdir -p "${cache_root}" "${config_root}" 2>/dev/null || [ ! -w "${cache_root}" ] || [ ! -w "${config_root}" ]; then
  export HOME=/tmp/quake-ai-home
  cache_root="${HOME}/.cache"
  config_root="${HOME}/.config"
  mkdir -p "${cache_root}" "${config_root}"
fi

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${cache_root}/pip}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-${config_root}/miopen}"
mkdir -p "${PIP_CACHE_DIR}" "${MIOPEN_USER_DB_PATH}"

mkdir -p "${QUAKE_AI_ARTIFACT_ROOT}/runs"
mkdir -p "${QUAKE_AI_ARTIFACT_ROOT}/corpus"

cd /workspace
exec "$@"

#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace${PYTHONPATH:+:${PYTHONPATH}}"
export QUAKE_AI_DEVICE="${QUAKE_AI_DEVICE:-gpu}"
export QUAKE_AI_ARTIFACT_ROOT="${QUAKE_AI_ARTIFACT_ROOT:-/artifacts}"

mkdir -p "${QUAKE_AI_ARTIFACT_ROOT}/runs"
mkdir -p "${QUAKE_AI_ARTIFACT_ROOT}/corpus"

cd /workspace
exec "$@"

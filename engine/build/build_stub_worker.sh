#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
OUTPUT_PATH=${1:-"${ENGINE_DIR}/build/native_stub_worker"}

if ! command -v cc >/dev/null 2>&1; then
  echo "cc is required to build the native stub worker" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"
cc -O2 -std=c99 -Wall -Wextra -o "${OUTPUT_PATH}" "${ENGINE_DIR}/native_stub.c"
printf '%s\n' "${OUTPUT_PATH}"

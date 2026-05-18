#!/usr/bin/env bash
# Build script for the standalone QWD classifier binary.
# Pure byte-stream parser — no engine, no recast, no upstream Quake source.
# Mirrors src/demo/classify.py:_classify_qw line-for-line.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
REPO_ROOT=$(cd "${SRC_DIR}/.." && pwd)

OUTPUT_PATH=${1:-"${REPO_ROOT}/assets/bin/qw_classifier"}
SOURCE="${SRC_DIR}/demo/qw_classifier.c"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

cc -O2 -Wall -Wextra -D_GNU_SOURCE \
   -I "${SRC_DIR}/engine/common" \
   -o "${OUTPUT_PATH}" \
   "${SOURCE}" -lm

echo "${OUTPUT_PATH}"

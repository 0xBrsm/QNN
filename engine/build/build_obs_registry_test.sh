#!/usr/bin/env bash
# Build + run the obs-registry Gate-1 + WS2 protocol test
# (src/engine/tests/qnn_obs_registry_test.c): default-plan bit parity
# against the pre-refactor packer, compile-error paths, the restored
# atlas (72, unpacked) constructor, the demand-driven compute proof,
# declaration-JSON parse round-trips, the OP_ATTACH_DECL layout reply,
# and the wire-identity shim table.
#
# Standalone: links only qnn_obs_registry.c + qnn_obs_shim.c + the
# test main.  The upstream Quake checkout is needed for headers alone
# (qnn.h includes quakedef.h); no upstream object is compiled or
# linked.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=build_common.sh
source "${SCRIPT_DIR}/build_common.sh"

OUTPUT_PATH=${1:-"${REPO_ROOT}/assets/bin/qnn_obs_registry_test"}

check_build_deps "the obs registry test"

BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/obs-registry-test-build.XXXXXX")
trap 'rm -rf "${BUILD_ROOT}"' EXIT

prepare_upstream "${BUILD_ROOT}" "${COMMON_PATCHES[@]}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"
OBJ_DIR="${BUILD_ROOT}/obj"
mkdir -p "${OBJ_DIR}"
OBJECTS=()

compile_c_strict "${ENGINE_DIR}/common/qnn_obs_registry.c"
compile_c_strict "${ENGINE_DIR}/common/qnn_obs_shim.c"
compile_c_strict "${ENGINE_DIR}/tests/qnn_obs_registry_test.c"

cc -O2 -o "${OUTPUT_PATH}" "${OBJECTS[@]}" -lm

printf '%s\n' "${OUTPUT_PATH}"
"${OUTPUT_PATH}"

#!/usr/bin/env bash
# Build + run the action-look round-trip test
# (src/engine/tests/qnn_action_look_test.c).
#
# Pins QNN_ApplyActionLook as the exact inverse of qnn_collect_main.c's label
# construction: build the label from a known aim change, apply it, and require
# the view angles to land back on the target. The mixed yaw+pitch rows are the
# point — a pure-axis test passes even with the pre-E9 atan2 pitch term, which
# is how that bug survived (a26-superiority-decomposition.md E9).
#
# qnn_input.c is compiled directly; the two upstream helpers it needs
# (QNN_Clamp, anglemod) are defined locally in the test TU, so nothing else
# from the client has to link.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=build_common.sh
source "${SCRIPT_DIR}/build_common.sh"

OUTPUT_PATH=${1:-"${REPO_ROOT}/assets/bin/qnn_action_look_test"}

check_build_deps "the action look test"

BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/action-look-test-build.XXXXXX")
trap 'rm -rf "${BUILD_ROOT}"' EXIT

prepare_upstream "${BUILD_ROOT}" "${COMMON_PATCHES[@]}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"
OBJ_DIR="${BUILD_ROOT}/obj"
mkdir -p "${OBJ_DIR}"
OBJECTS=()

EXTRA_INCLUDES="${EXTRA_INCLUDES:-} -ffunction-sections -fdata-sections"
export EXTRA_INCLUDES

compile_c_strict "${ENGINE_DIR}/nq/qnn_input.c"
compile_c_strict "${ENGINE_DIR}/tests/qnn_action_look_test.c"

cc -O2 -Wl,--gc-sections -o "${OUTPUT_PATH}" "${OBJECTS[@]}" -lm

printf '%s\n' "${OUTPUT_PATH}"
"${OUTPUT_PATH}"

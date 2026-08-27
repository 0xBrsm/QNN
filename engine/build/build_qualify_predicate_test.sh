#!/usr/bin/env bash
# Build + run the entity-qualification predicate test
# (src/engine/tests/qnn_qualify_predicate_test.c).
#
# Pins QNN_QualifyEntity (a26 FULL: modality ladder + recency threshold)
# against QNN_QualifyCombatEntity (a27 COMBAT: exact vis == now) on
# synthetic entities, so "these two agree on the in-LOS case" is a
# mechanical fact rather than an argument.
#
# The predicates are file-static, so the test #includes qnn_oracle.c and
# links the store TU it depends on for the primary-observation helpers.
# Only the predicates are invoked; the emit path is never entered.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=build_common.sh
source "${SCRIPT_DIR}/build_common.sh"

OUTPUT_PATH=${1:-"${REPO_ROOT}/assets/bin/qnn_qualify_predicate_test"}

check_build_deps "the qualify predicate test"

BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/qualify-test-build.XXXXXX")
trap 'rm -rf "${BUILD_ROOT}"' EXIT

prepare_upstream "${BUILD_ROOT}" "${COMMON_PATCHES[@]}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"
OBJ_DIR="${BUILD_ROOT}/obj"
mkdir -p "${OBJ_DIR}"
OBJECTS=()

# Real store TU: it owns QNN_PrimaryObservationTimestamp/ModalityId, which
# decide that an ACTOR's primary observation source is VIS. Stubbing those
# would assume away the very thing under test.
#
# qnn_store.c also holds the whole store-update/trace machinery, which pulls
# in most of the engine. Per-function sections + --gc-sections keep only what
# main actually reaches (the two primary-observation helpers), so nothing has
# to be stubbed and no unrelated engine object is needed.
EXTRA_INCLUDES="${EXTRA_INCLUDES:-} -ffunction-sections -fdata-sections"
export EXTRA_INCLUDES

compile_c_strict "${ENGINE_DIR}/common/qnn_store.c"
compile_c_strict "${ENGINE_DIR}/tests/qnn_qualify_predicate_test.c"

cc -O2 -Wl,--gc-sections -o "${OUTPUT_PATH}" "${OBJECTS[@]}" -lm

printf '%s\n' "${OUTPUT_PATH}"
"${OUTPUT_PATH}"

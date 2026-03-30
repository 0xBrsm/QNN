#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=build_common.sh
source "${SCRIPT_DIR}/build_common.sh"

OUTPUT_PATH=${1:-"${REPO_ROOT}/assets/bin/quake_demo_worker"}

# Demo worker has an extra host.c patch on top of the common patches.
PATCHES=(
  "${COMMON_PATCHES[@]}"
  "${ENGINE_DIR}/patches/host.c.patch"
)

CUSTOM_SOURCES=(
  "${ENGINE_DIR}/worker/qnn_sys.c"
  "${ENGINE_DIR}/worker/qnn_demo_main.c"
  "${ENGINE_DIR}/worker/qnn_input.c"
  "${ENGINE_DIR}/worker/qnn_sound.c"
  "${ENGINE_DIR}/worker/qnn_obs.c"
  "${ENGINE_DIR}/worker/qnn_metrics.c"
  "${ENGINE_DIR}/worker/qnn_world.c"
)

check_build_deps "the Quake demo worker"

BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/quake-demo-worker-build.XXXXXX")
trap 'rm -rf "${BUILD_ROOT}"' EXIT

prepare_upstream "${BUILD_ROOT}" "${PATCHES[@]}"

build_worker "${OUTPUT_PATH}" "${CUSTOM_SOURCES[@]}"

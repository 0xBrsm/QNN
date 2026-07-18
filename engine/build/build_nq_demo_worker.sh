#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=build_common.sh
source "${SCRIPT_DIR}/build_common.sh"

OUTPUT_PATH=${1:-"${REPO_ROOT}/assets/bin/nq_demo_worker"}

# Demo worker has an extra host.c patch on top of the common patches.
PATCHES=(
  "${COMMON_PATCHES[@]}"
  "${ENGINE_DIR}/patches/host.c.patch"
)

CUSTOM_SOURCES=(
  "${ENGINE_DIR}/common/qnn_context.c"
  "${ENGINE_DIR}/nq/qnn_sys.c"
  "${ENGINE_DIR}/common/qnn_sys_common.c"
  "${ENGINE_DIR}/nq/qnn_collect_main.c"
  "${ENGINE_DIR}/common/qnn_collect_helpers.c"
  "${ENGINE_DIR}/nq/qnn_input.c"
  "${ENGINE_DIR}/nq/qnn_predict.c"
  "${ENGINE_DIR}/common/qnn_map.c"
  "${ENGINE_DIR}/common/qnn_entity.c"
  "${ENGINE_DIR}/nq/qnn_players.c"
  "${ENGINE_DIR}/common/qnn_event.c"
  "${ENGINE_DIR}/common/qnn_sound.c"
  "${ENGINE_DIR}/common/qnn_oracle.c"
  "${ENGINE_DIR}/common/qnn_spatial.c"
  "${ENGINE_DIR}/nq/qnn_self.c"
  "${ENGINE_DIR}/common/qnn_self_common.c"
  "${ENGINE_DIR}/nq/qnn_progs_stub.c"
  "${ENGINE_DIR}/common/qnn_io.c"
  "${ENGINE_DIR}/common/qnn_metrics.c"
  "${ENGINE_DIR}/common/qnn_fault.c"
  "${ENGINE_DIR}/common/qnn_watchdog.c"
  "${ENGINE_DIR}/common/qnn_store.c"
  "${ENGINE_DIR}/common/qnn_tick.c"
  "${ENGINE_DIR}/nq/qnn_phys.c"
)

check_build_deps "the Quake demo worker"

BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/quake-demo-worker-build.XXXXXX")
trap 'rm -rf "${BUILD_ROOT}"' EXIT

prepare_upstream "${BUILD_ROOT}" "${PATCHES[@]}"

build_worker "${OUTPUT_PATH}" "${CUSTOM_SOURCES[@]}"

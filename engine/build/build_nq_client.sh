#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=build_common.sh
source "${SCRIPT_DIR}/build_common.sh"

OUTPUT_PATH=${1:-"${REPO_ROOT}/assets/bin/nq_client"}

# Same patches as the trainer/collect builds.
PATCHES=(
  "${COMMON_PATCHES[@]}"
  "${ENGINE_DIR}/patches/host.c.patch"
)

# Client-only source set: drops training (qnn_reward), collect helpers
# (qnn_collect_helpers, qnn_metrics, qnn_watchdog), and demo physics
# (qnn_phys).  Everything that touches sv.* lives in those files; the
# obs/action path that remains is pure cl.* state.
CUSTOM_SOURCES=(
  "${ENGINE_DIR}/nq/qnn_sys.c"
  "${ENGINE_DIR}/common/qnn_sys_common.c"
  "${ENGINE_DIR}/nq/qnn_client_main.c"
  "${ENGINE_DIR}/nq/qnn_input.c"
  "${ENGINE_DIR}/common/qnn_map.c"
  "${ENGINE_DIR}/common/qnn_entity.c"
  "${ENGINE_DIR}/nq/qnn_players.c"
  "${ENGINE_DIR}/common/qnn_event.c"
  "${ENGINE_DIR}/common/qnn_sound.c"
  "${ENGINE_DIR}/common/qnn_oracle.c"
  "${ENGINE_DIR}/common/qnn_spatial.c"
  "${ENGINE_DIR}/nq/qnn_self.c"
  "${ENGINE_DIR}/common/qnn_self_common.c"
  "${ENGINE_DIR}/common/qnn_io.c"
  "${ENGINE_DIR}/common/qnn_fault.c"
  "${ENGINE_DIR}/common/qnn_store.c"
  "${ENGINE_DIR}/common/qnn_tick.c"
)

check_build_deps "the Quake network client"

BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/quake-client-build.XXXXXX")
trap 'rm -rf "${BUILD_ROOT}"' EXIT

prepare_upstream "${BUILD_ROOT}" "${PATCHES[@]}"

build_worker "${OUTPUT_PATH}" "${CUSTOM_SOURCES[@]}"

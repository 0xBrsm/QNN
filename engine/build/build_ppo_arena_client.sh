#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=build_common.sh
source "${SCRIPT_DIR}/build_common.sh"

OUTPUT_PATH=${1:-"${REPO_ROOT}/assets/bin/ppo_arena_client"}
PATCHES=(
  "${COMMON_PATCHES[@]}"
  "${ENGINE_DIR}/patches/host.c.patch"
  "${ENGINE_DIR}/patches/arena/cl_main.c.patch"
  "${ENGINE_DIR}/patches/arena/cl_parse.c.patch"
  "${ENGINE_DIR}/patches/arena/net_main.c.patch"
)
CUSTOM_SOURCES=(
  "${ENGINE_DIR}/nq/qnn_sys.c"
  "${ENGINE_DIR}/common/qnn_sys_common.c"
  "${ENGINE_DIR}/common/qnn_context.c"
  "${ENGINE_DIR}/nq/qnn_arena_client_main.c"
  "${ENGINE_DIR}/nq/qnn_arena_observer.c"
  "${ENGINE_DIR}/nq/qnn_client_runtime_stub.c"
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
  "${ENGINE_DIR}/common/qnn_obs_registry.c"
  "${ENGINE_DIR}/common/qnn_obs_shim.c"
  "${ENGINE_DIR}/common/qnn_metrics.c"
  "${ENGINE_DIR}/common/qnn_fault.c"
  "${ENGINE_DIR}/nq/qnn_reward.c"
  "${ENGINE_DIR}/common/qnn_store.c"
  "${ENGINE_DIR}/common/qnn_tick.c"
)

check_build_deps "the grouped PPO arena policy client"
BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/quake-arena-client-build.XXXXXX")
trap 'rm -rf "${BUILD_ROOT}"' EXIT
prepare_upstream "${BUILD_ROOT}" "${PATCHES[@]}"
build_worker "${OUTPUT_PATH}" "${CUSTOM_SOURCES[@]}"

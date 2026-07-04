#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=build_common.sh
source "${SCRIPT_DIR}/build_common.sh"

OUTPUT_PATH=${1:-"${REPO_ROOT}/assets/bin/nq_client"}

# Same patches as the trainer/collect builds, plus client-only hooks:
#  - host.c.patch
#  - console.c.patch: route the Quake console (Con_Printf) to the terminal;
#    only nq_client wires QNN_ConsoleOutput, so workers keep stock console.c.
#  - cl_parse_rcon.c.patch: route svc_print through QNN_ConsoleRelay for the
#    chat-driven (tell) remote console; client-only for the same reason.
PATCHES=(
  "${COMMON_PATCHES[@]}"
  "${ENGINE_DIR}/patches/host.c.patch"
  "${ENGINE_DIR}/patches/console.c.patch"
  "${ENGINE_DIR}/patches/cl_parse_rcon.c.patch"
)

# Client-only source set: drops training (qnn_reward), collect helpers
# (qnn_collect_helpers, qnn_metrics, qnn_watchdog), and demo physics
# (qnn_phys).  Everything that touches sv.* lives in those files; the
# obs/action path that remains is pure cl.* state.
CUSTOM_SOURCES=(
  "${ENGINE_DIR}/nq/qnn_sys.c"
  "${ENGINE_DIR}/nq/qnn_client_console.c"
  "${ENGINE_DIR}/common/qnn_sys_common.c"
  "${ENGINE_DIR}/nq/qnn_client_main.c"
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
  "${ENGINE_DIR}/common/qnn_fault.c"
  "${ENGINE_DIR}/common/qnn_store.c"
  "${ENGINE_DIR}/common/qnn_tick.c"
  # In-process ONNX inference (ORT C API wrapper + tick_result→tensor
  # packer + sticky-weapon/argmax decode). Replaces the old stdin/stdout
  # protocol to Python.
  "${ENGINE_DIR}/common/qnn_onnx.c"
  # Minimal qnn_runtime definition (the full one lives in qnn_collect_helpers.c
  # which the client doesn't link).
  "${ENGINE_DIR}/nq/qnn_client_runtime_stub.c"
)

check_build_deps "the Quake network client"

# Fetch ORT (header + .so) into vendor/onnxruntime if it's not there yet.
# tools/onnx_smoke/build.sh owns the curl + unpack logic.
ORT_DIR="${REPO_ROOT}/vendor/onnxruntime"
if [[ ! -f "${ORT_DIR}/lib/libonnxruntime.so" ]]; then
  bash "${REPO_ROOT}/tools/onnx_smoke/build.sh" >/dev/null
fi

# Surface ORT to build_common.sh's compile/link.
export EXTRA_INCLUDES="-I${ORT_DIR}/include"
export EXTRA_LINK_FLAGS="-L${ORT_DIR}/lib -lonnxruntime -Wl,-rpath,${ORT_DIR}/lib"

BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/quake-client-build.XXXXXX")
trap 'rm -rf "${BUILD_ROOT}"' EXIT

prepare_upstream "${BUILD_ROOT}" "${PATCHES[@]}"

build_worker "${OUTPUT_PATH}" "${CUSTOM_SOURCES[@]}"

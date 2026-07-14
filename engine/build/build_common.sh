#!/usr/bin/env bash
# Shared build logic sourced by build_ppo_worker.sh and
# build_nq_demo_worker.sh.  Callers must set SCRIPT_DIR before sourcing.

set -euo pipefail

ENGINE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SRC_DIR=$(cd "${ENGINE_DIR}/.." && pwd)
REPO_ROOT=$(cd "${SRC_DIR}/.." && pwd)

UPSTREAM_URL=${QUAKE_UPSTREAM_URL:-"https://github.com/id-Software/Quake.git"}
UPSTREAM_COMMIT=${QUAKE_UPSTREAM_COMMIT:-"bf4ac424ce754894ac8f1dae6a3981954bc9852d"}
QNN_VENDOR_ROOT=${QNN_VENDOR_ROOT:-"${REPO_ROOT}/vendor"}
VENDOR_DIR="${QNN_VENDOR_ROOT}/recastnavigation"

# ── Shared upstream source list ────────────────────────────────────

UPSTREAM_SOURCES=(
  chase.c
  cl_demo.c
  cl_input.c
  cl_main.c
  cl_parse.c
  cl_tent.c
  cmd.c
  common.c
  console.c
  crc.c
  cvar.c
  d_edge.c
  d_fill.c
  d_init.c
  d_modech.c
  d_part.c
  d_polyse.c
  d_scan.c
  d_sky.c
  d_sprite.c
  d_surf.c
  d_vars.c
  d_zpoint.c
  draw.c
  host.c
  host_cmd.c
  keys.c
  mathlib.c
  menu.c
  model.c
  net_bsd.c
  net_dgrm.c
  net_loop.c
  net_main.c
  net_udp.c
  net_vcr.c
  nonintel.c
  pr_cmds.c
  pr_edict.c
  pr_exec.c
  r_aclip.c
  r_alias.c
  r_bsp.c
  r_draw.c
  r_edge.c
  r_efrag.c
  r_light.c
  r_main.c
  r_misc.c
  r_part.c
  r_sky.c
  r_sprite.c
  r_surf.c
  r_vars.c
  sbar.c
  screen.c
  sv_main.c
  sv_move.c
  sv_phys.c
  sv_user.c
  view.c
  wad.c
  world.c
  zone.c
  cd_null.c
  vid_null.c
)

# ── Shared C++ nav sources ─────────────────────────────────────────

CUSTOM_CXX_SOURCES=(
  "${ENGINE_DIR}/common/qnn_navmesh.cpp"
  "${ENGINE_DIR}/common/qnn_link.cpp"
  "${ENGINE_DIR}/common/qnn_cluster.cpp"
  "${ENGINE_DIR}/common/qnn_route.cpp"
)

NAV_CXX_SOURCES=(
  "${VENDOR_DIR}/Recast/Source/Recast.cpp"
  "${VENDOR_DIR}/Recast/Source/RecastAlloc.cpp"
  "${VENDOR_DIR}/Recast/Source/RecastArea.cpp"
  "${VENDOR_DIR}/Recast/Source/RecastAssert.cpp"
  "${VENDOR_DIR}/Recast/Source/RecastContour.cpp"
  "${VENDOR_DIR}/Recast/Source/RecastFilter.cpp"
  "${VENDOR_DIR}/Recast/Source/RecastLayers.cpp"
  "${VENDOR_DIR}/Recast/Source/RecastMesh.cpp"
  "${VENDOR_DIR}/Recast/Source/RecastMeshDetail.cpp"
  "${VENDOR_DIR}/Recast/Source/RecastRasterization.cpp"
  "${VENDOR_DIR}/Recast/Source/RecastRegion.cpp"
  "${VENDOR_DIR}/Detour/Source/DetourAlloc.cpp"
  "${VENDOR_DIR}/Detour/Source/DetourAssert.cpp"
  "${VENDOR_DIR}/Detour/Source/DetourCommon.cpp"
  "${VENDOR_DIR}/Detour/Source/DetourNavMesh.cpp"
  "${VENDOR_DIR}/Detour/Source/DetourNavMeshBuilder.cpp"
  "${VENDOR_DIR}/Detour/Source/DetourNavMeshQuery.cpp"
  "${VENDOR_DIR}/Detour/Source/DetourNode.cpp"
)

# ── Shared patches ─────────────────────────────────────────────────

COMMON_PATCHES=(
  "${ENGINE_DIR}/patches/cl_parse.c.patch"
  "${ENGINE_DIR}/patches/cmd.c.patch"
  "${ENGINE_DIR}/patches/com_parse.c.patch"
  "${ENGINE_DIR}/patches/common.c.patch"
  "${ENGINE_DIR}/patches/common.h.patch"
  "${ENGINE_DIR}/patches/host_cmd.c.patch"
  "${ENGINE_DIR}/patches/net.h.patch"
  "${ENGINE_DIR}/patches/net_dgrm.c.patch"
  "${ENGINE_DIR}/patches/net_udp.c.patch"
  "${ENGINE_DIR}/patches/pr_edict.c.patch"
  "${ENGINE_DIR}/patches/quakedef.h.patch"
  "${ENGINE_DIR}/patches/r_efrag.c.patch"
  "${ENGINE_DIR}/patches/sv_main.c.patch"
  "${ENGINE_DIR}/patches/world.h.patch"
  "${ENGINE_DIR}/patches/64bit/host_cmd.c.patch"
  "${ENGINE_DIR}/patches/64bit/pr_cmds.c.patch"
  "${ENGINE_DIR}/patches/64bit/sv_main.c.patch"
)

# ── Dependency checks ──────────────────────────────────────────────

require_command() {
  local cmd="$1"
  local purpose="$2"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "${cmd} is required to ${purpose}" >&2
    exit 1
  fi
}

check_build_deps() {
  local worker_label="${1:-the Quake worker}"
  require_command cc "build ${worker_label}"
  require_command c++ "build ${worker_label}"
  require_command git "fetch the pinned Quake source"
  require_command patch "apply the local Quake worker overlays"
  if [[ ! -f "${VENDOR_DIR}/Recast/Include/Recast.h" || ! -f "${VENDOR_DIR}/Detour/Include/DetourNavMesh.h" ]]; then
    echo "Vendored Recast/Detour sources are missing under ${VENDOR_DIR}" >&2
    exit 1
  fi
}

# ── Upstream checkout ──────────────────────────────────────────────

prepare_upstream() {
  local build_root="$1"
  local patches=("${@:2}")

  UPSTREAM_DIR="${build_root}/upstream"
  WORKTREE_DIR="${build_root}/WinQuake"

  if [[ -n "${QUAKE_UPSTREAM_DIR:-}" ]]; then
    mkdir -p "${UPSTREAM_DIR}"
    git -C "${QUAKE_UPSTREAM_DIR}" archive "${UPSTREAM_COMMIT}" | tar -x -C "${UPSTREAM_DIR}"
  else
    git init -q "${UPSTREAM_DIR}"
    git -C "${UPSTREAM_DIR}" remote add origin "${UPSTREAM_URL}"
    git -C "${UPSTREAM_DIR}" fetch -q --depth 1 origin "${UPSTREAM_COMMIT}"
    git -C "${UPSTREAM_DIR}" checkout -q --detach FETCH_HEAD
  fi

  cp -R "${UPSTREAM_DIR}/WinQuake" "${WORKTREE_DIR}"

  local patch_path
  for patch_path in "${patches[@]}"; do
    patch -d "${WORKTREE_DIR}" -p0 < "${patch_path}"
  done
}

# ── Compile helpers ────────────────────────────────────────────────

OBJECTS=()

compile_c() {
  local src="$1"
  local obj="${OBJ_DIR}/$(basename "${src}").o"
  cc \
    -std=gnu89 \
    -O2 \
    -fcommon \
    -w \
    -I"${WORKTREE_DIR}" \
    -I"${ENGINE_DIR}/common" \
    -I"${ENGINE_DIR}/nq" \
    -c "${src}" \
    -o "${obj}"
  OBJECTS+=("${obj}")
}

# Same as compile_c but with implicit-function-declaration promoted to an
# error.  Used for QNN-owned sources only — upstream Quake code predates
# clean prototypes and needs -w to compile at all.  Catches the kind of
# missing-prototype bug that silently broke SV_RecursiveHullCheck on QW.
compile_c_strict() {
  local src="$1"
  local obj="${OBJ_DIR}/$(basename "${src}").o"
  cc \
    -std=gnu89 \
    -O2 \
    -fcommon \
    -w \
    -Werror=implicit-function-declaration \
    -I"${WORKTREE_DIR}" \
    -I"${ENGINE_DIR}/common" \
    -I"${ENGINE_DIR}/nq" \
    ${EXTRA_INCLUDES:-} \
    -c "${src}" \
    -o "${obj}"
  OBJECTS+=("${obj}")
}

compile_cxx() {
  local src="$1"
  local obj="${OBJ_DIR}/$(basename "${src}").o"
  c++ \
    -std=c++17 \
    -O2 \
    -w \
    -I"${WORKTREE_DIR}" \
    -I"${ENGINE_DIR}/common" \
    -I"${ENGINE_DIR}/nq" \
    -I"${VENDOR_DIR}/Recast/Include" \
    -I"${VENDOR_DIR}/Detour/Include" \
    -c "${src}" \
    -o "${obj}"
  OBJECTS+=("${obj}")
}

# ── Main build loop ───────────────────────────────────────────────

build_worker() {
  local output_path="$1"
  shift
  local custom_sources=("$@")

  mkdir -p "$(dirname "${output_path}")"
  OBJ_DIR="${BUILD_ROOT}/obj"
  mkdir -p "${OBJ_DIR}"
  OBJECTS=()

  local source
  for source in "${UPSTREAM_SOURCES[@]}"; do
    compile_c "${WORKTREE_DIR}/${source}"
  done
  for source in "${custom_sources[@]}"; do
    compile_c_strict "${source}"
  done
  for source in "${CUSTOM_CXX_SOURCES[@]}"; do
    compile_cxx "${source}"
  done
  for source in "${NAV_CXX_SOURCES[@]}"; do
    compile_cxx "${source}"
  done

  # -rdynamic exports symbols into the dynamic table so
  # backtrace_symbols_fd() in qnn_fault.c can print function names
  # rather than raw addresses when a worker crashes.
  #
  # EXTRA_LINK_OBJECTS / EXTRA_LINK_FLAGS let individual workers
  # (e.g. nq_client linking libqnn + libonnxruntime) extend the link
  # without forcing every worker to inherit those deps.
  c++ \
    -O2 \
    -w \
    -rdynamic \
    -o "${output_path}" \
    "${OBJECTS[@]}" \
    ${EXTRA_LINK_OBJECTS:-} \
    ${EXTRA_LINK_FLAGS:-} \
    -lm

  printf '%s\n' "${output_path}"
}

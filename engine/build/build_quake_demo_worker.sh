#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SRC_DIR=$(cd "${ENGINE_DIR}/.." && pwd)
REPO_ROOT=$(cd "${SRC_DIR}/.." && pwd)

UPSTREAM_URL=${QUAKE_UPSTREAM_URL:-"https://github.com/id-Software/Quake.git"}
UPSTREAM_COMMIT=${QUAKE_UPSTREAM_COMMIT:-"bf4ac424ce754894ac8f1dae6a3981954bc9852d"}
OUTPUT_PATH=${1:-"${REPO_ROOT}/artifacts/bin/quake_demo_worker"}

PATCHES=(
  "${ENGINE_DIR}/patches/common.c.patch"
  "${ENGINE_DIR}/patches/common-pak-case.patch"
  "${ENGINE_DIR}/patches/com_parse.c.patch"
  "${ENGINE_DIR}/patches/common.h-offsetof.patch"
  "${ENGINE_DIR}/patches/world.h.patch"
  "${ENGINE_DIR}/patches/host.c.patch"
  "${ENGINE_DIR}/patches/host_cmd.c.patch"
  "${ENGINE_DIR}/patches/net.h.patch"
  "${ENGINE_DIR}/patches/net_dgrm.c.patch"
  "${ENGINE_DIR}/patches/net_udp.c.patch"
  "${ENGINE_DIR}/patches/pr_edict.c.patch"
  "${ENGINE_DIR}/patches/sv_main.c.patch"
  "${ENGINE_DIR}/patches/64bit/pr_cmds.c.patch"
  "${ENGINE_DIR}/patches/64bit/host_cmd.c.patch"
  "${ENGINE_DIR}/patches/64bit/sv_main.c.patch"
)

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

CUSTOM_SOURCES=(
  "${ENGINE_DIR}/worker/qnn_worker_common.c"
  "${ENGINE_DIR}/worker/qnn_demo_worker_main.c"
  "${ENGINE_DIR}/worker/qnn_worker_input.c"
  "${ENGINE_DIR}/worker/qnn_worker_sound.c"
  "${ENGINE_DIR}/worker/qnn_worker_token.c"
  "${ENGINE_DIR}/worker/qnn_world_model.c"
)

if ! command -v cc >/dev/null 2>&1; then
  echo "cc is required to build the Quake demo worker" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to fetch the pinned Quake source" >&2
  exit 1
fi

if ! command -v patch >/dev/null 2>&1; then
  echo "patch is required to apply the local Quake worker overlays" >&2
  exit 1
fi

BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/quake-demo-worker-build.XXXXXX")
trap 'rm -rf "${BUILD_ROOT}"' EXIT

UPSTREAM_DIR="${BUILD_ROOT}/upstream"
WORKTREE_DIR="${BUILD_ROOT}/WinQuake"

git init -q "${UPSTREAM_DIR}"
git -C "${UPSTREAM_DIR}" remote add origin "${UPSTREAM_URL}"
git -C "${UPSTREAM_DIR}" fetch -q --depth 1 origin "${UPSTREAM_COMMIT}"
git -C "${UPSTREAM_DIR}" checkout -q --detach FETCH_HEAD

cp -R "${UPSTREAM_DIR}/WinQuake" "${WORKTREE_DIR}"

for patch_path in "${PATCHES[@]}"; do
  patch -d "${WORKTREE_DIR}" -p0 < "${patch_path}"
done

SOURCE_PATHS=()
for source in "${UPSTREAM_SOURCES[@]}"; do
  SOURCE_PATHS+=("${WORKTREE_DIR}/${source}")
done
SOURCE_PATHS+=("${CUSTOM_SOURCES[@]}")

mkdir -p "$(dirname "${OUTPUT_PATH}")"

cc \
  -std=gnu89 \
  -O2 \
  -fcommon \
  -w \
  -I"${WORKTREE_DIR}" \
  -o "${OUTPUT_PATH}" \
  "${SOURCE_PATHS[@]}" \
  -lm

printf '%s\n' "${OUTPUT_PATH}"

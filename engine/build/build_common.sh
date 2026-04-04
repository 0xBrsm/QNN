#!/usr/bin/env bash
# Shared build logic sourced by build_quake_worker.sh and
# build_quake_demo_worker.sh.  Callers must set SCRIPT_DIR before sourcing.

set -euo pipefail

ENGINE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SRC_DIR=$(cd "${ENGINE_DIR}/.." && pwd)
REPO_ROOT=$(cd "${SRC_DIR}/.." && pwd)

UPSTREAM_URL=${QUAKE_UPSTREAM_URL:-"https://github.com/id-Software/Quake.git"}
UPSTREAM_COMMIT=${QUAKE_UPSTREAM_COMMIT:-"bf4ac424ce754894ac8f1dae6a3981954bc9852d"}
VENDOR_DIR="${REPO_ROOT}/vendor/recastnavigation"

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
  "${ENGINE_DIR}/worker/qnn_navmesh.cpp"
  "${ENGINE_DIR}/worker/qnn_link.cpp"
  "${ENGINE_DIR}/worker/qnn_cluster.cpp"
  "${ENGINE_DIR}/worker/qnn_route.cpp"
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
  "${ENGINE_DIR}/patches/common.c.patch"
  "${ENGINE_DIR}/patches/common-pak-case.patch"
  "${ENGINE_DIR}/patches/com_parse.c.patch"
  "${ENGINE_DIR}/patches/common.h-offsetof.patch"
  "${ENGINE_DIR}/patches/world.h.patch"
  "${ENGINE_DIR}/patches/host_cmd.c.patch"
  "${ENGINE_DIR}/patches/net.h.patch"
  "${ENGINE_DIR}/patches/net_dgrm.c.patch"
  "${ENGINE_DIR}/patches/net_udp.c.patch"
  "${ENGINE_DIR}/patches/pr_edict.c.patch"
  "${ENGINE_DIR}/patches/sv_main.c.patch"
  "${ENGINE_DIR}/patches/cl_parse.c.patch"
  "${ENGINE_DIR}/patches/64bit/pr_cmds.c.patch"
  "${ENGINE_DIR}/patches/64bit/host_cmd.c.patch"
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

  git init -q "${UPSTREAM_DIR}"
  git -C "${UPSTREAM_DIR}" remote add origin "${UPSTREAM_URL}"
  git -C "${UPSTREAM_DIR}" fetch -q --depth 1 origin "${UPSTREAM_COMMIT}"
  git -C "${UPSTREAM_DIR}" checkout -q --detach FETCH_HEAD

  cp -R "${UPSTREAM_DIR}/WinQuake" "${WORKTREE_DIR}"

  local patch_path
  for patch_path in "${patches[@]}"; do
    patch -d "${WORKTREE_DIR}" -p0 < "${patch_path}"
  done

  # Shared inline patches applied to all worker builds:

  # 1. Make upstream qboolean C++-safe for the nav/oracle worker sources.
  python3 - "${WORKTREE_DIR}/common.h" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()
pattern = r'typedef enum \{false, true\}\s+qboolean;'
replacement = (
    '#ifdef __cplusplus\n'
    'typedef int qboolean;\n'
    '#else\n'
    'typedef enum {false, true}\t\tqboolean;\n'
    '#endif'
)
text, count = re.subn(pattern, replacement, text, count=1)
if count != 1:
    raise SystemExit("failed to patch qboolean definition in common.h")
path.write_text(text)
PY

  # 2. Increase MAX_OSPATH from 128 to 512 (long demo filenames)
  python3 - "${WORKTREE_DIR}/quakedef.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text()
p.write_text(t.replace("#define\tMAX_OSPATH\t\t128", "#define\tMAX_OSPATH\t\t512"))
PY

  # 3. snprintf in COM_FindFile (bounded path concatenation)
  python3 - "${WORKTREE_DIR}/common.c" <<'PY'
from pathlib import Path; import sys
path = Path(sys.argv[1]); text = path.read_text()
text = text.replace(
    'sprintf (netpath, "%s/%s",search->filename, filename);',
    'snprintf (netpath, MAX_OSPATH, "%s/%s",search->filename, filename);')
text = text.replace(
    'sprintf (cachepath,"%s%s", com_cachedir, netpath);',
    'snprintf (cachepath, MAX_OSPATH, "%s%s", com_cachedir, netpath);')
text = text.replace(
    'sprintf (cachepath,"%s%s", com_cachedir, netpath+2);',
    'snprintf (cachepath, MAX_OSPATH, "%s%s", com_cachedir, netpath+2);')
path.write_text(text)
PY

  # 4. Guard R_AddEfrags against NULL worldmodel (headless demo playback)
  python3 - "${WORKTREE_DIR}/r_efrag.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text()
t = t.replace(
    'if (!ent->model)\n\t\treturn;',
    'if (!ent->model)\n\t\treturn;\n\n\tif (!cl.worldmodel || !cl.worldmodel->nodes)\n\t\treturn;')
p.write_text(t)
PY

  # 5. Non-fatal model precache failures (headless doesn't need all models)
  #    Without this, demos on maps like e4m3 fail because progs/star.mdl
  #    isn't in our PAK files — the engine returns early and worldmodel is NULL.
  python3 - "${WORKTREE_DIR}/cl_parse.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text()
old = '''		cl.model_precache[i] = Mod_ForName (model_precache[i], false);
		if (cl.model_precache[i] == NULL)
		{
			Con_Printf("Model %s not found\\n", model_precache[i]);
			return;
		}'''
new = '''		cl.model_precache[i] = Mod_ForName (model_precache[i], false);
		if (cl.model_precache[i] == NULL)
		{
			Con_Printf("Model %s not found (non-fatal)\\n", model_precache[i]);
		}'''
t = t.replace(old, new)
p.write_text(t)
PY
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
    -I"${ENGINE_DIR}/worker" \
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
    -I"${ENGINE_DIR}/worker" \
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
    compile_c "${source}"
  done
  for source in "${CUSTOM_CXX_SOURCES[@]}"; do
    compile_cxx "${source}"
  done
  for source in "${NAV_CXX_SOURCES[@]}"; do
    compile_cxx "${source}"
  done

  c++ \
    -O2 \
    -w \
    -o "${output_path}" \
    "${OBJECTS[@]}" \
    -lm

  printf '%s\n' "${output_path}"
}

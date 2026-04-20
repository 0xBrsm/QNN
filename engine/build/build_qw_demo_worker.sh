#!/usr/bin/env bash
# Build script for the QuakeWorld demo worker.
#
# Checks out the upstream QW/client/ sources, applies minimal headless
# patches, compiles with the QW-specific worker sources, and links
# against the same navmesh/recast libraries as the NQ demo worker.
#
# The resulting binary speaks the same QOBS protocol as the NQ worker
# so bc_collect.py can consume its output directly.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SRC_DIR=$(cd "${ENGINE_DIR}/.." && pwd)
REPO_ROOT=$(cd "${SRC_DIR}/.." && pwd)

UPSTREAM_URL=${QUAKE_UPSTREAM_URL:-"https://github.com/id-Software/Quake.git"}
UPSTREAM_COMMIT=${QUAKE_UPSTREAM_COMMIT:-"bf4ac424ce754894ac8f1dae6a3981954bc9852d"}
VENDOR_DIR="${REPO_ROOT}/vendor/recastnavigation"

OUTPUT_PATH=${1:-"${REPO_ROOT}/assets/bin/qw_demo_worker"}

# ── QW upstream source list ──────────────────────────────────────
# These are the QW/client/ .c files needed for headless demo playback.
# We use the software renderer stubs (vid_null, cd_null, in_null) and
# our own sound/sys/main implementations.

QW_UPSTREAM_SOURCES=(
  cl_cam.c
  cl_demo.c
  cl_ents.c
  cl_input.c
  cl_main.c
  cl_parse.c
  cl_pred.c
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
  keys.c
  mathlib.c
  md4.c
  menu.c
  model.c
  net_chan.c
  net_udp.c
  nonintel.c
  pmove.c
  pmovetst.c
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
  skin.c
  view.c
  wad.c
  zone.c
  cd_null.c
  vid_null.c
  qnn_engine_compat.c
)

# ── QW worker sources ────────────────────────────────────────────
# QW-specific worker files live in src/engine/qw/.
# Shared qnn_* files are in src/engine/common/.

QW_CUSTOM_SOURCES=(
  "${ENGINE_DIR}/qw/qnn_sys.c"
  "${ENGINE_DIR}/common/qnn_sys_common.c"
  "${ENGINE_DIR}/qw/qnn_collect_main.c"
  "${ENGINE_DIR}/common/qnn_match.c"
  "${ENGINE_DIR}/common/qnn_collect_helpers.c"
  "${ENGINE_DIR}/qw/qnn_self.c"
  "${ENGINE_DIR}/qw/qnn_input.c"
  "${ENGINE_DIR}/common/qnn_event.c"
  "${ENGINE_DIR}/common/qnn_sound.c"
  "${ENGINE_DIR}/common/qnn_map.c"
  "${ENGINE_DIR}/common/qnn_entity.c"
  "${ENGINE_DIR}/common/qnn_oracle.c"
  "${ENGINE_DIR}/common/qnn_spatial.c"
  "${ENGINE_DIR}/common/qnn_io.c"
  "${ENGINE_DIR}/common/qnn_metrics.c"
  "${ENGINE_DIR}/common/qnn_store.c"
  "${ENGINE_DIR}/qw/qnn_phys.c"
  "${ENGINE_DIR}/qw/qnn_stubs.c"
)

# ── C++ nav sources (shared with NQ) ─────────────────────────────

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

# ── Dependency checks ────────────────────────────────────────────

require_command() {
  local cmd="$1"
  local purpose="$2"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "${cmd} is required to ${purpose}" >&2
    exit 1
  fi
}

require_command cc "build the QW demo worker"
require_command c++ "build the QW demo worker"
require_command git "fetch the pinned Quake source"
require_command python3 "apply inline patches"

if [[ ! -f "${VENDOR_DIR}/Recast/Include/Recast.h" || ! -f "${VENDOR_DIR}/Detour/Include/DetourNavMesh.h" ]]; then
  echo "Vendored Recast/Detour sources are missing under ${VENDOR_DIR}" >&2
  exit 1
fi

# ── Upstream checkout (QW/client) ────────────────────────────────

BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/qw-demo-worker-build.XXXXXX")
trap 'rm -rf "${BUILD_ROOT}"' EXIT

UPSTREAM_DIR="${BUILD_ROOT}/upstream"
WORKTREE_DIR="${BUILD_ROOT}/QWClient"

git init -q "${UPSTREAM_DIR}"
git -C "${UPSTREAM_DIR}" remote add origin "${UPSTREAM_URL}"
git -C "${UPSTREAM_DIR}" fetch -q --depth 1 origin "${UPSTREAM_COMMIT}"
git -C "${UPSTREAM_DIR}" checkout -q --detach FETCH_HEAD

cp -R "${UPSTREAM_DIR}/QW/client" "${WORKTREE_DIR}"

# ── Inline patches for headless QW operation ─────────────────────

# 1. Make qboolean C++-safe
python3 - "${WORKTREE_DIR}/common.h" <<'PY'
from pathlib import Path
import re, sys
path = Path(sys.argv[1])
text = path.read_text(errors='surrogateescape')
pattern = r'typedef enum \{false, true\}\s+qboolean;'
replacement = (
    '#ifdef __cplusplus\n'
    'typedef int qboolean;\n'
    '#else\n'
    'typedef enum {false, true}\tqboolean;\n'
    '#endif'
)
text, count = re.subn(pattern, replacement, text, count=1)
if count != 1:
    raise SystemExit("failed to patch qboolean definition in common.h")
path.write_text(text, errors='surrogateescape')
PY

# 2. Increase MAX_OSPATH (128→512) and MAX_MSGLEN (1450→65536).
#    Many QWD files from modern servers (FTE, MVDSV) contain messages
#    larger than vanilla QW's 1450-byte MAX_MSGLEN.  The headless
#    worker only reads demos — no network — so the large buffer is safe.
python3 - "${WORKTREE_DIR}/bothdefs.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
t = t.replace("#define\tMAX_OSPATH\t\t128", "#define\tMAX_OSPATH\t\t512")
t = t.replace("#define\tMAX_MSGLEN\t\t1450", "#define\tMAX_MSGLEN\t\t65536")
p.write_text(t, errors='surrogateescape')
PY

# 2b. Use ephemeral port for UDP socket so parallel workers don't clash.
#     The headless worker only reads demos — it never connects to a real
#     server — but QW's NET_Init still binds a UDP socket.  Changing
#     PORT_CLIENT to PORT_ANY (0) lets the OS assign a unique port per
#     process, enabling parallel collection with --workers 30.
python3 - "${WORKTREE_DIR}/protocol.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
t = t.replace("#define\tPORT_CLIENT\t27001", "#define\tPORT_CLIENT\tPORT_ANY")
p.write_text(t, errors='surrogateescape')
PY

# 3. Disable STRUCT_FROM_LINK pointer arithmetic warning (GCC/64-bit)
python3 - "${WORKTREE_DIR}/common.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
old = '#define\tSTRUCT_FROM_LINK(l,t,m) ((t *)((byte *)l - (int)&(((t *)0)->m)))'
new = '#define\tSTRUCT_FROM_LINK(l,t,m) ((t *)((byte *)l - (size_t)&(((t *)0)->m)))'
p.write_text(t.replace(old, new))
PY

# 4. Add NQ-compatible fields to QW client_state_t so shared worker
#    code (qnn_entity.c, qnn_store.c, qnn_oracle.c, etc.) compiles.
#    Fields added: viewentity, maxclients, items, velocity, onground,
#    inwater, mtime[2].  Also adds scoreboard_t and cl_entities[].
python3 - "${WORKTREE_DIR}/client.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')

# Add NQ-compatible scoreboard_t before client_state_t
scoreboard_compat = '''
/* ── NQ-compatible scoreboard_t for shared worker code ────────── */
typedef struct
{
\tchar\tname[MAX_SCOREBOARDNAME];
\tfloat\tentertime;
\tint\t\tfrags;
\tint\t\tcolors;
} scoreboard_t;
'''

# Insert before "typedef struct\n{\n\tint\t\t\tservercount"
marker = '//\n// the client_state_t structure is wiped completely at every\n// server signon\n//'
t = t.replace(marker, scoreboard_compat + '\n' + marker)

# Add NQ-compatible fields to client_state_t, before the closing brace
old_end = '// all player information\n\tplayer_info_t\tplayers[MAX_CLIENTS];\n} client_state_t;'
new_end = '''// all player information
\tplayer_info_t\tplayers[MAX_CLIENTS];

/* ── NQ-compatible fields for shared QNN worker code ──────────── */
\tint\t\t\tviewentity;\t\t/* = playernum + 1, set by QW worker */
\tint\t\t\tmaxclients;\t\t/* = MAX_CLIENTS */
\tscoreboard_t\t*scores;\t/* points to qnn_scores_compat[] */
\tint\t\t\titems;\t\t\t/* = stats[STAT_ITEMS] */
\tvec3_t\t\tvelocity;\t\t/* = simvel */
\tqboolean\tonground;\t\t/* from playerstate */
\tqboolean\tinwater;\t\t/* approximated */
\tdouble\t\tmtime[2];\t\t/* [0] = cl.time, [1] = prev time */
} client_state_t;'''
t = t.replace(old_end, new_end)

# Add extern cl_entities[] declaration (provide a dummy entity_t array)
t += '''
/* ── NQ-compatible entity array for shared QNN worker code ────── */
#define QW_CL_ENTITIES_SIZE MAX_EDICTS
extern entity_t cl_entities[QW_CL_ENTITIES_SIZE];
'''

p.write_text(t, errors='surrogateescape')
PY

# 4b. Add string_t typedef (QW doesn't have progs — it's server-side only)
python3 - "${WORKTREE_DIR}/common.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
# Add string_t typedef at the end (before final blank lines)
if 'typedef int string_t;' not in t:
    t += '\n/* NQ progs compatibility for shared worker code */\ntypedef int string_t;\n'
p.write_text(t, errors='surrogateescape')
PY

# 4c. Create the cl_entities[] array and scoreboard compat in a new file
cat > "${WORKTREE_DIR}/qnn_engine_compat.c" <<'COMPAT'
/*
 * qnn_engine_compat.c — NQ-compatible globals for the shared QNN
 * worker code.  Generated by build_qw_demo_worker.sh; not checked in.
 *
 * Provides cl_entities[] (entity_t array) and qnn_scores_compat[]
 * (scoreboard_t array) that the shared worker files expect.
 * Refreshed each frame by the QW collect main loop via
 * QNN_SyncEngineCompat().
 */
#include "quakedef.h"

entity_t cl_entities[QW_CL_ENTITIES_SIZE];
scoreboard_t qnn_scores_compat[MAX_CLIENTS];

/* Forward decl — defined in patched cl_parse.c, written each
 * svc_playerinfo message so it always holds the most recent state
 * per player regardless of cl.frames[] circular-buffer aliasing. */
extern player_state_t qnn_mvd_latest_playerstate[MAX_CLIENTS];

/* Called once at startup and after map load to populate cl_entities
 * from QW's cl_baselines[] array.
 *
 * Also sets cl.num_entities — QW never writes this field (only NQ's
 * CL_EntityNum bumps it), but shared QNN code uses it to bound the
 * cl_entities[] iteration in QNN_MapBuildFromBaselines and
 * QNN_EntityClassifyKnown. Without this, items and movers (which only
 * enter qnn_store via the baseline path) are silently skipped. */
void QNN_SyncBaselines(void)
{
    int i;
    int max_baseline = 0;
    for (i = 0; i < MAX_EDICTS; ++i)
    {
        int mi = cl_baselines[i].modelindex;
        cl_entities[i].baseline = cl_baselines[i];
        VectorCopy(cl_baselines[i].origin, cl_entities[i].origin);
        VectorCopy(cl_baselines[i].origin, cl_entities[i].msg_origins[0]);
        VectorCopy(cl_baselines[i].origin, cl_entities[i].msg_origins[1]);
        /* Also seed cl_entities[].model from the precached model. QW
         * packetentities only carry entries for entities that move or
         * change state, so stationary brush movers (closed doors,
         * un-triggered plats) would otherwise leave model == NULL and
         * get skipped by QNN_EntityClassifyKnown. */
        if (mi > 0 && mi < MAX_MODELS)
        {
            cl_entities[i].model = cl.model_precache[mi];
            max_baseline = i;
        }
        else
        {
            cl_entities[i].model = NULL;
        }
    }
    cl.num_entities = max_baseline + 1;
}

/* Called each frame to sync NQ-compat fields from QW state */
void QNN_SyncEngineCompat(void)
{
    int i;

    cl.viewentity = cl.playernum + 1;
    cl.maxclients = MAX_CLIENTS;
    cl.items = cl.stats[STAT_ITEMS];
    VectorCopy(cl.simvel, cl.velocity);
    cl.mtime[0] = cl.time;
    cl.mtime[1] = cl.time - host_frametime;

    /* Sync scoreboard from player_info_t */
    for (i = 0; i < MAX_CLIENTS; ++i)
    {
        Q_strncpy(qnn_scores_compat[i].name, cl.players[i].name, MAX_SCOREBOARDNAME - 1);
        qnn_scores_compat[i].name[MAX_SCOREBOARDNAME - 1] = '\0';
        qnn_scores_compat[i].entertime = cl.players[i].entertime;
        qnn_scores_compat[i].frags = cl.players[i].frags;
        qnn_scores_compat[i].colors = (cl.players[i]._topcolor << 4) | cl.players[i]._bottomcolor;
    }
    cl.scores = qnn_scores_compat;

    /* Sync ground state from playerstate.  In MVD we read from the
     * dedicated latest-state array maintained by the MVD playerinfo
     * parser — cl.frames[] can't be trusted because its circular-buffer
     * slots get reused every UPDATE_BACKUP packets and delta-compressed
     * packets may not update every player each frame. */
    if (cl.playernum >= 0 && cl.playernum < MAX_CLIENTS)
    {
        player_state_t *ps;
        if (cls.mvdplayback)
            ps = &qnn_mvd_latest_playerstate[cl.playernum];
        else if (cl.validsequence > 0)
            ps = &cl.frames[cl.validsequence & UPDATE_MASK].playerstate[cl.playernum];
        else
            ps = NULL;
        if (ps != NULL)
            cl.onground = (ps->onground != -1) ? true : false;
    }
    cl.inwater = false; /* TODO: detect from pmove */
}
COMPAT

# 4c. Add qnn_engine_compat.c to upstream sources list
# (This is handled in the source list below)

# 4d. Add msg_origins and baseline to QW entity_t (NQ has them, QW doesn't)
python3 - "${WORKTREE_DIR}/render.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
# Add msg_origins and baseline after origin in entity_t
old = '\tvec3_t\t\t\t\t\torigin;\n\tvec3_t\t\t\t\t\tangles;'
new = ('\tvec3_t\t\t\t\t\torigin;\n'
       '\tvec3_t\t\t\t\t\tmsg_origins[2];\t/* NQ compat */\n'
       '\tvec3_t\t\t\t\t\tangles;\n'
       '\tentity_state_t\t\t\tbaseline;\t\t/* NQ compat */\n'
       '\tdouble\t\t\t\t\tmsgtime;\t\t/* NQ compat */')
t = t.replace(old, new)
p.write_text(t, errors='surrogateescape')
PY

# 4e. Add NQ-compatible fields to QW common.h that shared worker code needs
python3 - "${WORKTREE_DIR}/bothdefs.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
# Add STAT_FRAGS if not defined (QW comments it out)
if 'STAT_FRAGS' not in t or '//define\tSTAT_FRAGS' in t:
    t = t.replace('//define\tSTAT_FRAGS\t\t\t1', '#define\tSTAT_FRAGS\t\t\t1')
p.write_text(t, errors='surrogateescape')
PY

# 5. Remove #include "winquake.h" from cl_main.c (not available on Linux)
python3 - "${WORKTREE_DIR}/cl_main.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
t = t.replace('#include "winquake.h"\n', '/* #include "winquake.h" — removed for headless build */\n')
# Also stub out the Windows-specific SetWindowText calls
t = t.replace('SetWindowText (mainwindow, "QuakeWorld: disconnected");', '/* SetWindowText removed for headless */')
p.write_text(t, errors='surrogateescape')
PY

# 5. Remove #include "winquake.h" from cl_cam.c
python3 - "${WORKTREE_DIR}/cl_cam.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
t = t.replace('#include "winquake.h"\n', '/* #include "winquake.h" — removed for headless build */\n')
p.write_text(t, errors='surrogateescape')
PY

# 6. Stub Host_WriteConfiguration (writes config.cfg — not needed for demo
#    worker). A whitespace-tolerant regex replace: upstream has trailing tabs
#    inside some blank lines, which a literal str.replace would miss silently.
#    This was the primary cause of mass worker crashes under concurrent
#    collect runs: every worker raced to write assets/<gamedir>/config.cfg.
python3 - "${WORKTREE_DIR}/cl_main.c" <<'PY'
from pathlib import Path; import sys, re
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
pat = re.compile(
    r'void\s+Host_WriteConfiguration\s*\(\s*void\s*\)\s*\{.*?\n\}\n',
    re.DOTALL,
)
new = 'void Host_WriteConfiguration (void)\n{\n\t/* stubbed for headless demo worker */\n}\n'
t, n = pat.subn(new, t, count=1)
assert n == 1, f"Host_WriteConfiguration stub patch did not match (n={n})"
p.write_text(t, errors='surrogateescape')
print("patch 6: Host_WriteConfiguration stubbed")
PY

# 7. Print Host_Error to stderr (same as NQ host.c.patch)
python3 - "${WORKTREE_DIR}/cl_main.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
old = '\tCon_Printf ("Host_Error: %s\\n",string);'
new = '\tCon_Printf ("Host_Error: %s\\n",string);\n\tfprintf(stderr, "Host_Error: %s\\n", string);'
t = t.replace(old, new)
p.write_text(t, errors='surrogateescape')
PY

# 8. Non-fatal model precache failures in cl_parse.c
python3 - "${WORKTREE_DIR}/cl_parse.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
# QW's model precache uses NULL checks differently
old = '''		if (!cl.model_precache[nummodels])
			Host_Error ("Model %s not found", cl.model_name[nummodels]);'''
new = '''		if (!cl.model_precache[nummodels])
			Con_Printf ("Model %s not found (non-fatal)\\n", cl.model_name[nummodels]);'''
t = t.replace(old, new)
p.write_text(t, errors='surrogateescape')
PY

# 9. Guard R_AddEfrags against NULL worldmodel
python3 - "${WORKTREE_DIR}/r_efrag.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
t = t.replace(
    'if (!ent->model)\n\t\treturn;',
    'if (!ent->model)\n\t\treturn;\n\n\tif (!cl.worldmodel || !cl.worldmodel->nodes)\n\t\treturn;')
p.write_text(t, errors='surrogateescape')
PY

# 10. snprintf for safe path concatenation in common.c
python3 - "${WORKTREE_DIR}/common.c" <<'PY'
from pathlib import Path; import sys
path = Path(sys.argv[1]); text = path.read_text(errors='surrogateescape')
text = text.replace(
    'sprintf (netpath, "%s/%s",search->filename, filename);',
    'snprintf (netpath, MAX_OSPATH, "%s/%s",search->filename, filename);')
path.write_text(text, errors='surrogateescape')
PY

# 11. Patch Host_Init to skip VID/Draw/SCR/R/CDAudio/Sbar init for headless
#     We replace the platform-specific init block with just CL_Init.
python3 - "${WORKTREE_DIR}/cl_main.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')

# Replace the big platform-specific init block
old = '''#ifdef __linux__
	IN_Init ();
	CDAudio_Init ();
	VID_Init (host_basepal);
	Draw_Init ();
	SCR_Init ();
	R_Init ();

//	S_Init ();		// S_Init is now done as part of VID. Sigh.

	cls.state = ca_disconnected;
	Sbar_Init ();
	CL_Init ();
#else
	VID_Init (host_basepal);
	Draw_Init ();
	SCR_Init ();
	R_Init ();
//	S_Init ();		// S_Init is now done as part of VID. Sigh.
#ifdef GLQUAKE
	S_Init();
#endif

	cls.state = ca_disconnected;
	CDAudio_Init ();
	Sbar_Init ();
	CL_Init ();
	IN_Init ();
#endif'''

new = '''	/* Headless demo worker: skip VID/Draw/SCR/R/S/CDAudio init */
	VID_Init (host_basepal);
	cls.state = ca_disconnected;
	CL_Init ();'''

t = t.replace(old, new)
p.write_text(t, errors='surrogateescape')
PY

# 12. Stub CL_StopUpload (referenced in CL_Disconnect)
python3 - "${WORKTREE_DIR}/cl_main.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
# Add a stub if not already defined
if 'void CL_StopUpload' not in t:
    # It's declared in cl_parse.c, just add a stub declaration
    pass
p.write_text(t, errors='surrogateescape')
PY

# 13. Stub SCR_UpdateScreen and related rendering calls in Host_Frame
python3 - "${WORKTREE_DIR}/cl_main.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
# Replace SCR_UpdateScreen call
t = t.replace('SCR_UpdateScreen ();', '/* SCR_UpdateScreen (); — headless */')
# Replace S_Update calls
t = t.replace(
    '''	if (cls.state == ca_active)
	{
		S_Update (r_origin, vpn, vright, vup);
		CL_DecayLights ();
	}
	else
		S_Update (vec3_origin, vec3_origin, vec3_origin, vec3_origin);

	CDAudio_Update();''',
    '''	if (cls.state == ca_active)
		CL_DecayLights ();
	/* S_Update, CDAudio_Update — headless */''')
p.write_text(t, errors='surrogateescape')
PY

# 14. Fix M_Menu_Quit_f reference in CL_Quit_f
python3 - "${WORKTREE_DIR}/cl_main.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
t = t.replace(
    '''	if (1 /* key_dest != key_console */ /* && cls.state != ca_dedicated */)
	{
		M_Menu_Quit_f ();
		return;
	}''',
    '''	/* M_Menu_Quit_f removed for headless */''')
p.write_text(t, errors='surrogateescape')
PY

# 15. Remove Windows-specific CL_Windows_f
python3 - "${WORKTREE_DIR}/cl_main.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
t = t.replace('#ifdef _WINDOWS\n#include <windows.h>', '/* Windows section removed for headless */\n#if 0')
p.write_text(t, errors='surrogateescape')
PY

# 16. Provide an empty winquake.h so any residual includes don't fail
cat > "${WORKTREE_DIR}/winquake.h" <<'EOF'
/* winquake.h — empty stub for headless QW demo worker build */
#ifndef WINQUAKE_H
#define WINQUAKE_H
#endif
EOF

# 17. Add NQ-compatible field aliases in entity_state_t and entity_t.
#     NQ: entity_state_t has .skin, QW has .skinnum → add .skin alias
#     entity_t: QW has .skinnum but not .effects (it's in entity_state_t)
python3 - "${WORKTREE_DIR}/protocol.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
# Add skin alias after skinnum in entity_state_t
t = t.replace('} entity_state_t;',
              '} entity_state_t;\n/* NQ compat: shared code uses .skin */\n#define skin skinnum\n')
p.write_text(t, errors='surrogateescape')
PY

# Add effects field to entity_t in render.h (NQ has it, QW doesn't)
python3 - "${WORKTREE_DIR}/render.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
# Check if entity_t already has effects
if '\tint\t\t\t\t\t\teffects;' not in t and 'int\t\t\teffects' not in t:
    # Add effects after skinnum
    t = t.replace('\tint\t\t\t\t\t\tskinnum;\t\t// for Alias models',
                  '\tint\t\t\t\t\t\tskinnum;\t\t// for Alias models\n\tint\t\t\t\t\t\teffects;\t\t// NQ compat')
p.write_text(t, errors='surrogateescape')
PY

# Fix screen.h: block_drawing and scr_skipupdate need extern
python3 - "${WORKTREE_DIR}/screen.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
t = t.replace('qboolean\tscr_skipupdate;', 'extern qboolean\tscr_skipupdate;')
t = t.replace('qboolean\tblock_drawing;', 'extern qboolean\tblock_drawing;')
p.write_text(t, errors='surrogateescape')
PY

# Fix min/max macros for C++ compatibility
python3 - "${WORKTREE_DIR}/quakedef.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
# Wrap min/max definitions to avoid C++ conflicts
old = '#ifndef max\n#define max(a,b) ((a) > (b) ? (a) : (b))\n#define min(a,b) ((a) < (b) ? (a) : (b))\n#endif'
new = '#ifndef __cplusplus\n#ifndef max\n#define max(a,b) ((a) > (b) ? (a) : (b))\n#define min(a,b) ((a) < (b) ? (a) : (b))\n#endif\n#endif'
t = t.replace(old, new)
p.write_text(t, errors='surrogateescape')
PY

# Add trace_t compatibility type to quakedef.h (after all includes)
python3 - "${WORKTREE_DIR}/quakedef.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
if 'trace_t' not in t:
    t += '''
/* NQ server compat — trace_t used by shared worker code */
typedef struct {
    int allsolid;
    int startsolid;
    int inopen, inwater;
    float fraction;
    vec3_t endpos;
    struct { vec3_t normal; float dist; } plane;
    void *ent;
} trace_t;

/* NQ server compat stubs — referenced by shared worker code */
typedef struct edict_s { int dummy; } edict_t;
extern edict_t *sv_player;
typedef struct { edict_t *edicts; int num_edicts; int max_edicts;
    struct model_s *worldmodel; struct model_s *models[256]; } server_t;
extern server_t sv;
typedef struct { float time; void *self; } globalvars_t;
extern globalvars_t *pr_global_struct;
extern char *pr_strings;
'''
p.write_text(t, errors='surrogateescape')
PY

# 17b. Add #include <stdint.h> to common.h (needed for uint8_t etc.)
python3 - "${WORKTREE_DIR}/common.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
if '#include <stdint.h>' not in t:
    t = '#include <stdint.h>\n' + t
p.write_text(t, errors='surrogateescape')
PY

# 18. Fix buildnum.c: 'time' conflicts with <time.h>
python3 - "${WORKTREE_DIR}/buildnum.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
# Just stub the whole file to return a constant build number
t = '''#include "quakedef.h"
int build_number(void) { return 1; }
'''
p.write_text(t, errors='surrogateescape')
PY

# ── MVD (Multi-View Demo) support patches ───────────────────────

# 19. Add MVD dem_* constants and DF_* playerinfo flags to protocol.h
python3 - "${WORKTREE_DIR}/protocol.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
mvd_defs = '''
/* ── MVD demo message types (low 3 bits of type byte) ────────── */
#define dem_cmd         0
#define dem_read        1
#define dem_set         2
#define dem_multiple    3
#define dem_single      4
#define dem_stats       5
#define dem_all         6
#define dem_mask        7

/* ── MVD playerinfo DF_* flags (replace PF_* in MVD mode) ────── */
#define DF_ORIGIN1      (1<<0)
#define DF_ORIGIN2      (1<<1)
#define DF_ORIGIN3      (1<<2)
#define DF_ANGLES1      (1<<3)
#define DF_ANGLES2      (1<<4)
#define DF_ANGLES3      (1<<5)
#define DF_EFFECTS      (1<<6)
#define DF_SKINNUM      (1<<7)
#define DF_DEAD         (1<<8)
#define DF_GIB          (1<<9)
#define DF_WEAPONFRAME  (1<<10)
#define DF_MODEL        (1<<11)
'''
if 'dem_multiple' not in t:
    t += mvd_defs
p.write_text(t, errors='surrogateescape')
PY

# 20. Add MVD fields to client_static_t in client.h
python3 - "${WORKTREE_DIR}/client.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
mvd_fields = '''
\t/* ── MVD playback state ──────────────────────────────────── */
\tqboolean\tmvdplayback;
\tint\t\t\tmvd_lasttype;
\tint\t\t\tmvd_lastto;
\tfloat\t\tdemopackettime;
'''
# Insert before closing brace of client_static_t
marker = '} client_static_t;'
if 'mvdplayback' not in t and marker in t:
    t = t.replace(marker, mvd_fields + marker)
p.write_text(t, errors='surrogateescape')
PY

# 21. Add CL_GetMVDMessage() to cl_demo.c and MVD detection in CL_PlayDemo_f
python3 - "${WORKTREE_DIR}/cl_demo.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')

# 21a. Remove local dem_cmd/dem_read/dem_set defines (now in protocol.h)
t = t.replace('#define dem_cmd\t\t0\n', '/* dem_cmd/dem_read/dem_set now in protocol.h */\n')
t = t.replace('#define dem_read\t1\n', '')
t = t.replace('#define dem_set\t\t2\n', '')

# 21b. Add MVD delegation at top of CL_GetDemoMessage
old_getdemo = 'qboolean CL_GetDemoMessage (void)\n{'
new_getdemo = '''qboolean CL_GetMVDMessage (void);

qboolean CL_GetDemoMessage (void)
{
\tif (cls.mvdplayback)
\t\treturn CL_GetMVDMessage();'''
if 'CL_GetMVDMessage' not in t:
    t = t.replace(old_getdemo, new_getdemo)

# 21c. Add MVD extension detection in CL_PlayDemo_f
#      Find where COM_DefaultExtension is called and add .mvd check
old_play = '\tCOM_DefaultExtension (name, ".qwd");'
new_play = '''\t{
\t\tint namelen = strlen(name);
\t\tcls.mvdplayback = false;
\t\tcls.demopackettime = 0.0f;
\t\tif (namelen > 4 && !Q_strcasecmp(name + namelen - 4, ".mvd"))
\t\t\tcls.mvdplayback = true;
\t\telse
\t\t\tCOM_DefaultExtension (name, ".qwd");
\t}'''
if 'cls.mvdplayback = false' not in t:
    t = t.replace(old_play, new_play)

# 21d. Reset MVD state in CL_StopPlayback
old_stop = 'cls.demoplayback = false;'
new_stop = 'cls.demoplayback = false;\n\tcls.mvdplayback = false;'
if 'cls.mvdplayback = false' not in t:
    t = t.replace(old_stop, new_stop, 1)

# 21e. Append CL_GetMVDMessage function at end of file
mvd_reader = r'''

/* ── MVD message reader ────────────────────────────────────────── */

qboolean CL_GetMVDMessage (void)
{
    int     r, i;
    byte    c, msec_byte;
    float   demotime;

readnext:
    /* Read 1-byte msec delta */
    r = fread(&msec_byte, 1, 1, cls.demofile);
    if (r != 1)
    {
        CL_StopPlayback();
        return 0;
    }
    cls.demopackettime += msec_byte * 0.001f;
    demotime = cls.demopackettime;

    /* Increment netchan sequences on new time frames.
     * CL_ParseClientdata uses incoming_acknowledged to index cl.frames[].
     * Without this, all frames land in slot 0 and the state machine stalls. */
    if (msec_byte > 0)
    {
        cls.netchan.incoming_sequence++;
        cls.netchan.incoming_acknowledged++;
        cls.netchan.last_received = realtime;
    }

    /* Read 1-byte type: low 3 bits = type, upper 5 = player */
    r = fread(&c, 1, 1, cls.demofile);
    if (r != 1)
    {
        CL_StopPlayback();
        return 0;
    }

    cls.mvd_lasttype = c & dem_mask;
    cls.mvd_lastto = (c >> 3) & 0x1f;

    (void)demotime;

    /* Timing: wait until realtime catches up */
    if (!cls.timedemo && cls.state >= ca_onserver)
    {
        if (realtime < demotime)
        {
            fseek(cls.demofile, ftell(cls.demofile) - 2, SEEK_SET);
            cls.demopackettime -= msec_byte * 0.001f;
            return 0;
        }
    }
    else
    {
        realtime = demotime;
    }

    switch (cls.mvd_lasttype)
    {
    case dem_multiple:
        /* Read 4-byte player bitmask */
        r = fread(&i, 4, 1, cls.demofile);
        if (r != 1) { CL_StopPlayback(); return 0; }
        cls.mvd_lastto = LittleLong(i);
        /* Fall through to read message payload */

    case dem_single:
    case dem_stats:
    case dem_all:
    case dem_read:
        /* Read 4-byte message length */
        r = fread(&net_message.cursize, 4, 1, cls.demofile);
        if (r != 1) { CL_StopPlayback(); return 0; }
        net_message.cursize = LittleLong(net_message.cursize);
        if (net_message.cursize > MAX_MSGLEN)
        {
            Con_Printf("MVD demo message > MAX_MSGLEN (%d)\n",
                net_message.cursize);
            CL_StopPlayback();
            return 0;
        }
        r = fread(net_message.data, net_message.cursize, 1, cls.demofile);
        if (r != 1) { CL_StopPlayback(); return 0; }
        break;

    case dem_cmd:
        /* MVD files should not contain dem_cmd — skip if found */
        goto readnext;

    case dem_set:
        r = fread(&i, 4, 1, cls.demofile);
        if (r != 1) { CL_StopPlayback(); return 0; }
        cls.netchan.outgoing_sequence = LittleLong(i);
        r = fread(&i, 4, 1, cls.demofile);
        if (r != 1) { CL_StopPlayback(); return 0; }
        cls.netchan.incoming_sequence = LittleLong(i);
        goto readnext;

    default:
        Con_Printf("Corrupted MVD demo (type %d).\n", cls.mvd_lasttype);
        CL_StopPlayback();
        return 0;
    }

    return 1;
}
'''
if 'CL_GetMVDMessage' not in t or 'qboolean CL_GetMVDMessage (void)\n{' not in t:
    # Only append the function body if the forward decl exists but not the body
    if 'qboolean CL_GetMVDMessage (void)\n{' not in t:
        t += mvd_reader
p.write_text(t, errors='surrogateescape')
PY

# 22. Add MVD playerinfo parsing to cl_ents.c
python3 - "${WORKTREE_DIR}/cl_ents.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')

# 22a. Add MVD dispatch at top of CL_ParsePlayerinfo
old_parse = 'void CL_ParsePlayerinfo (void)\n{'
new_parse = '''void CL_ParsePlayerinfo_MVD (void);

void CL_ParsePlayerinfo (void)
{
\tif (cls.mvdplayback)
\t{
\t\tCL_ParsePlayerinfo_MVD();
\t\treturn;
\t}'''
if 'CL_ParsePlayerinfo_MVD' not in t:
    t = t.replace(old_parse, new_parse)

# 22b. Append MVD playerinfo parser
mvd_playerinfo = r'''

/* Latest-known player state per client, maintained independently of
 * cl.frames[] so the QNN worker has a stable, monotonic source of
 * truth regardless of packet delta patterns or circular-buffer aliasing.
 * Populated by CL_ParsePlayerinfo_MVD every time the server sends an
 * svc_playerinfo for a given player.  Read directly by
 * QNN_SyncEngineCompat / QNN_GetPlayerState during MVD playback. */
player_state_t qnn_mvd_latest_playerstate[MAX_CLIENTS];

/* ── MVD playerinfo parser ─────────────────────────────────────── */

void CL_ParsePlayerinfo_MVD (void)
{
    int             num;
    int             flags;
    player_info_t   *info;
    player_state_t  *state;
    int             parsecountmod;

    parsecountmod = cl.parsecount & UPDATE_MASK;

    num = MSG_ReadByte();
    if (num >= MAX_CLIENTS)
    {
        Con_Printf("CL_ParsePlayerinfo_MVD: bad player %d\n", num);
        return;
    }

    info = &cl.players[num];
    state = &cl.frames[parsecountmod].playerstate[num];

    /* Start from the previous known-good state for this player so
     * unflagged fields carry forward correctly.  Using a dedicated
     * latest-state array avoids the circular-buffer staleness that
     * plagues reads from cl.frames[previous_parsecount] — the
     * MVD protocol is delta-compressed but cl.frames[] slots are
     * reused every UPDATE_BACKUP packets. */
    *state = qnn_mvd_latest_playerstate[num];
    state->messagenum = cl.parsecount;

    flags = MSG_ReadShort();
    state->frame = MSG_ReadByte();

    if (flags & DF_ORIGIN1)
        state->origin[0] = MSG_ReadCoord();
    if (flags & DF_ORIGIN2)
        state->origin[1] = MSG_ReadCoord();
    if (flags & DF_ORIGIN3)
        state->origin[2] = MSG_ReadCoord();

    /* MVD sends angles directly (not inside usercmd_t).
     * Write to both command.angles and viewangles like ezQuake does. */
    if (flags & DF_ANGLES1)
        state->command.angles[0] = MSG_ReadAngle16();
    if (flags & DF_ANGLES2)
        state->command.angles[1] = MSG_ReadAngle16();
    if (flags & DF_ANGLES3)
        state->command.angles[2] = MSG_ReadAngle16();
    VectorCopy(state->command.angles, state->viewangles);

    if (flags & DF_MODEL)
        state->modelindex = MSG_ReadByte();
    else
        state->modelindex = cl_playerindex;

    if (flags & DF_SKINNUM)
        state->skinnum = MSG_ReadByte();

    if (flags & DF_EFFECTS)
        state->effects = MSG_ReadByte();

    if (flags & DF_WEAPONFRAME)
        state->weaponframe = MSG_ReadByte();

    /* MVD does not send velocity — it stays at last known value */

    /* Dead/gib flags */
    if (flags & DF_DEAD)
        state->flags |= PF_DEAD;
    else
        state->flags &= ~PF_DEAD;
    if (flags & DF_GIB)
        state->flags |= PF_GIB;
    else
        state->flags &= ~PF_GIB;

    /* Publish the now-current state as the latest-known for this
     * player — QNN_SyncEngineCompat / QNN_GetPlayerState read from
     * this array during MVD playback. */
    qnn_mvd_latest_playerstate[num] = *state;
}
'''
if 'CL_ParsePlayerinfo_MVD' not in t or 'void CL_ParsePlayerinfo_MVD (void)\n{' not in t:
    if 'void CL_ParsePlayerinfo_MVD (void)\n{' not in t:
        t += mvd_playerinfo
p.write_text(t, errors='surrogateescape')
PY

# 23. Route dem_stats to correct player in cl_parse.c
python3 - "${WORKTREE_DIR}/cl_parse.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
# Find CL_SetStat and add MVD player filtering at the start
old_setstat = 'void CL_SetStat (int stat, int value)\n{'
new_setstat = '''void CL_SetStat (int stat, int value)
{
\t/* MVD: dem_stats targets a specific player. Only apply to
\t * cl.stats[] if it targets our tracked player. */
\tif (cls.mvdplayback && cls.mvd_lasttype == dem_stats
\t\t&& cls.mvd_lastto != cl.playernum)
\t\treturn;'''
if 'mvd_lasttype == dem_stats' not in t:
    t = t.replace(old_setstat, new_setstat)
p.write_text(t, errors='surrogateescape')
PY

# 24. MVD view angle sync in qnn_engine_compat.c
#     When MVD, copy tracked player's angles to cl.viewangles
python3 - "${WORKTREE_DIR}/qnn_engine_compat.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
old_inwater = '    cl.inwater = false; /* TODO: detect from pmove */'
new_inwater = '''    cl.inwater = false; /* TODO: detect from pmove */

    /* MVD: sync cl.viewangles from the tracked player's latest-state
     * array (populated by CL_ParsePlayerinfo_MVD).  Reading directly
     * from qnn_mvd_latest_playerstate avoids cl.frames[] circular-
     * buffer aliasing: it holds exactly the most-recently-decoded
     * state for this player, regardless of which parsecount slot that
     * happened to write into. */
    if (cls.mvdplayback && cl.playernum >= 0 && cl.playernum < MAX_CLIENTS)
        VectorCopy(qnn_mvd_latest_playerstate[cl.playernum].viewangles, cl.viewangles);'''
if 'mvd_ps' not in t:
    t = t.replace(old_inwater, new_inwater)
p.write_text(t, errors='surrogateescape')
PY

# 25. Netchan bypass for demo playback in cl_main.c
#     MVD messages have no netchan header — skip Netchan_Process entirely
#     and just call MSG_BeginReading().
#     QWD dem_read messages DO have an 8-byte netchan header (sequence +
#     ack) but validation is unnecessary during replay and the headless
#     build's sequence state diverges from the recorded traffic.  Strip
#     the header and update sequences manually so cl.frames[] indexing
#     works for QNN_ExtractActionFromUsercmd.
python3 - "${WORKTREE_DIR}/cl_main.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')

# Find the Netchan_Process call in CL_ReadPackets and add demo bypass.
# Upstream pattern: if (!Netchan_Process(&cls.netchan)) continue;
old_netchan = "\t\tif (!Netchan_Process(&cls.netchan))\n\t\t\tcontinue;\t\t// wasn't accepted for some reason"
new_netchan = """\t\tif (cls.demoplayback)
\t\t{
\t\t\tMSG_BeginReading ();
\t\t\tif (!cls.mvdplayback)
\t\t\t{
\t\t\t\t/* QWD dem_read: strip 8-byte netchan header and
\t\t\t\t * update sequences for cl.frames[] indexing. */
\t\t\t\tunsigned seq = (unsigned)MSG_ReadLong ();
\t\t\t\tunsigned ack = (unsigned)MSG_ReadLong ();
\t\t\t\tcls.netchan.incoming_sequence = seq & ~(1u << 31);
\t\t\t\tcls.netchan.incoming_acknowledged = ack & ~(1u << 31);
\t\t\t\tcls.netchan.last_received = realtime;
\t\t\t}
\t\t}
\t\telse
\t\t{
\t\t\tif (!Netchan_Process(&cls.netchan))
\t\t\t\tcontinue;\t\t// wasn't accepted for some reason
\t\t}"""
count = t.count(old_netchan)
print(f'patch 25: netchan match count = {count}', file=sys.stderr)
if count == 0:
    idx = t.find('Netchan_Process')
    if idx >= 0:
        print(f'patch 25: context around Netchan_Process: {repr(t[idx-40:idx+80])}', file=sys.stderr)
if 'QWD dem_read: strip' not in t and 'MVD: no netchan header' not in t:
    t = t.replace(old_netchan, new_netchan, 1)
p.write_text(t, errors='surrogateescape')
PY

# 25a. Hook svc_print and svc_centerprint in cl_parse.c to forward
#      match text to QNN_MatchCheckPrint (same approach as NQ's
#      cl_parse.c.patch).  QW's svc_print reads a print-level byte
#      before the text string; NQ's does not.
python3 - "${WORKTREE_DIR}/cl_parse.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')

# svc_print: capture the MSG_ReadString result
old_print = '\t\t\tCon_Printf ("%s", MSG_ReadString ());\n\t\t\tcon_ormask = 0;'
new_print = """\t\t\t{
\t\t\t\tchar *_qnn_print_text = MSG_ReadString ();
\t\t\t\textern void QNN_MatchCheckPrint(const char *);
\t\t\t\tCon_Printf ("%s", _qnn_print_text);
\t\t\t\tQNN_MatchCheckPrint(_qnn_print_text);
\t\t\t}
\t\t\tcon_ormask = 0;"""

# svc_centerprint: same pattern
old_center = '\t\tcase svc_centerprint:\n\t\t\tSCR_CenterPrint (MSG_ReadString ());'
new_center = """\t\tcase svc_centerprint:
\t\t{
\t\t\tchar *_qnn_cp_text = MSG_ReadString ();
\t\t\textern void QNN_MatchCheckPrint(const char *);
\t\t\tSCR_CenterPrint (_qnn_cp_text);
\t\t\tQNN_MatchCheckPrint(_qnn_cp_text);
\t\t}"""

count_p = t.count(old_print)
count_c = t.count(old_center)
print(f'patch 25a: svc_print match={count_p}, svc_centerprint match={count_c}', file=sys.stderr)
if '_qnn_print_text' not in t:
    t = t.replace(old_print, new_print, 1)
if '_qnn_cp_text' not in t:
    t = t.replace(old_center, new_center, 1)
p.write_text(t, errors='surrogateescape')
PY

# 25b. Support -game argument for QW filesystem search path.
#      QW's COM_InitFilesystem only adds basedir/id1 and basedir/qw.
#      Unlike NQ, it ignores -game on the command line (the gamedir is
#      normally set by svc_serverinfo from the server at connect time).
#      For our headless worker, demos live in arbitrary directories that
#      must be on the search path before playdemo can open them.
python3 - "${WORKTREE_DIR}/common.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')

# After the base search paths (id1 + qw), add a -game handler
old = '\t// any set gamedirs will be freed up to here\n\tcom_base_searchpaths = com_searchpaths;'
new = '''\t// -game <dir>: add a custom game directory to the search path.
\t// QW normally sets gamedir from svc_serverinfo, but the headless
\t// demo worker needs it on the command line.
\ti = COM_CheckParm ("-game");
\tif (i && i < com_argc-1)
\t\tCOM_AddGameDirectory (va("%s/%s", com_basedir, com_argv[i+1]));

\t// any set gamedirs will be freed up to here
\tcom_base_searchpaths = com_searchpaths;'''

count = t.count(old)
print(f'patch 25b: -game handler match count = {count}', file=sys.stderr)
if 'COM_CheckParm ("-game")' not in t:
    t = t.replace(old, new, 1)
p.write_text(t, errors='surrogateescape')
PY

# 26. FTE extension handling + MVD serverdata in cl_parse.c
#     MVD files from modern MVDSV (0.31+) prefix svc_serverdata with FTE
#     extension markers before protocol 28. We must skip these markers.
#     Also, MVD playernum field is a float (demo start time), not a byte.
python3 - "${WORKTREE_DIR}/cl_parse.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')

# 26a. Replace the protocol version check with FTE-aware loop.
# Upstream reads one long and checks against PROTOCOL_VERSION.
# MVD files have FTE/FTE2/MVD1 marker longs before the real protocol 28.
old_protover = '\tprotover = MSG_ReadLong ();\n\tif (protover != PROTOCOL_VERSION && \n\t\t!(cls.demoplayback && (protover == 26 || protover == 27 || protover == 28)))\n\t\tHost_EndGame ("Server returned version %i, not %i\\nYou probably need to upgrade.\\nCheck http://www.quakeworld.net/", protover, PROTOCOL_VERSION);'

new_protover = '''\t/* FTE extension markers: MVDSV 0.31+ prepends extension bitmasks
\t * before protocol 28. Loop reading longs until we hit 28. */
\t{
\t\tunsigned int fte_ext = 0, fte_ext2 = 0, mvd_ext = 0;
\t\tprotover = MSG_ReadLong ();
\t\twhile (protover != PROTOCOL_VERSION)
\t\t{
\t\t\tif (protover == 0x58455446) /* "FTEX" - FTE extensions */
\t\t\t\tfte_ext = (unsigned int)MSG_ReadLong();
\t\t\telse if (protover == 0x32455446) /* "FTE2" */
\t\t\t\tfte_ext2 = (unsigned int)MSG_ReadLong();
\t\t\telse if (protover == 0x3144564D) /* "MVD1" */
\t\t\t\tmvd_ext = (unsigned int)MSG_ReadLong();
\t\t\telse if (cls.demoplayback && (protover == 26 || protover == 27))
\t\t\t\tbreak; /* old demo compat */
\t\t\telse
\t\t\t{
\t\t\t\tHost_EndGame("Server returned version %i, not %i", protover, PROTOCOL_VERSION);
\t\t\t\treturn;
\t\t\t}
\t\t\tprotover = MSG_ReadLong();
\t\t}
\t\t(void)fte_ext; (void)fte_ext2; (void)mvd_ext;
\t}'''

if 'fte_ext' not in t:
    count = t.count(old_protover)
    print(f'patch 26a: protover match count = {count}', file=sys.stderr)
    t = t.replace(old_protover, new_protover, 1)

# 26b. Replace playernum + spectator block for MVD support.
old_playernum = '''\t// parse player slot, high bit means spectator
\tcl.playernum = MSG_ReadByte ();
\tif (cl.playernum & 128)
\t{
\t\tcl.spectator = true;
\t\tcl.playernum &= ~128;
\t}'''
new_playernum = '''\t// parse player slot, high bit means spectator
\tif (cls.mvdplayback)
\t{
\t\t/* MVD: demo start time (float) instead of playernum (byte).
\t\t * Force spectator at slot MAX_CLIENTS-1 per ezQuake convention. */
\t\t{
\t\t\tfloat mvd_start = MSG_ReadFloat();
\t\t\tcls.netchan.last_received = mvd_start;
\t\t}
\t\tcl.playernum = MAX_CLIENTS - 1;
\t\tcl.spectator = true;
\t}
\telse
\t{
\t\tcl.playernum = MSG_ReadByte ();
\t\tif (cl.playernum & 128)
\t\t{
\t\t\tcl.spectator = true;
\t\t\tcl.playernum &= ~128;
\t\t}
\t}'''
if 'mvd_start' not in t:
    count = t.count(old_playernum)
    print(f'patch 26b: playernum match count = {count}', file=sys.stderr)
    t = t.replace(old_playernum, new_playernum, 1)

p.write_text(t, errors='surrogateescape')
PY

# 28. FTE_PEXT_FLOATCOORDS: override MSG_ReadCoord/Angle for float precision.
#     When this extension is active, coords are 4-byte floats instead of
#     2-byte 13-bit fixed-point, and angles change size too.
python3 - "${WORKTREE_DIR}/common.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')

# Add a global flag for float coords
if 'qnn_fte_floatcoords' not in t:
    # Add the global before MSG_ReadCoord
    old_rc = 'float MSG_ReadCoord (void)\n{\n\treturn MSG_ReadShort() * (1.0/8);'
    new_rc = '''int qnn_fte_floatcoords = 0;

float MSG_ReadCoord (void)
{
\tif (qnn_fte_floatcoords)
\t\treturn MSG_ReadFloat();
\treturn MSG_ReadShort() * (1.0/8);'''
    c1 = t.count(old_rc); print(f'patch 28: ReadCoord count = {c1}', file=sys.stderr)
    t = t.replace(old_rc, new_rc, 1)

    # Override MSG_ReadAngle for float coords
    old_ra = 'float MSG_ReadAngle (void)\n{\n\treturn MSG_ReadChar() * (360.0/256);'
    new_ra = '''float MSG_ReadAngle (void)
{
\tif (qnn_fte_floatcoords)
\t\treturn MSG_ReadShort() * (360.0/65536);
\treturn MSG_ReadChar() * (360.0/256);'''
    c2 = t.count(old_ra); print(f'patch 28: ReadAngle count = {c2}', file=sys.stderr)
    t = t.replace(old_ra, new_ra, 1)

    # MSG_ReadAngle16 stays as short — MVDSV writes angle16 as shorts
    # even with FTE_PEXT_FLOATCOORDS. Only MSG_ReadCoord and MSG_ReadAngle
    # change size.

p.write_text(t, errors='surrogateescape')
PY

# 29. Set qnn_fte_floatcoords flag when the extension is detected in serverdata.
python3 - "${WORKTREE_DIR}/cl_parse.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')

# After the FTE extension loop, set the flag if FTE_PEXT_FLOATCOORDS (0x8000)
old_void = '\t\t(void)fte_ext; (void)fte_ext2; (void)mvd_ext;'
new_void = '''\t\t/* Activate float coordinate reads if server uses big coords. */
\t\t{
\t\t\textern int qnn_fte_floatcoords;
\t\t\tqnn_fte_floatcoords = (fte_ext & 0x8000) ? 1 : 0;
\t\t}'''
if 'qnn_fte_floatcoords' not in t:
    t = t.replace(old_void, new_void, 1)

p.write_text(t, errors='surrogateescape')
PY

# 30. Skip sound/model downloads during demo playback.
#     CL_ParseSoundlist calls Sound_NextDownload which tries HTTP downloads
#     that fail and disconnect. For demo playback, all data is in the file.
python3 - "${WORKTREE_DIR}/cl_parse.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')

# Make Sound_NextDownload skip downloads during demo playback
old_snd = 'void Sound_NextDownload (void)\n{'
new_snd = '''void Sound_NextDownload (void)
{
\tif (cls.demoplayback)
\t{
\t\t/* Demo playback: all sounds are in PAK files, skip downloads.
\t\t * Precache them and request modellist to continue init chain. */
\t\tint qi;
\t\tfor (qi = 1; qi < MAX_SOUNDS; qi++)
\t\t{
\t\t\tif (!cl.sound_name[qi][0]) break;
\t\t\tcl.sound_precache[qi] = S_PrecacheSound(cl.sound_name[qi]);
\t\t}
\t\tMSG_WriteByte (&cls.netchan.message, clc_stringcmd);
\t\tMSG_WriteString (&cls.netchan.message, va("modellist %i %i", cl.servercount, 0));
\t\treturn;
\t}'''
if 'Demo playback: all sounds' not in t:
    t = t.replace(old_snd, new_snd, 1)

# Make Model_NextDownload skip downloads during demo playback
old_mdl = 'void Model_NextDownload (void)\n{'
new_mdl = '''void Model_NextDownload (void)
{
\tif (cls.demoplayback)
\t{
\t\t/* Demo playback: skip HTTP downloads. Load models from PAK
\t\t * and proceed to prespawn. */
\t\tint i;
\t\tfor (i = 1; i < MAX_MODELS; i++)
\t\t{
\t\t\tif (!cl.model_name[i][0])
\t\t\t\tbreak;
\t\t\tcl.model_precache[i] = Mod_ForName(cl.model_name[i], false);
\t\t}
\t\tcl.worldmodel = cl.model_precache[1];
\t\tif (!cl.worldmodel)
\t\t{
\t\t\tfprintf(stderr, "[qw-demo] map BSP not found: %s\\n", cl.model_name[1]);
\t\t\tCL_Disconnect();
\t\t\treturn;
\t\t}
\t\t/* Request prespawn to continue init. */
\t\tMSG_WriteByte (&cls.netchan.message, clc_stringcmd);
\t\tMSG_WriteString (&cls.netchan.message, va("prespawn %i 0 %i", cl.servercount, cl.worldmodel->checksum2));
\t\treturn;
\t}'''
if 'Demo playback: skip HTTP' not in t:
    t = t.replace(old_mdl, new_mdl, 1)

p.write_text(t, errors='surrogateescape')
PY

# 30a. Bump MAX_PACKET_ENTITIES from 64 to 256 for MVD demos.
#      Modern MVDSV demos exceed 64 entities per packet.
python3 - "${WORKTREE_DIR}/protocol.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
old = '#define\tMAX_PACKET_ENTITIES\t64'
new = '#define\tMAX_PACKET_ENTITIES\t256'
if 'MAX_PACKET_ENTITIES\t256' not in t:
    t = t.replace(old, new, 1)
p.write_text(t, errors='surrogateescape')
PY

# 30b. Treat svc_bad (0) as end-of-message during MVD playback.
#      MVD messages can have trailing zero padding after the last real svc.
python3 - "${WORKTREE_DIR}/cl_parse.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
old = 'Host_EndGame ("CL_ParseServerMessage: Illegible server message");'
new = '''if (cls.mvdplayback && cmd == svc_bad)
\t\t\t\tbreak; /* MVD trailing zeros — end of message */
\t\t\tHost_EndGame ("CL_ParseServerMessage: Illegible svc 0x%02x at offset %d of %d", cmd, msg_readcount, net_message.cursize);'''
if 'MVD trailing zeros' not in t:
    t = t.replace(old, new, 1)
p.write_text(t, errors='surrogateescape')
PY

echo "==> QW sources patched for headless build (with MVD support)"

# ── Compile ──────────────────────────────────────────────────────

OBJ_DIR="${BUILD_ROOT}/obj"
mkdir -p "${OBJ_DIR}"
mkdir -p "$(dirname "${OUTPUT_PATH}")"
OBJECTS=()

compile_c() {
  local src="$1"
  local obj="${OBJ_DIR}/$(basename "${src}").o"
  cc \
    -std=gnu89 \
    -O2 \
    -fcommon \
    -w \
    -DQNN_QW_BUILD \
    -I"${WORKTREE_DIR}" \
    -I"${ENGINE_DIR}/common" \
    -I"${ENGINE_DIR}/qw" \
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
    -DQNN_QW_BUILD \
    -I"${WORKTREE_DIR}" \
    -I"${ENGINE_DIR}/common" \
    -I"${ENGINE_DIR}/qw" \
    -I"${VENDOR_DIR}/Recast/Include" \
    -I"${VENDOR_DIR}/Detour/Include" \
    -c "${src}" \
    -o "${obj}"
  OBJECTS+=("${obj}")
}

# Compile upstream QW sources
for source in "${QW_UPSTREAM_SOURCES[@]}"; do
  compile_c "${WORKTREE_DIR}/${source}"
done

# Compile QW worker sources
for source in "${QW_CUSTOM_SOURCES[@]}"; do
  compile_c "${source}"
done

# Compile shared C++ nav sources
for source in "${CUSTOM_CXX_SOURCES[@]}"; do
  compile_cxx "${source}"
done
for source in "${NAV_CXX_SOURCES[@]}"; do
  compile_cxx "${source}"
done

# Link
c++ \
  -O2 \
  -w \
  -o "${OUTPUT_PATH}" \
  "${OBJECTS[@]}" \
  -lm

printf '%s\n' "${OUTPUT_PATH}"

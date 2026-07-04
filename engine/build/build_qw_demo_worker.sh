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
  pr_exec.c
  pr_edict.c
  pr_cmds.c
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
  "${ENGINE_DIR}/common/qnn_collect_helpers.c"
  "${ENGINE_DIR}/common/qnn_weapon.c"
  "${ENGINE_DIR}/common/qnn_mvd_collect.c"
  "${ENGINE_DIR}/qw/qnn_qwd_collect.c"
  "${ENGINE_DIR}/qw/qnn_self.c"
  "${ENGINE_DIR}/common/qnn_self_common.c"
  "${ENGINE_DIR}/qw/qnn_input.c"
  "${ENGINE_DIR}/common/qnn_event.c"
  "${ENGINE_DIR}/common/qnn_sound.c"
  "${ENGINE_DIR}/common/qnn_map.c"
  "${ENGINE_DIR}/common/qnn_entity.c"
  "${ENGINE_DIR}/qw/qnn_players.c"
  "${ENGINE_DIR}/common/qnn_oracle.c"
  "${ENGINE_DIR}/common/qnn_spatial.c"
  "${ENGINE_DIR}/common/qnn_io.c"
  "${ENGINE_DIR}/common/qnn_metrics.c"
  "${ENGINE_DIR}/common/qnn_fault.c"
  "${ENGINE_DIR}/common/qnn_watchdog.c"
  "${ENGINE_DIR}/common/qnn_store.c"
  "${ENGINE_DIR}/common/qnn_tick.c"
  "${ENGINE_DIR}/qw/qnn_phys.c"
  "${ENGINE_DIR}/qw/qnn_stubs.c"
  "${ENGINE_DIR}/qw/qnn_progs_stubs.c"
  "${ENGINE_DIR}/qw/qnn_progs.c"
  "${ENGINE_DIR}/qw/qnn_pmove_hooks.c"
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

# Also bring in the QW server-side QC VM sources (pr_exec.c, pr_edict.c,
# pr_cmds.c + their headers).  The worker normally doesn't need them, but
# the optional --sanitize-inputs collect mode loads qwprogs.dat through
# the real VM to evaluate per-tick operative-input predicates exactly as
# the server engine would.
cp "${UPSTREAM_DIR}/QW/server/pr_exec.c"  "${WORKTREE_DIR}/"
cp "${UPSTREAM_DIR}/QW/server/pr_edict.c" "${WORKTREE_DIR}/"
cp "${UPSTREAM_DIR}/QW/server/pr_cmds.c"  "${WORKTREE_DIR}/"
cp "${UPSTREAM_DIR}/QW/server/progs.h"    "${WORKTREE_DIR}/"
cp "${UPSTREAM_DIR}/QW/server/progdefs.h" "${WORKTREE_DIR}/"
cp "${UPSTREAM_DIR}/QW/server/pr_comp.h"  "${WORKTREE_DIR}/"
cp "${UPSTREAM_DIR}/QW/server/qwsvdef.h"  "${WORKTREE_DIR}/"
cp "${UPSTREAM_DIR}/QW/server/server.h"   "${WORKTREE_DIR}/"
cp "${UPSTREAM_DIR}/QW/server/world.h"    "${WORKTREE_DIR}/"

# ── Patch helper ─────────────────────────────────────────────────
#
# Most patches replace a single literal string in one upstream file.
# `apply_subst <file> [guard]` reads OLD on its caller's stdin up to a
# separator line "===NEW===", then NEW until EOF, and substitutes once.
# Pass `guard` to skip when a marker string is already present in the
# target (idempotent rebuilds).  Errors out if OLD doesn't match exactly
# once, so silent upstream drift surfaces as a hard build failure.
APPLY_SUBST_PY='
import sys
from pathlib import Path
path, guard = Path(sys.argv[1]), sys.argv[2]
data = sys.stdin.read()
sep = "\n===NEW===\n"
if sep not in data:
    sys.exit(f"apply_subst: missing ===NEW=== separator (file={path.name})")
old, new = data.split(sep, 1)
if new.endswith("\n"):
    new = new[:-1]
text = path.read_text(errors="surrogateescape")
if guard and guard in text:
    sys.exit(0)
n = text.count(old)
if n != 1:
    sys.exit(f"apply_subst: {path.name}: expected 1 match for old text, got {n}")
path.write_text(text.replace(old, new, 1), errors="surrogateescape")
'
apply_subst() {
	python3 -c "$APPLY_SUBST_PY" "${WORKTREE_DIR}/$1" "${2-}"
}

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

# 2a. Increase the small Z_Malloc zone. The default QW zone is only
#     128 KiB and long-lived demo workers can exhaust it on aliases,
#     cvars, and search-path bookkeeping across sequential demos.
apply_subst zone.c <<'EOF'
#define	DYNAMIC_SIZE	0x20000
===NEW===
#define	DYNAMIC_SIZE	0x800000
EOF

# 2b. Use ephemeral port for UDP socket so parallel workers don't clash.
#     The headless worker only reads demos — it never connects to a real
#     server — but QW's NET_Init still binds a UDP socket.  Changing
#     PORT_CLIENT to PORT_ANY (0) lets the OS assign a unique port per
#     process, enabling parallel collection with --workers 30.
apply_subst protocol.h <<'EOF'
#define	PORT_CLIENT	27001
===NEW===
#define	PORT_CLIENT	PORT_ANY
EOF

# 3. Disable STRUCT_FROM_LINK pointer arithmetic warning (GCC/64-bit)
apply_subst common.h <<'EOF'
#define	STRUCT_FROM_LINK(l,t,m) ((t *)((byte *)l - (int)&(((t *)0)->m)))
===NEW===
#define	STRUCT_FROM_LINK(l,t,m) ((t *)((byte *)l - (size_t)&(((t *)0)->m)))
EOF

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
\tint\t\t\tmaxclients;\t\t/* parsed from cl.serverinfo "maxclients", clamped to MAX_CLIENTS */
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
 * enter qnn_store via the baseline path) are silently skipped.
 *
 * Per-frame visibility is then refreshed by QNN_SyncPacketEntities,
 * which mirrors vanilla NQ's CL_RelinkEntities behavior — entities
 * not in the current packet's PVS list get model = NULL so the
 * shared QNN_EntityClassifyKnown walk drops them naturally. */
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
        /* Seed cl_entities[].model from the precached model so the
         * one-time QNN_MapBuildFromBaselines pass (run at map load)
         * sees every map item / mover.  QNN_SyncPacketEntities then
         * owns model freshness per frame from cl.frames[]. */
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

/* Apply a single entity_state_t snapshot to a cl_entities[] slot.
 * Shared helper between the per-frame packet entity refresh and any
 * future per-state writer.  Updates the NQ-compat fields the shared
 * QNN_EntityClassifyKnown / QNN_StoreUpdate path reads. */
static void QNN_ApplyEntityState(entity_t *ent, const entity_state_t *s,
    struct model_s *mdl, float now)
{
    VectorCopy(ent->msg_origins[0], ent->msg_origins[1]);
    VectorCopy(s->origin, ent->msg_origins[0]);
    VectorCopy(s->origin, ent->origin);
    VectorCopy(s->angles, ent->angles);
    ent->model   = mdl;
    ent->effects = s->effects;
    ent->skinnum = s->skinnum;
    ent->frame   = s->frame;
    ent->msgtime = now;
}

/* Refresh cl_entities[] per frame to match the recording's current
 * PVS universe, mirroring vanilla NQ's CL_RelinkEntities convention:
 *   cl_entities[i].model != NULL  <=>  entity in this frame's PVS.
 *
 * QWD demos are server-PVS-filtered — packet_entities carries the
 * full PVS list each tick (vanilla QW delta-decode rebuilds the full
 * set from prior frame).  Sweep clears model for any non-player slot
 * not refreshed this frame so the shared QNN_EntityClassifyKnown
 * model-NULL check drops everything outside PVS.
 *
 * MVD recordings carry the whole map every frame (no server cull).
 * The geometric QNN_EntityInPvs overlay in qnn_entity.c is the
 * per-viewer filter for those, so we leave baselines live here. */
void QNN_SyncPacketEntities(void)
{
    float now = (float)cl.mtime[0];
    int   i;

    if (cls.mvdplayback)
        return;

    if (cl.validsequence > 0)
    {
        packet_entities_t *pack = &cl.frames[cl.validsequence & UPDATE_MASK].packet_entities;
        for (i = 0; i < pack->num_entities; ++i)
        {
            entity_state_t *s = &pack->entities[i];
            int             entnum = s->number;
            struct model_s *mdl;

            if (entnum <= 0 || entnum >= QW_CL_ENTITIES_SIZE)
                continue;
            /* Slots 1..maxclients are players, owned by qnn_players.c
             * via playerstate.  Don't disturb them. */
            if (entnum >= 1 && entnum <= cl.maxclients)
                continue;
            if (s->modelindex <= 0 || s->modelindex >= MAX_MODELS)
                continue;
            mdl = cl.model_precache[s->modelindex];
            if (mdl == NULL)
                continue;

            QNN_ApplyEntityState(&cl_entities[entnum], s, mdl, now);
        }
    }

    /* Sweep: drop any non-player cl_entity not refreshed this frame.
     * Skip the local viewentity (cl.viewentity) and the player band
     * (1..maxclients).  QNN_EntityClassifyKnown filters on model
     * != NULL so this is the per-frame PVS gate. */
    for (i = cl.maxclients + 1; i < cl.num_entities; ++i)
    {
        if (i == cl.viewentity)
            continue;
        if (cl_entities[i].msgtime != now)
            cl_entities[i].model = NULL;
    }
}

/* Called each frame to sync NQ-compat fields from QW state */
void QNN_SyncEngineCompat(void)
{
    int i;

    cl.viewentity = cl.playernum + 1;
    {
        int parsed = atoi(Info_ValueForKey(cl.serverinfo, "maxclients"));
        if (parsed <= 0 || parsed > MAX_CLIENTS)
            parsed = MAX_CLIENTS;
        cl.maxclients = parsed;
    }
    cl.items = cl.stats[STAT_ITEMS];
    VectorCopy(cl.simvel, cl.velocity);
    /* Do NOT overwrite cl.mtime[0] / cl.mtime[1] here.  Vanilla QW
     * sets cl.mtime[0] from svc_time during cl_parse — this is the
     * server's wall clock when the packet was sent, which we need
     * at sub-emit-frame resolution for the MVD per-event fire
     * back-shift to compute each sound's phase within the emit
     * window.  An earlier version of this patch synced them to
     * cl.time for NQ-compat downstream code, but that clobbered the
     * per-svc-time advance and pinned cl.mtime[0] to the emit
     * boundary.  If downstream NQ code relies on cl.mtime[0] for
     * state interpolation, audit those call sites instead. */

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

# 4f. Add qnn_engine_compat.c to upstream sources list
# (This is handled in the source list below)

# 4d. Add msg_origins and baseline to QW entity_t (NQ has them, QW doesn't)
apply_subst render.h <<'EOF'
	vec3_t					origin;
	vec3_t					angles;
===NEW===
	vec3_t					origin;
	vec3_t					msg_origins[2];	/* NQ compat */
	vec3_t					angles;
	entity_state_t			baseline;		/* NQ compat */
	double					msgtime;		/* NQ compat */
EOF

# 4e. Uncomment STAT_FRAGS (QW ships it commented out; NQ-shared code needs it).
apply_subst bothdefs.h '#define	STAT_FRAGS			1' <<'EOF'
//define	STAT_FRAGS			1
===NEW===
#define	STAT_FRAGS			1
EOF

# 5a. Remove #include "winquake.h" from cl_main.c (not available on Linux)
#     and stub the Windows-specific SetWindowText call.
apply_subst cl_main.c <<'EOF'
#include "winquake.h"
===NEW===
/* #include "winquake.h" — removed for headless build */
EOF
apply_subst cl_main.c <<'EOF'
SetWindowText (mainwindow, "QuakeWorld: disconnected");
===NEW===
/* SetWindowText removed for headless */
EOF

# 5b. Remove #include "winquake.h" from cl_cam.c
apply_subst cl_cam.c <<'EOF'
#include "winquake.h"
===NEW===
/* #include "winquake.h" — removed for headless build */
EOF

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
apply_subst cl_main.c <<'EOF'
	Con_Printf ("Host_Error: %s\n",string);
===NEW===
	Con_Printf ("Host_Error: %s\n",string);
	fprintf(stderr, "Host_Error: %s\n", string);
EOF

# 7b. Centralize the Host_Frame rate gate.  Replaces the cl_maxfps
#     30/72 floor/ceiling clamp with a single call into qnn_tick.c so
#     a single cvar (qnn_tick_hz) controls the engine tick rate across
#     QW demo collect, NQ demo collect, NQ trainer, and NQ live client.
#     When qnn_tick_hz is 0 (unset), the legacy clamped behavior is
#     preserved verbatim by passing it as the native cap.
apply_subst cl_main.c <<'EOF'
	realtime += time;
	if (oldrealtime > realtime)
		oldrealtime = 0;

	if (cl_maxfps.value)
		fps = max(30.0, min(cl_maxfps.value, 72.0));
	else
		fps = max(30.0, min(rate.value/80.0, 72.0));

	if (!cls.timedemo && realtime - oldrealtime < 1.0/fps)
		return;			// framerate is too high
===NEW===
	{
		extern qboolean QNN_TickGate(qboolean is_timedemo,
			float incoming_time, float native_cap_hz,
			double *p_realtime, double *p_oldrealtime);
		float native_cap;
		if (cl_maxfps.value)
			native_cap = max(30.0, min(cl_maxfps.value, 72.0));
		else
			native_cap = max(30.0, min(rate.value/80.0, 72.0));
		if (!QNN_TickGate(cls.timedemo, time, native_cap,
				&realtime, &oldrealtime))
			return;
	}
EOF

# 7c. Remove the now-unused `float fps;` local in Host_Frame.
apply_subst cl_main.c <<'EOF'
	int			pass1, pass2, pass3;
	float fps;
	if (setjmp (host_abort) )
===NEW===
	int			pass1, pass2, pass3;
	if (setjmp (host_abort) )
EOF

# 9. Guard R_AddEfrags against NULL worldmodel
apply_subst r_efrag.c <<'EOF'
if (!ent->model)
		return;
===NEW===
if (!ent->model)
		return;

	if (!cl.worldmodel || !cl.worldmodel->nodes)
		return;
EOF

# 10. snprintf for safe path concatenation in common.c
apply_subst common.c <<'EOF'
sprintf (netpath, "%s/%s",search->filename, filename);
===NEW===
snprintf (netpath, MAX_OSPATH, "%s/%s",search->filename, filename);
EOF

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

# 13. Stub SCR_UpdateScreen call in Host_Frame.  S_Update / CDAudio_Update
#     no-op without sound init (skipped in patch 11), so we leave them.
apply_subst cl_main.c <<'EOF'
SCR_UpdateScreen ();
===NEW===
/* SCR_UpdateScreen (); — headless */
EOF

# 14. Fix M_Menu_Quit_f reference in CL_Quit_f
apply_subst cl_main.c <<'EOF'
	if (1 /* key_dest != key_console */ /* && cls.state != ca_dedicated */)
	{
		M_Menu_Quit_f ();
		return;
	}
===NEW===
	/* M_Menu_Quit_f removed for headless */
EOF

# 15. Remove Windows-specific CL_Windows_f
apply_subst cl_main.c <<'EOF'
#ifdef _WINDOWS
#include <windows.h>
===NEW===
/* Windows section removed for headless */
#if 0
EOF

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

# 21d. Reset MVD state in CL_StopPlayback (upstream source uses `= 0;`,
# not `= false;` — the earlier needle silently no-op'd).
old_stop = 'cls.demoplayback = 0;'
new_stop = 'cls.demoplayback = 0;\n\tcls.mvdplayback = false;'
if 'cls.mvdplayback = false' not in t:
    t = t.replace(old_stop, new_stop, 1)

# 21d-mtime.  Set cl.mtime[0]/cl.mtime[1] from each record's demotime in
# CL_GetDemoMessage.  QW has no svc_time message (the opcode is reserved
# but commented out in protocol.h), so the QW client never natively
# populates cl.mtime[0].  The shared NQ-compat fields were previously
# kept = cl.time via the QNN_SyncEngineCompat sync, but that pinned the
# value to the emit boundary and threw away the per-record sub-emit
# timing we need for the MVD per-event fire back-shift.  This patch
# threads demotime through each record so sounds parsed during one
# record's dem_read get cl.mtime[0] == the record's actual demotime,
# not the emit-window start.
old_after_gate = 'if (cls.state < ca_demostart)\n\t\tHost_Error ("CL_GetDemoMessage: cls.state != ca_active");'
new_after_gate = ('cl.mtime[1] = cl.mtime[0];\n'
                  '\tcl.mtime[0] = demotime;\n'
                  '\tif (cls.state < ca_demostart)\n'
                  '\t\tHost_Error ("CL_GetDemoMessage: cls.state != ca_active");')
if 'cl.mtime[0] = demotime;' not in t:
    t = t.replace(old_after_gate, new_after_gate, 1)

# 21e. Append CL_GetMVDMessage function at end of file
mvd_reader = r'''

/* ── MVD message reader ────────────────────────────────────────── */

qboolean CL_GetMVDMessage (void)
{
    int     r, i;
    byte    c, msec_byte;
    float   demotime;
    extern int qnn_mvd_anchor_player;

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

    /* Stamp message time from demo time.  QWD playback gets cl.mtime
     * from packet parsing; nothing on the MVD path writes it, leaving
     * cl.mtime[0] at 0.0 forever.  Everything timestamped off mtime
     * then breaks silently: sound events carry native_time 0, so the
     * fire/jump back-shift ring computes an out-of-range press offset
     * and drops every sound-driven label write; entity msgtime (the
     * recency obs) never advances. */
    cl.mtime[1] = cl.mtime[0];
    cl.mtime[0] = demotime;

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

    /* Per-recipient demux.  When locked to an anchor player, skip blocks
     * not addressed to that slot so the worker sees exactly the stream
     * the player received — including its own weapon-fire sound (a player
     * is always in its own PHS), and excluding fires it never heard.  The
     * payload was already consumed above, so the file position is correct
     * to fetch the next block.  Broadcast blocks (dem_all/dem_read) and
     * the unlocked case (anchor < 0) are never gated.  dem_single carries
     * a 5-bit target; dem_multiple a 32-bit recipient mask. */
    if (cls.mvdplayback && qnn_mvd_anchor_player >= 0
        && qnn_mvd_anchor_player < MAX_CLIENTS)
    {
        if (cls.mvd_lasttype == dem_single)
        {
            if (cls.mvd_lastto != (unsigned)qnn_mvd_anchor_player)
                goto readnext;
        }
        else if (cls.mvd_lasttype == dem_multiple)
        {
            if (!(cls.mvd_lastto & (1u << qnn_mvd_anchor_player)))
                goto readnext;
        }
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

/* MVD anchor player slot (see qnn.h).  -1 = unlocked / legacy spectator.
 * Set by the worker before each per-player collect pass. */
int qnn_mvd_anchor_player = -1;

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

# 23. Route dem_stats to correct player in cl_parse.c (MVD-only).
apply_subst cl_parse.c 'mvd_lasttype == dem_stats' <<'EOF'
void CL_SetStat (int stat, int value)
{
===NEW===
void CL_SetStat (int stat, int value)
{
	/* MVD: dem_stats targets a specific player. Only apply to
	 * cl.stats[] if it targets our tracked player. */
	if (cls.mvdplayback && cls.mvd_lasttype == dem_stats
		&& cls.mvd_lastto != cl.playernum)
		return;
EOF

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
\t\t\t\t * update sequences for cl.frames[] indexing.
\t\t\t\t *
\t\t\t\t * CL_GetDemoMessage returns 1 for dem_cmd / dem_set
\t\t\t\t * too, but only dem_read refreshes net_message — so
\t\t\t\t * after a dem_cmd this same server packet is still in
\t\t\t\t * net_message and would be RE-PARSED, re-firing every
\t\t\t\t * non-idempotent event (sounds, temp entities). Upstream
\t\t\t\t * QW relies on Netchan_Process to drop the stale/
\t\t\t\t * duplicate sequence (seq <= incoming_sequence); we
\t\t\t\t * bypass Netchan_Process here, so replicate that guard. */
\t\t\t\tunsigned seq = (unsigned)MSG_ReadLong () & ~(1u << 31);
\t\t\t\tunsigned ack = (unsigned)MSG_ReadLong () & ~(1u << 31);
\t\t\t\tif (seq <= (unsigned)cls.netchan.incoming_sequence
\t\t\t\t\t&& cls.netchan.incoming_sequence != 0)
\t\t\t\t\tcontinue;\t\t// stale net_message after dem_cmd/dem_set
\t\t\t\tcls.netchan.incoming_sequence = seq;
\t\t\t\tcls.netchan.incoming_acknowledged = ack;
\t\t\t\tcls.netchan.last_received = realtime;
\t\t\t}
\t\t}
\t\telse
\t\t{
\t\t\tif (!Netchan_Process(&cls.netchan))
\t\t\t\tcontinue;\t\t// wasn't accepted for some reason
\t\t}"""
if 'QWD dem_read: strip' not in t and 'MVD: no netchan header' not in t:
    t = t.replace(old_netchan, new_netchan, 1)
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
\t\t\telse if (cls.demoplayback && protover >= 24 && protover <= 27)
\t\t\t\tbreak; /* old demo compat: protocols 24-27 from pre-2.30 clients */
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
\t\textern int qnn_mvd_anchor_player;
\t\t/* MVD: demo start time (float) instead of playernum (byte). */
\t\t{
\t\t\tfloat mvd_start = MSG_ReadFloat();
\t\t\tcls.netchan.last_received = mvd_start;
\t\t}
\t\t/* Lock to the anchor player when set: the self/observation
\t\t * pipeline keys on cl.playernum, so this makes the proven
\t\t * single-POV emission path apply to MVD verbatim.  Unlocked
\t\t * (-1) keeps the legacy spectator slot. */
\t\tif (qnn_mvd_anchor_player >= 0 && qnn_mvd_anchor_player < MAX_CLIENTS)
\t\t{
\t\t\tcl.playernum = qnn_mvd_anchor_player;
\t\t\tcl.spectator = false;
\t\t}
\t\telse
\t\t{
\t\t\tcl.playernum = MAX_CLIENTS - 1;
\t\t\tcl.spectator = true;
\t\t}
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
    t = t.replace(old_rc, new_rc, 1)

    # Override MSG_ReadAngle for float coords
    old_ra = 'float MSG_ReadAngle (void)\n{\n\treturn MSG_ReadChar() * (360.0/256);'
    new_ra = '''float MSG_ReadAngle (void)
{
\tif (qnn_fte_floatcoords)
\t\treturn MSG_ReadShort() * (360.0/65536);
\treturn MSG_ReadChar() * (360.0/256);'''
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


# 30a2. Bump MAX_STATIC_ENTITIES from 128 to 512 (same value ezQuake uses).
#       Long multi-map demos can accumulate well past 128 per-map statics
#       even after CL_ClearState runs each map change — QW's vanilla 128
#       cap was chosen for single-level play.  At 128, dgvsgq1-style
#       15-map demo series Host_EndGame mid-replay.
python3 - "${WORKTREE_DIR}/client.h" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
old = '#define\tMAX_STATIC_ENTITIES\t128'
new = '#define\tMAX_STATIC_ENTITIES\t512'
if 'MAX_STATIC_ENTITIES\t512' not in t:
    t = t.replace(old, new, 1)
p.write_text(t, errors='surrogateescape')
PY

# 30b. Skip packet entities whose model index is invalid or missing.
#      Some demos reference models that were not precached successfully in
#      headless playback. Vanilla CL_LinkPacketEntities dereferences
#      cl.model_precache[s1->modelindex] unconditionally; for data collection
#      this visual entity can be dropped instead of crashing the worker.
python3 - "${WORKTREE_DIR}/cl_ents.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
old = '''\t\t// if set to invisible, skip
\t\tif (!s1->modelindex)
\t\t\tcontinue;

\t\t// create a new entity'''
new = '''\t\t// if set to invisible, skip
\t\tif (!s1->modelindex)
\t\t\tcontinue;

\t\t// QNN headless collect: demos can reference absent or out-of-range
\t\t// precache entries. Skip the visual entity rather than dereferencing
\t\t// a NULL/garbage model pointer in the renderer path.
\t\tif (s1->modelindex >= MAX_MODELS || cl.model_precache[s1->modelindex] == NULL)
\t\t\tcontinue;

\t\t// create a new entity'''
if 'QNN headless collect: demos can reference absent' not in t:
    if old not in t:
        raise SystemExit("failed to patch CL_LinkPacketEntities model guard")
    t = t.replace(old, new, 1)

# 30b2. CL_ParsePlayerinfo: vanilla QW uses `num > MAX_CLIENTS` with a Sys_Error.
#       1) `>` should be `>=` — num is an index into [0..MAX_CLIENTS-1].  The
#          stricter compare catches the off-by-one silently corrupting slot
#          MAX_CLIENTS. 2) We replace the fatal Sys_Error with a swallow +
#          return so one malformed playerinfo doesn't kill the replay.
old_ppi = '''\tnum = MSG_ReadByte ();
\tif (num > MAX_CLIENTS)
\t\tSys_Error ("CL_ParsePlayerinfo: bad num");'''
new_ppi = '''\tnum = MSG_ReadByte ();
\tif (num >= MAX_CLIENTS)
\t{
\t\tfprintf(stderr, "[qw-demo] CL_ParsePlayerinfo: bad num %d — skipping\\n", num);
\t\treturn;
\t}'''
if '[qw-demo] CL_ParsePlayerinfo: bad num' not in t:
    if old_ppi not in t:
        raise SystemExit("failed to patch CL_ParsePlayerinfo bad-num guard")
    t = t.replace(old_ppi, new_ppi, 1)

p.write_text(t, errors='surrogateescape')
PY

# 30b2c. svc_spawnbaseline: vanilla QW indexes cl_baselines[i] without any
#        bounds check on the i = MSG_ReadShort() it just read.  Malformed
#        demos (or our own parser desyncing on something we tolerated
#        earlier in the stream) hand i values like 2048 or -16384 here,
#        which segfaults inside CL_ParseBaseline on the first write.
#        Skip out-of-range entries while still draining the payload so
#        we stay aligned with the rest of the message.
python3 - "${WORKTREE_DIR}/cl_parse.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
old_sb = '''\t\tcase svc_spawnbaseline:
\t\t\ti = MSG_ReadShort ();
\t\t\tCL_ParseBaseline (&cl_baselines[i]);
\t\t\tbreak;'''
new_sb = '''\t\tcase svc_spawnbaseline:
\t\t{
\t\t\tentity_state_t qnn_baseline_junk;
\t\t\ti = MSG_ReadShort ();
\t\t\tif (i < 0 || i >= MAX_EDICTS)
\t\t\t{
\t\t\t\tfprintf(stderr, "[qw-demo] svc_spawnbaseline: bad entnum %d — skipping\\n", i);
\t\t\t\tCL_ParseBaseline (&qnn_baseline_junk);
\t\t\t\tbreak;
\t\t\t}
\t\t\tCL_ParseBaseline (&cl_baselines[i]);
\t\t\tbreak;
\t\t}'''
if '[qw-demo] svc_spawnbaseline:' not in t:
    if old_sb not in t:
        raise SystemExit("failed to patch svc_spawnbaseline bounds guard")
    t = t.replace(old_sb, new_sb, 1)
p.write_text(t, errors='surrogateescape')
PY

# 30b3. CL_SetStat: `stat < 0 || stat >= MAX_CL_STATS` → Sys_Error.  MVDSV
#       / ezQuake extended stats occasionally push indices above MAX_CL_STATS
#       (32 in vanilla, 96 in ezQuake).  Raising the cap is risky (affects
#       cl.stats[] indexing everywhere), so just skip out-of-range stats.
python3 - "${WORKTREE_DIR}/cl_parse.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
old = '\tif (stat < 0 || stat >= MAX_CL_STATS)\n\t\tSys_Error ("CL_SetStat: %i is invalid", stat);'
new = ('\tif (stat < 0 || stat >= MAX_CL_STATS)\n'
       '\t{\n'
       '\t\tfprintf(stderr, "[qw-demo] CL_SetStat: %i is invalid — skipping\\n", stat);\n'
       '\t\treturn;\n'
       '\t}')
if '[qw-demo] CL_SetStat:' not in t:
    if old not in t:
        raise SystemExit('failed to patch CL_SetStat bad-stat guard')
    t = t.replace(old, new, 1)
p.write_text(t, errors='surrogateescape')
PY

# 30b5b. CL_SetSolidEntities: zero physents[] before rebuild so stale model
#        pointers from a previous frame can't leak into pmove.  Without this
#        a freed-model pointer (e.g. after a mid-demo map change whose
#        svc_modellist got swallowed by our tolerance patches) sits in
#        physents past the numphysent cap and slips through if anything
#        iterates beyond that cap.  Zeroing is cheap (one array × sizeof).
python3 - "${WORKTREE_DIR}/cl_ents.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
old = '\tpmove.physents[0].model = cl.worldmodel;\n\tVectorCopy (vec3_origin, pmove.physents[0].origin);\n\tpmove.physents[0].info = 0;\n\tpmove.numphysent = 1;'
new = ('\tmemset(pmove.physents, 0, sizeof(pmove.physents));\n'
       '\tpmove.physents[0].model = cl.worldmodel;\n'
       '\tVectorCopy (vec3_origin, pmove.physents[0].origin);\n'
       '\tpmove.physents[0].info = 0;\n'
       '\tpmove.numphysent = 1;')
if 'memset(pmove.physents, 0, sizeof(pmove.physents))' not in t:
    if old not in t:
        raise SystemExit('failed to patch CL_SetSolidEntities physents memset')
    t = t.replace(old, new, 1)
p.write_text(t, errors='surrogateescape')
PY

# 30b5a. Disable other-player prediction.  CL_LinkPlayers runs client-side
#        pmove prediction for every visible player every frame; the result
#        is only used to nudge the rendered entity's origin.  Two MVD demos
#        SIGSEGV inside PM_TestPlayerPosition via this path (stale model
#        pointer in pmove.physents after mid-demo map change).  The worker
#        never renders, so we don't need the predicted origin — use the
#        raw state origin and skip the predict entirely.  Matches vanilla
#        QW behaviour when cl_predict_players=0.
python3 - "${WORKTREE_DIR}/cl_main.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
old1 = 'cvar_t\tcl_predict_players = {"cl_predict_players", "1"};'
new1 = 'cvar_t\tcl_predict_players = {"cl_predict_players", "0"};'
if new1 not in t:
    t = t.replace(old1, new1, 1)
old2 = 'cvar_t\tcl_predict_players2 = {"cl_predict_players2", "1"};'
new2 = 'cvar_t\tcl_predict_players2 = {"cl_predict_players2", "0"};'
if new2 not in t and old2 in t:
    t = t.replace(old2, new2, 1)
p.write_text(t, errors='surrogateescape')
PY

# 30b5. PM_TestPlayerPosition: guard against stale model pointers in physents.
#       Two MVD demos SIGSEGV via CL_LinkPlayers → CL_PredictUsercmd →
#       PlayerMove → NudgePosition → PM_TestPlayerPosition, dereffing
#       `&physents[i].model->hulls[1]` with a non-NULL but invalid pointer
#       (likely stale after a map change eats a partial svc sequence).
#       Skip any physent whose model->hulls[1] looks unusable for collision,
#       same spirit as the CL_LinkPacketEntities modelindex guard.
python3 - "${WORKTREE_DIR}/pmovetst.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
old = '''\tfor (i=0 ; i< pmove.numphysent ; i++)
\t{
\t\tpe = &pmove.physents[i];
\t// get the clipping hull
\t\tif (pe->model)
\t\t\thull = &pmove.physents[i].model->hulls[1];
\t\telse
\t\t{'''
new = '''\tfor (i=0 ; i< pmove.numphysent ; i++)
\t{
\t\tpe = &pmove.physents[i];
\t// get the clipping hull
\t\tif (pe->model)
\t\t{
\t\t\t/* QNN headless collect: svc-tolerance can leave model_precache
\t\t\t * entries stale after a skipped-message sequence.  Reject any
\t\t\t * physent whose BSP hulls aren't actually populated so we don't
\t\t\t * chase a freed hull pointer into SIGSEGV-land. */
\t\t\thull = &pmove.physents[i].model->hulls[1];
\t\t\tif (hull == NULL || hull->firstclipnode == 0)
\t\t\t\tcontinue;
\t\t}
\t\telse
\t\t{'''
if 'QNN headless collect: svc-tolerance' not in t:
    if old not in t:
        raise SystemExit('failed to patch PM_TestPlayerPosition stale-model guard')
    t = t.replace(old, new, 1)
p.write_text(t, errors='surrogateescape')
PY

# 30b4. CL_ParseTEnt: unknown temp-entity type → Sys_Error.  ezQuake servers
#       emit TE types beyond the vanilla set (chat bubbles, blood spikes,
#       etc.).  For data collection we don't render, so swallow and skip.
python3 - "${WORKTREE_DIR}/cl_tent.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
old = '\tdefault:\n\t\tSys_Error ("CL_ParseTEnt: bad type");'
new = ('\tdefault:\n'
       '\t\tfprintf(stderr, "[qw-demo] CL_ParseTEnt: bad type %d — skipping\\n", type);\n'
       '\t\treturn;')
if '[qw-demo] CL_ParseTEnt: bad type' not in t:
    if old not in t:
        raise SystemExit('failed to patch CL_ParseTEnt bad-type guard')
    t = t.replace(old, new, 1)
p.write_text(t, errors='surrogateescape')
PY

# 30c. Tolerate unknown svc codes instead of terminating the demo.
#      Many QWD demos from FTE/ezQuake-era servers contain extension svc
#      codes (0x53 voicechat, 0x8d extended-entities, etc.) that vanilla
#      QW's parser doesn't recognize.  The default case originally called
#      Host_EndGame, wiping the entire replay.  For data-collection we
#      instead break out of the per-message switch: the current net_message
#      is abandoned but demo playback continues with the next one.  Trailing
#      zero padding on MVD messages (svc_bad) is also treated as end-of-
#      message for the same reason.
python3 - "${WORKTREE_DIR}/cl_parse.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
old = 'Host_EndGame ("CL_ParseServerMessage: Illegible server message");'
new = '''if (cls.mvdplayback && cmd == svc_bad)
\t\t\t\tbreak; /* MVD trailing zeros — end of message */
\t\t\t/* Extended server protocols (FTE, ezQuake) use svc codes the
\t\t\t   vanilla QW parser doesn't know.  For data collection we'd
\t\t\t   rather abandon this one network message than terminate the
\t\t\t   entire demo — we can't know the extension's payload length,
\t\t\t   so the rest of this packet is lost, but subsequent demo
\t\t\t   packets are independent framing and remain parseable. */
\t\t\treturn;'''
if 'Extended server protocols' not in t:
    t = t.replace(old, new, 1)
p.write_text(t, errors='surrogateescape')
PY

# 30d. Treat mid-demo svc_disconnect (after successful signon) as a no-op
#      instead of fatal Host_EndGame, so multi-session recordings don't
#      truncate at the first inter-session boundary.  Tournament demos are
#      commonly recorded across multiple sessions (warmup → disconnect →
#      match → disconnect → next-map) and the demo file contains a fresh
#      svc_serverdata after each disconnect.  The upstream client's chain
#      (svc_disconnect → Host_EndGame → CL_Disconnect → CL_StopPlayback)
#      closes the demofile and ends playback, which is GUI-client
#      behavior — wrong for data collection.
#
#      Approach: when state == ca_active (signon completed), simply
#      `return` from the message handler — don't change cls.state or
#      touch the demo file.  CL_GetDemoMessage requires state >=
#      ca_demostart so we can't drop to ca_disconnected, and CL_ClearState
#      would free the hunk (Mod_ClearAll / Hunk_FreeToLowMark) and
#      invalidate cl_entities / qnn_store references.  The follow-up
#      svc_serverdata calls CL_ClearState itself when it arrives.
#
#      Pre-signon disconnects (state in ca_demostart / ca_connected) fall
#      through to upstream Host_EndGame so signon failures still terminate
#      cleanly — running the engine with a partially-initialised cl can
#      SIGSEGV on subsequent svc messages.
python3 - "${WORKTREE_DIR}/cl_parse.c" <<'PY'
from pathlib import Path; import sys
p = Path(sys.argv[1]); t = p.read_text(errors='surrogateescape')
old = '''		case svc_disconnect:
			if (cls.state == ca_connected)
				Host_EndGame ("Server disconnected\\n"
					"Server version may not be compatible");
			else
				Host_EndGame ("Server disconnected");
			break;'''
new = '''		case svc_disconnect:
			/* Multi-session demo recording: the file contains a
			   fresh svc_serverdata after each disconnect.  Reset
			   client state but keep the demofile open so playback
			   transitions cleanly into the next session.  Live
			   non-demo connections still hit the upstream
			   Host_EndGame path. */
			if (cls.demoplayback && cls.state == ca_active)
				return;
			if (cls.state == ca_connected)
				Host_EndGame ("Server disconnected\\n"
					"Server version may not be compatible");
			else
				Host_EndGame ("Server disconnected");
			break;'''
if 'Multi-session demo recording' not in t:
    t = t.replace(old, new, 1)
p.write_text(t, errors='surrogateescape')
PY

# 26. Hook pmove.c JumpButton success branch to set qnn_pmove_jump_attacked
#     so the QWD labeler's per-cmd pmove driver can detect physics jumps
#     directly (no snapshot.grounded lag, no K-gate workaround).  Single
#     line added inside JumpButton after the velocity+270 bump; guarded
#     so rebuilds are idempotent.
apply_subst pmove.c 'qnn_pmove_jump_attacked' <<'EOF'
	onground = -1;
	pmove.velocity[2] += 270;

	pmove.oldbuttons |= BUTTON_JUMP;	// don't jump again until released
}
===NEW===
	onground = -1;
	pmove.velocity[2] += 270;

	{ extern int qnn_pmove_jump_attacked; qnn_pmove_jump_attacked = 1; }

	pmove.oldbuttons |= BUTTON_JUMP;	// don't jump again until released
}
EOF

# 27. Reload the QC VM after a (mid-demo) map change.  CL_ParseServerData
#     runs CL_ClearState -> Hunk_FreeToLowMark, which frees the hunk the QC
#     VM (progs/pr_functions/pr_strings/edicts, allocated once by
#     QNN_ProgsInit) lives in.  On multi-map / multi-session demos the next
#     QC call then faults in ED_FindFunction on dangling memory.  Flag the
#     reset here; qnn_progs.c reloads the VM lazily before the next QC call.
#     Guarded so rebuilds are idempotent.
apply_subst cl_parse.c 'QNN_ProgsNotifyWorldReset' <<'EOF'
	CL_ClearState ();
===NEW===
	CL_ClearState ();
	/* QNN: the hunk holding the QC VM was just freed by CL_ClearState;
	   flag it so qnn_progs.c reloads the VM before the next QC call. */
	{ extern void QNN_ProgsNotifyWorldReset(void); QNN_ProgsNotifyWorldReset(); }
EOF

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

# Same as compile_c but with implicit-function-declaration promoted to an
# error.  Used for QNN-owned sources only — upstream Quake code predates
# clean prototypes and needs -w to compile at all.  Catches the kind of
# missing-prototype bug that silently broke SV_RecursiveHullCheck.
compile_c_strict() {
  local src="$1"
  local obj="${OBJ_DIR}/$(basename "${src}").o"
  cc \
    -std=gnu89 \
    -O2 \
    -fcommon \
    -w \
    -Werror=implicit-function-declaration \
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
  compile_c_strict "${source}"
done

# Compile shared C++ nav sources
for source in "${CUSTOM_CXX_SOURCES[@]}"; do
  compile_cxx "${source}"
done
for source in "${NAV_CXX_SOURCES[@]}"; do
  compile_cxx "${source}"
done

# Link.  -rdynamic exports symbols into the dynamic table so
# backtrace_symbols_fd() in qnn_fault.c can print function names rather
# than raw addresses when a worker crashes.
c++ \
  -O2 \
  -w \
  -rdynamic \
  -o "${OUTPUT_PATH}" \
  "${OBJECTS[@]}" \
  -lm

# Install qwprogs.dat into the asset search path (assets/qw/).  Required
# by the worker's --sanitize-inputs mode, which loads it through the
# real QC VM (qnn_progs.c) for per-tick predicate evaluation.  Sourced
# from the same vendor commit as the C sources above.
QWPROGS_DST="${REPO_ROOT}/assets/qw/qwprogs.dat"
mkdir -p "$(dirname "${QWPROGS_DST}")"
cp "${UPSTREAM_DIR}/QW/progs/qwprogs.dat" "${QWPROGS_DST}"

printf '%s\n' "${OUTPUT_PATH}"

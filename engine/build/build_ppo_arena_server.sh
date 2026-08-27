#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=build_common.sh
source "${SCRIPT_DIR}/build_common.sh"

OUTPUT_PATH=${1:-"${REPO_ROOT}/assets/bin/ppo_arena_server"}
PATCHES=(
  "${COMMON_PATCHES[@]}"
  "${ENGINE_DIR}/patches/host.c.patch"
  "${ENGINE_DIR}/patches/arena/host.c.patch"
  "${ENGINE_DIR}/patches/arena/cl_main.c.patch"
  "${ENGINE_DIR}/patches/arena/cl_parse.c.patch"
  "${ENGINE_DIR}/patches/arena/cl_demo.c.patch"
  "${ENGINE_DIR}/patches/arena/net_main.c.patch"
  "${ENGINE_DIR}/patches/arena/net_loop.c.patch"
  "${ENGINE_DIR}/patches/arena/net_dgrm.c.patch"
  "${ENGINE_DIR}/patches/arena/sv_main.c.patch"
  "${ENGINE_DIR}/patches/arena/sv_user.c.patch"
)
CUSTOM_SOURCES=(
  "${ENGINE_DIR}/nq/qnn_sys.c"
  "${ENGINE_DIR}/common/qnn_sys_common.c"
  "${ENGINE_DIR}/common/qnn_context.c"
  "${ENGINE_DIR}/nq/qnn_arena_server_main.c"
  "${ENGINE_DIR}/nq/qnn_arena_engine.c"
  "${ENGINE_DIR}/nq/qnn_arena_observer.c"
  "${ENGINE_DIR}/nq/qnn_arena_virtual.c"
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
  "${ENGINE_DIR}/nq/qnn_progs_server.c"
  "${ENGINE_DIR}/common/qnn_io.c"
  "${ENGINE_DIR}/common/qnn_obs_registry.c"
  "${ENGINE_DIR}/common/qnn_obs_shim.c"
  "${ENGINE_DIR}/common/qnn_metrics.c"
  "${ENGINE_DIR}/common/qnn_fault.c"
  "${ENGINE_DIR}/nq/qnn_reward.c"
  "${ENGINE_DIR}/common/qnn_store.c"
  "${ENGINE_DIR}/common/qnn_tick.c"
)

check_build_deps "the grouped PPO arena server"
BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/quake-arena-server-build.XXXXXX")
trap 'rm -rf "${BUILD_ROOT}"' EXIT
prepare_upstream "${BUILD_ROOT}" "${PATCHES[@]}"

# The arena progs.dat uses the same training builtins as ppo_worker.
python3 - "${WORKTREE_DIR}/pr_cmds.c" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
needle = '#define\tRETURN_EDICT(e) (((int *)pr_globals)[OFS_RETURN] = EDICT_TO_PROG(e))\n'
insert = needle + '\n' + ''.join(
    f'void {name} (void);\n' for name in (
        'PF_qnn_training_note_shot', 'PF_qnn_training_note_damage',
        'PF_qnn_training_note_death', 'PF_qnn_training_note_item',
        'PF_qnn_checkextension',
    )
)
if needle in text and 'PF_qnn_training_note_shot' not in text:
    text = text.replace(needle, insert, 1)
table = 'PF_precache_file,\n\nPF_setspawnparms\n};\n'
replacement = (
    'PF_precache_file,\n\nPF_setspawnparms,\n'
    + 'PF_qnn_training_note_shot,\n'
    + 'PF_qnn_training_note_damage,\n'
    + 'PF_qnn_training_note_death,\n'
    + 'PF_qnn_training_note_item,\n'
    + 'PF_Fixme,\n' * 16
    + 'PF_qnn_checkextension,\n'
    + 'PF_Fixme,\n' * 20
    + '};\n'
)
if table in text:
    text = text.replace(table, replacement, 1)
path.write_text(text, encoding="utf-8")
PY

build_worker "${OUTPUT_PATH}" "${CUSTOM_SOURCES[@]}"

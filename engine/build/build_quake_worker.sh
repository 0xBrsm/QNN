#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=build_common.sh
source "${SCRIPT_DIR}/build_common.sh"

OUTPUT_PATH=${1:-"${REPO_ROOT}/assets/bin/quake_worker"}

PATCHES=("${COMMON_PATCHES[@]}" "${ENGINE_DIR}/patches/host.c.patch")

CUSTOM_SOURCES=(
  "${ENGINE_DIR}/worker/qnn_sys.c"
  "${ENGINE_DIR}/worker/qnn_trainer_main.c"
  "${ENGINE_DIR}/worker/qnn_input.c"
  "${ENGINE_DIR}/worker/qnn_sound.c"
  "${ENGINE_DIR}/worker/qnn_obs.c"
  "${ENGINE_DIR}/worker/qnn_metrics.c"
  "${ENGINE_DIR}/worker/qnn_reward.c"
  "${ENGINE_DIR}/worker/qnn_world.c"
)

check_build_deps "the Quake worker"

BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/quake-worker-build.XXXXXX")
trap 'rm -rf "${BUILD_ROOT}"' EXIT

prepare_upstream "${BUILD_ROOT}" "${PATCHES[@]}"

# Inject training builtin declarations into pr_cmds.c
python3 - "${WORKTREE_DIR}/pr_cmds.c" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

decl_needle = '#define\tRETURN_EDICT(e) (((int *)pr_globals)[OFS_RETURN] = EDICT_TO_PROG(e))\n'
decl_insert = (
    decl_needle
    + '\n'
    + 'void PF_qnn_training_note_shot (void);\n'
    + 'void PF_qnn_training_note_damage (void);\n'
    + 'void PF_qnn_training_note_death (void);\n'
    + 'void PF_qnn_training_note_item (void);\n'
    + 'void PF_qnn_checkextension (void);\n'
)
if decl_needle in text and 'PF_qnn_training_note_shot' not in text:
    text = text.replace(decl_needle, decl_insert, 1)

table_needle = 'PF_precache_file,\n\nPF_setspawnparms\n};\n'
table_insert = (
    'PF_precache_file,\n\n'
    + 'PF_setspawnparms,\n'                        # 78
    + 'PF_qnn_training_note_shot,\n'               # 79
    + 'PF_qnn_training_note_damage,\n'             # 80
    + 'PF_qnn_training_note_death,\n'              # 81
    + 'PF_qnn_training_note_item,\n'               # 82
    + 'PF_Fixme,\n' * 16                           # 83-98
    + 'PF_qnn_checkextension,\n'                   # 99 — returns 0 (no extensions)
    + 'PF_Fixme,\n' * 20                           # 100-119
    + '};\n'
)
if table_needle in text:
    text = text.replace(table_needle, table_insert, 1)

path.write_text(text, encoding="utf-8")
PY

build_worker "${OUTPUT_PATH}" "${CUSTOM_SOURCES[@]}"

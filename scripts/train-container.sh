#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SRC_ROOT}/.." && pwd)"

if [ -n "${HOST_SRC_ROOT:-}" ]; then
  HOST_SRC_ROOT="${HOST_SRC_ROOT}"
elif [ -n "${HOST_WORKSPACE:-}" ]; then
  HOST_SRC_ROOT="${HOST_WORKSPACE}/src"
else
  HOST_SRC_ROOT="${SRC_ROOT}"
fi

if [ -n "${HOST_REPO_ROOT:-}" ]; then
  HOST_REPO_ROOT="${HOST_REPO_ROOT}"
elif [ -n "${HOST_WORKSPACE:-}" ]; then
  HOST_REPO_ROOT="${HOST_WORKSPACE}"
else
  HOST_REPO_ROOT="${REPO_ROOT}"
fi

if [ -n "${HOST_ARTIFACT_ROOT:-}" ]; then
  HOST_ARTIFACT_ROOT="${HOST_ARTIFACT_ROOT}"
elif [ -n "${HOST_WORKSPACE:-}" ]; then
  HOST_ARTIFACT_ROOT="${HOST_WORKSPACE}/artifacts"
else
  HOST_ARTIFACT_ROOT="$(dirname "$HOST_SRC_ROOT")/artifacts"
fi

TRAINING_AMD_TARGET="${TRAINING_AMD_TARGET:-auto}"
TRAINING_IMAGE_TAG="${TRAINING_IMAGE_TAG:-quake-ai-trainer-amd:local}"
QUAKE_AI_DEVICE="${QUAKE_AI_DEVICE:-gpu}"
ROCBLAS_USE_HIPBLASLT="${ROCBLAS_USE_HIPBLASLT:-0}"
WSL_LIBDXCORE_PATH="${WSL_LIBDXCORE_PATH:-/usr/lib/wsl/lib/libdxcore.so}"
WSL_LIBROCDXG_PATH="${WSL_LIBROCDXG_PATH:-/opt/rocm/lib/librocdxg.so}"
HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
HOST_QUAKE_ASSET_ROOT="${HOST_QUAKE_ASSET_ROOT:-${QUAKE_BASEDIR:-}}"
TRAINING_SHM_SIZE="${TRAINING_SHM_SIZE:-16gb}"
TRAINER_USER_UID="${TRAINER_USER_UID:-1000}"
TRAINER_USER_GID="${TRAINER_USER_GID:-1000}"

_looks_like_quake_basedir() {
  local candidate="$1"
  [ -d "${candidate}/id1" ] || return 1
  [ -f "${candidate}/id1/PAK0.PAK" ] || [ -f "${candidate}/id1/PAK1.PAK" ] || [ -f "${candidate}/id1/pak0.pak" ] || [ -f "${candidate}/id1/pak1.pak" ]
}

host_path_has_quake_assets() {
  local candidate="$1"
  docker run --rm -v "${candidate}:/assets:ro" ubuntu:24.04 bash -lc 'test -d /assets/id1 && (test -f /assets/id1/PAK0.PAK || test -f /assets/id1/PAK1.PAK || test -f /assets/id1/pak0.pak || test -f /assets/id1/pak1.pak)' >/dev/null 2>&1
}

host_path_exists() {
  local candidate="$1"
  local mount_target="$2"
  docker run --rm -v "${candidate}:${mount_target}:ro" ubuntu:24.04 bash -lc "test -e ${mount_target}" >/dev/null 2>&1
}

resolve_host_quake_asset_root() {
  local candidates=()
  if [ -n "${HOST_QUAKE_ASSET_ROOT}" ]; then
    candidates+=("${HOST_QUAKE_ASSET_ROOT}")
  fi
  if _looks_like_quake_basedir "${REPO_ROOT}/assets"; then
    candidates+=("${HOST_REPO_ROOT}/assets")
  fi
  if _looks_like_quake_basedir "$(dirname "$SRC_ROOT")/assets"; then
    candidates+=("$(dirname "$HOST_SRC_ROOT")/assets")
  fi

  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -n "${candidate}" ] && host_path_has_quake_assets "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

docker_has_device() {
  local device_path="$1"
  shift
  docker run --rm "$@" ubuntu:24.04 bash -lc "test -e ${device_path}" >/dev/null 2>&1
}

TRAINING_TARGET=""
COMPOSE_FILE=""

prepare_compose() {
  if [ -n "${COMPOSE_FILE}" ] && [ -n "${TRAINING_TARGET}" ]; then
    return 0
  fi

  case "${TRAINING_AMD_TARGET}" in
    wsl|amd-wsl)
      TRAINING_TARGET="wsl"
      ;;
    rocm|linux|amd-rocm)
      TRAINING_TARGET="rocm"
      ;;
    auto)
      if docker_has_device /dev/dxg --device /dev/dxg; then
        TRAINING_TARGET="wsl"
      elif docker_has_device /dev/kfd --device /dev/kfd --device /dev/dri; then
        TRAINING_TARGET="rocm"
      else
        echo "No AMD accelerator device is visible to Docker. Expose /dev/dxg for WSL ROCm or /dev/kfd plus /dev/dri for native Linux ROCm." >&2
        exit 1
      fi
      ;;
    *)
      echo "unsupported TRAINING_AMD_TARGET: ${TRAINING_AMD_TARGET}" >&2
      exit 1
      ;;
  esac

  case "${TRAINING_TARGET}" in
    wsl)
      COMPOSE_FILE="${SRC_ROOT}/docker/training/compose.amd-wsl.yaml"
      ;;
    rocm)
      COMPOSE_FILE="${SRC_ROOT}/docker/training/compose.amd-rocm.yaml"
      ;;
  esac
}

require_target_prereqs() {
  prepare_compose
  case "${TRAINING_TARGET}" in
    wsl)
      host_path_exists "${WSL_LIBDXCORE_PATH}" /runtime/libdxcore.so || { echo "Missing WSL libdxcore.so at ${WSL_LIBDXCORE_PATH}" >&2; exit 1; }
      host_path_exists "${WSL_LIBROCDXG_PATH}" /runtime/librocdxg.so || { echo "Missing WSL librocdxg.so at ${WSL_LIBROCDXG_PATH}" >&2; exit 1; }
      ;;
    rocm)
      docker_has_device /dev/kfd --device /dev/kfd --device /dev/dri || {
        echo "Docker cannot access /dev/kfd and /dev/dri for native Linux ROCm." >&2
        exit 1
      }
      ;;
  esac
}

ensure_host_artifact_root() {
  docker run --rm -v "${HOST_ARTIFACT_ROOT}:/artifacts" ubuntu:24.04 bash -lc "mkdir -p /artifacts/runs /artifacts/corpus && chown -R ${TRAINER_USER_UID}:${TRAINER_USER_GID} /artifacts" >/dev/null
}

export HOST_REPO_ROOT HOST_SRC_ROOT HOST_ARTIFACT_ROOT TRAINING_AMD_TARGET TRAINING_IMAGE_TAG QUAKE_AI_DEVICE ROCBLAS_USE_HIPBLASLT WSL_LIBDXCORE_PATH WSL_LIBROCDXG_PATH HSA_ENABLE_DXG_DETECTION TRAINING_SHM_SIZE TRAINER_USER_UID TRAINER_USER_GID

compose() {
  prepare_compose
  docker compose -f "$COMPOSE_FILE" "$@"
}

usage() {
  cat <<'EOF'
Usage:
  scripts/train-container.sh build
  scripts/train-container.sh check
  scripts/train-container.sh target
  scripts/train-container.sh shell
  scripts/train-container.sh install-frikbotnex
  scripts/train-container.sh build-worker
  scripts/train-container.sh live-check
  scripts/train-container.sh migrate-legacy
  scripts/train-container.sh run -- <command...>
  scripts/train-container.sh e1m1-corpus {bc|ppo|eval|distill|all}
  scripts/train-container.sh e1m1-world {plan|report|check|collect|bc|eval-bc|ppo|eval|all}
  scripts/train-container.sh e1m1-world-verify {plan|report|check|collect|bc|eval-bc|ppo|eval|all}
  scripts/train-container.sh combat-verify {plan|report|check|collect|bc|eval-bc|ppo|eval|all}
  scripts/train-container.sh combat {plan|report|check|collect|bc|eval-bc|ppo|eval|all}
  scripts/train-container.sh combat-bot-verify {plan|report|check|collect|bc|eval-bc|ppo|eval|all}
  scripts/train-container.sh combat-bot {plan|report|check|collect|bc|eval-bc|ppo|eval|all}
  scripts/train-container.sh combat-bot-<scenario>[-verify] {plan|report|check|collect|bc|eval-bc|ppo|eval|all}

Environment:
  HOST_REPO_ROOT      Host path to the dev repo root. Defaults to HOST_WORKSPACE when available.
  HOST_SRC_ROOT       Host path to the publishable src repo. Defaults to the current src root.
  HOST_ARTIFACT_ROOT  Host path for writable corpus and run artifacts. Defaults to a sibling 'artifacts' directory.
  TRAINING_AMD_TARGET AMD runtime target: auto, wsl, or rocm. Defaults to auto.
  TRAINING_IMAGE_TAG  Docker image tag for the trainer image.
  QUAKE_AI_DEVICE     Torch device request inside the trainer. Defaults to 'gpu'.
  ROCBLAS_USE_HIPBLASLT Set to 0 to force rocBLAS/Tensile over hipBLASLt inside the AMD trainer. Defaults to 0.
  HOST_QUAKE_ASSET_ROOT Host path to the Quake asset basedir mounted into live worker runs at /assets.
  TRAINING_SHM_SIZE   Shared-memory size for the trainer container. Defaults to '16gb'.
  WSL_LIBDXCORE_PATH  Host path to libdxcore.so for AMD ROCm on WSL.
  WSL_LIBROCDXG_PATH  Host path to librocdxg.so for AMD ROCm on WSL.
  HSA_ENABLE_DXG_DETECTION Set to 1 to enable the ROCm WSL DXG bridge inside containers.

Examples:
  scripts/train-container.sh build
  scripts/train-container.sh check
  scripts/train-container.sh target
  scripts/train-container.sh install-frikbotnex
  scripts/train-container.sh build-worker
  scripts/train-container.sh live-check
  scripts/train-container.sh migrate-legacy
  scripts/train-container.sh run -- python -m quake_ai.check_accelerator --device gpu
  scripts/train-container.sh e1m1-corpus all
  scripts/train-container.sh e1m1-world all
  scripts/train-container.sh combat-bot-verify check
  scripts/train-container.sh combat-bot-open-dm4-verify check
EOF
}

trainer_run_args() {
  local needs_assets="$1"
  local asset_mode="${2:-ro}"
  local asset_root

  if [ "${needs_assets}" = "1" ]; then
    asset_root="$(resolve_host_quake_asset_root)" || {
      echo "Quake assets were not found on the host. Set HOST_QUAKE_ASSET_ROOT or QUAKE_BASEDIR to a basedir that contains id1/PAK0.PAK." >&2
      exit 1
    }
    printf '%s\n' "--env" "QUAKE_BASEDIR=/assets" "--volume" "${asset_root}:/assets:${asset_mode}"
    return 0
  fi
}

run_in_trainer() {
  local needs_assets="${1:-0}"
  shift
  local asset_mode="ro"
  if [ "${1:-}" = "ro" ] || [ "${1:-}" = "rw" ]; then
    asset_mode="$1"
    shift
  fi

  local extra_args=()
  local arg
  while IFS= read -r arg; do
    extra_args+=("${arg}")
  done < <(trainer_run_args "${needs_assets}" "${asset_mode}")

  ensure_host_artifact_root
  compose run --rm --build "${extra_args[@]}" trainer-amd "$@"
}

COMMAND="${1:-help}"

case "${COMMAND}" in
  target)
    shift
    prepare_compose
    printf '%s\n' "${TRAINING_TARGET}"
    ;;
  build)
    shift
    require_target_prereqs
    compose build trainer-amd "$@"
    ;;
  check)
    shift
    require_target_prereqs
    run_in_trainer 0 python -m quake_ai.check_accelerator --device "${1:-$QUAKE_AI_DEVICE}" --fail-on-error
    ;;
  shell)
    shift
    require_target_prereqs
    run_in_trainer 0 bash "$@"
    ;;
  install-frikbotnex)
    shift
    require_target_prereqs
    run_in_trainer 1 rw python -m quake_ai.install_frikbotnex --asset-root /assets "$@"
    ;;
  build-worker)
    shift
    require_target_prereqs
    run_in_trainer 0 bash engine/build/build_quake_worker.sh /artifacts/bin/quake_worker "$@"
    ;;
  live-check)
    shift
    require_target_prereqs
    run_in_trainer 1 python -m quake_ai.live_training --profile verify --action check --device "${QUAKE_AI_DEVICE}" "$@"
    ;;
  migrate-legacy)
    shift
    mkdir -p "${REPO_ROOT}/artifacts/runs" "${REPO_ROOT}/artifacts/corpus"
    if [ -d "${SRC_ROOT}/runs" ]; then
      rsync -a --remove-source-files "${SRC_ROOT}/runs/" "${REPO_ROOT}/artifacts/runs/"
      find "${SRC_ROOT}/runs" -type d -empty -delete
    fi
    if [ -d "${SRC_ROOT}/corpus" ]; then
      rsync -a --remove-source-files "${SRC_ROOT}/corpus/" "${REPO_ROOT}/artifacts/corpus/"
      find "${SRC_ROOT}/corpus" -type d -empty -delete
    fi
    ;;
  run)
    shift
    require_target_prereqs
    if [ "${1:-}" = "--" ]; then
      shift
    fi
    if [ "$#" -eq 0 ]; then
      echo "run requires a command" >&2
      exit 1
    fi
    run_in_trainer 0 "$@"
    ;;
  e1m1-corpus)
    shift
    require_target_prereqs
    action="${1:-all}"
    shift || true
    case "$action" in
      bc)
        run_in_trainer 0 python -m quake_ai.train_bc --config configs/bc_e1m1_corpus.yaml --device "${QUAKE_AI_DEVICE}" "$@"
        ;;
      ppo)
        run_in_trainer 0 python -m quake_ai.train_rl --config configs/ppo_e1m1_corpus.yaml --device "${QUAKE_AI_DEVICE}" "$@"
        ;;
      eval)
        run_in_trainer 0 python -m quake_ai.eval --config configs/eval_e1m1_corpus.yaml --device "${QUAKE_AI_DEVICE}" "$@"
        ;;
      distill)
        run_in_trainer 0 python -m quake_ai.train_distill --config configs/distill_e1m1_corpus.yaml --device "${QUAKE_AI_DEVICE}" "$@"
        ;;
      all)
        run_in_trainer 0 python -m quake_ai.train_bc --config configs/bc_e1m1_corpus.yaml --device "${QUAKE_AI_DEVICE}"
        run_in_trainer 0 python -m quake_ai.train_rl --config configs/ppo_e1m1_corpus.yaml --device "${QUAKE_AI_DEVICE}"
        run_in_trainer 0 python -m quake_ai.eval --config configs/eval_e1m1_corpus.yaml --device "${QUAKE_AI_DEVICE}"
        run_in_trainer 0 python -m quake_ai.train_distill --config configs/distill_e1m1_corpus.yaml --device "${QUAKE_AI_DEVICE}"
        ;;
      *)
        echo "unknown e1m1-corpus action: $action" >&2
        exit 1
        ;;
    esac
    ;;
  e1m1-world|e1m1-world-verify)
    shift
    require_target_prereqs
    action="${1:-all}"
    shift || true
    profile="corpus"
    if [ "${COMMAND}" = "e1m1-world-verify" ]; then
      profile="verify"
    fi
    run_in_trainer 1 python -m quake_ai.live_training --profile "${profile}" --action "${action}" --device "${QUAKE_AI_DEVICE}" "$@"
    ;;
  combat-verify|combat|combat-bot-verify|combat-bot|combat-bot-*)
    shift
    require_target_prereqs
    action="${1:-all}"
    shift || true
    run_in_trainer 1 python -m quake_ai.live_training --profile "${COMMAND}" --action "${action}" --device "${QUAKE_AI_DEVICE}" "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "unknown command: $1" >&2
    usage >&2
    exit 1
    ;;
esac

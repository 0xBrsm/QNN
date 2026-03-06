#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_SRC_ROOT="${HOST_SRC_ROOT:-$SRC_ROOT}"
HOST_ARTIFACT_ROOT="${HOST_ARTIFACT_ROOT:-$(dirname "$HOST_SRC_ROOT")/artifacts}"
TRAINING_AMD_TARGET="${TRAINING_AMD_TARGET:-auto}"
TRAINING_IMAGE_TAG="${TRAINING_IMAGE_TAG:-quake-ai-trainer-amd:local}"
QUAKE_AI_DEVICE="${QUAKE_AI_DEVICE:-gpu}"
WSL_LIBDXCORE_PATH="${WSL_LIBDXCORE_PATH:-/usr/lib/wsl/lib/libdxcore.so}"
WSL_HSA_RUNTIME_PATH="${WSL_HSA_RUNTIME_PATH:-/opt/rocm/lib/libhsa-runtime64.so.1}"

detect_target() {
  case "${TRAINING_AMD_TARGET}" in
    wsl|amd-wsl)
      printf 'wsl\n'
      return 0
      ;;
    rocm|linux|amd-rocm)
      printf 'rocm\n'
      return 0
      ;;
    auto)
      ;;
    *)
      echo "unsupported TRAINING_AMD_TARGET: ${TRAINING_AMD_TARGET}" >&2
      exit 1
      ;;
  esac

  if docker run --rm --device /dev/dxg ubuntu:24.04 bash -lc 'test -e /dev/dxg' >/dev/null 2>&1; then
    printf 'wsl\n'
    return 0
  fi
  printf 'rocm\n'
}

TRAINING_TARGET="$(detect_target)"
case "${TRAINING_TARGET}" in
  wsl)
    COMPOSE_FILE="${SRC_ROOT}/docker/training/compose.amd-wsl.yaml"
    ;;
  rocm)
    COMPOSE_FILE="${SRC_ROOT}/docker/training/compose.amd-rocm.yaml"
    ;;
esac

export HOST_SRC_ROOT HOST_ARTIFACT_ROOT TRAINING_AMD_TARGET TRAINING_IMAGE_TAG QUAKE_AI_DEVICE WSL_LIBDXCORE_PATH WSL_HSA_RUNTIME_PATH

compose() {
  docker compose -f "$COMPOSE_FILE" "$@"
}

usage() {
  cat <<'EOF'
Usage:
  scripts/train-container.sh build
  scripts/train-container.sh check
  scripts/train-container.sh target
  scripts/train-container.sh shell
  scripts/train-container.sh migrate-legacy
  scripts/train-container.sh run -- <command...>
  scripts/train-container.sh e1m1-corpus {bc|ppo|eval|distill|all}

Environment:
  HOST_SRC_ROOT       Host path to the publishable src repo. Defaults to the current src root.
  HOST_ARTIFACT_ROOT  Host path for writable corpus and run artifacts. Defaults to a sibling 'artifacts' directory.
  TRAINING_AMD_TARGET AMD runtime target: auto, wsl, or rocm. Defaults to auto.
  TRAINING_IMAGE_TAG  Docker image tag for the trainer image.
  QUAKE_AI_DEVICE     Torch device request inside the trainer. Defaults to 'gpu'.
  WSL_LIBDXCORE_PATH  Host path to libdxcore.so for AMD ROCm on WSL.
  WSL_HSA_RUNTIME_PATH Host path to libhsa-runtime64.so.1 for AMD ROCm on WSL.

Examples:
  scripts/train-container.sh build
  scripts/train-container.sh check
  scripts/train-container.sh target
  scripts/train-container.sh migrate-legacy
  scripts/train-container.sh run -- python -m quake_ai.check_accelerator --device gpu
  scripts/train-container.sh e1m1-corpus all
EOF
}

run_in_trainer() {
  compose run --rm trainer-amd "$@"
}

case "${1:-help}" in
  target)
    shift
    printf '%s\n' "${TRAINING_TARGET}"
    ;;
  build)
    shift
    compose build trainer-amd "$@"
    ;;
  check)
    shift
    run_in_trainer python -m quake_ai.check_accelerator --device "${1:-$QUAKE_AI_DEVICE}"
    ;;
  shell)
    shift
    run_in_trainer bash "$@"
    ;;
  migrate-legacy)
    shift
    mkdir -p "${HOST_ARTIFACT_ROOT}/runs" "${HOST_ARTIFACT_ROOT}/corpus"
    if [ -d "${HOST_SRC_ROOT}/runs" ]; then
      rsync -a --remove-source-files "${HOST_SRC_ROOT}/runs/" "${HOST_ARTIFACT_ROOT}/runs/"
      find "${HOST_SRC_ROOT}/runs" -type d -empty -delete
    fi
    if [ -d "${HOST_SRC_ROOT}/corpus" ]; then
      rsync -a --remove-source-files "${HOST_SRC_ROOT}/corpus/" "${HOST_ARTIFACT_ROOT}/corpus/"
      find "${HOST_SRC_ROOT}/corpus" -type d -empty -delete
    fi
    ;;
  run)
    shift
    if [ "${1:-}" = "--" ]; then
      shift
    fi
    if [ "$#" -eq 0 ]; then
      echo "run requires a command" >&2
      exit 1
    fi
    run_in_trainer "$@"
    ;;
  e1m1-corpus)
    shift
    action="${1:-all}"
    shift || true
    case "$action" in
      bc)
        run_in_trainer python -m quake_ai.train_bc --config configs/bc_e1m1_corpus.yaml --device "${QUAKE_AI_DEVICE}" "$@"
        ;;
      ppo)
        run_in_trainer python -m quake_ai.train_rl --config configs/ppo_e1m1_corpus.yaml --device "${QUAKE_AI_DEVICE}" "$@"
        ;;
      eval)
        run_in_trainer python -m quake_ai.eval --config configs/eval_e1m1_corpus.yaml --device "${QUAKE_AI_DEVICE}" "$@"
        ;;
      distill)
        run_in_trainer python -m quake_ai.train_distill --config configs/distill_e1m1_corpus.yaml --device "${QUAKE_AI_DEVICE}" "$@"
        ;;
      all)
        run_in_trainer python -m quake_ai.train_bc --config configs/bc_e1m1_corpus.yaml --device "${QUAKE_AI_DEVICE}"
        run_in_trainer python -m quake_ai.train_rl --config configs/ppo_e1m1_corpus.yaml --device "${QUAKE_AI_DEVICE}"
        run_in_trainer python -m quake_ai.eval --config configs/eval_e1m1_corpus.yaml --device "${QUAKE_AI_DEVICE}"
        run_in_trainer python -m quake_ai.train_distill --config configs/distill_e1m1_corpus.yaml --device "${QUAKE_AI_DEVICE}"
        ;;
      *)
        echo "unknown e1m1-corpus action: $action" >&2
        exit 1
        ;;
    esac
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

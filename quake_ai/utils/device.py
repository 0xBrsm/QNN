"""Torch device selection and runtime reporting."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass(frozen=True, slots=True)
class TorchDeviceSpec:
    requested: str
    resolved: str
    backend: str

    @property
    def device(self) -> torch.device:
        return torch.device(self.resolved)


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


def _cuda_backend() -> str:
    return "rocm" if getattr(torch.version, "hip", None) else "cuda"


def resolve_torch_device(requested: str | None = None) -> TorchDeviceSpec:
    raw = (requested or os.environ.get("QUAKE_AI_DEVICE") or "auto").strip().lower()
    if not raw:
        raw = "auto"

    if raw == "auto":
        if torch.cuda.is_available():
            return TorchDeviceSpec(requested="auto", resolved="cuda", backend=_cuda_backend())
        if _mps_available():
            return TorchDeviceSpec(requested="auto", resolved="mps", backend="mps")
        return TorchDeviceSpec(requested="auto", resolved="cpu", backend="cpu")

    if raw == "gpu":
        if torch.cuda.is_available():
            return TorchDeviceSpec(requested="gpu", resolved="cuda", backend=_cuda_backend())
        if _mps_available():
            return TorchDeviceSpec(requested="gpu", resolved="mps", backend="mps")
        raise RuntimeError("Requested a GPU device, but no supported accelerator is available")

    if raw == "rocm":
        raw = "cuda"

    if raw.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA/ROCm device, but torch.cuda.is_available() is false")
        return TorchDeviceSpec(requested=requested or raw, resolved=raw, backend=_cuda_backend())

    if raw == "mps":
        if not _mps_available():
            raise RuntimeError("Requested MPS device, but torch.backends.mps.is_available() is false")
        return TorchDeviceSpec(requested="mps", resolved="mps", backend="mps")

    if raw == "cpu":
        return TorchDeviceSpec(requested="cpu", resolved="cpu", backend="cpu")

    raise ValueError(f"Unsupported torch device request: {requested}")


def configure_torch_runtime(spec: TorchDeviceSpec) -> None:
    torch.set_float32_matmul_precision("high")
    if spec.backend == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def describe_torch_runtime(requested: str | None = None) -> Dict[str, Any]:
    try:
        spec = resolve_torch_device(requested)
        error = None
    except Exception as exc:  # pragma: no cover - only hit when runtime is misconfigured
        spec = TorchDeviceSpec(requested=requested or "auto", resolved="unavailable", backend="unavailable")
        error = str(exc)

    summary: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "requested_device": spec.requested,
        "resolved_device": spec.resolved,
        "backend": spec.backend,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_version": torch.version.cuda,
        "hip_version": getattr(torch.version, "hip", None),
        "mps_available": _mps_available(),
    }
    if error is not None:
        summary["error"] = error
    if torch.cuda.is_available():
        summary["devices"] = [
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
            }
            for idx in range(torch.cuda.device_count())
        ]
    return summary

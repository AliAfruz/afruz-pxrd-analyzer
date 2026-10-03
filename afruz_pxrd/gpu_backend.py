"""Optional GPU services with a strict, transparent CPU fallback.

Afruz PXRD deliberately imports CuPy lazily.  The application must continue to
start on computers without an NVIDIA GPU, while machines with a compatible
CUDA installation can accelerate the large structure-factor matrix products.
"""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import tempfile

import numpy as np


_DLL_DIRECTORY_HANDLES: list[object] = []


def _prepare_cuda_environment() -> None:
    """Expose an existing Windows CUDA toolkit without installing another one."""
    if not os.environ.get("CUPY_CACHE_DIR"):
        local = os.environ.get("LOCALAPPDATA")
        root = Path(local) if local else Path(tempfile.gettempdir())
        cache = root / "AfruzPXRD" / "CuPyCache"
        try:
            cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            cache = Path(tempfile.gettempdir()) / "AfruzPXRD-CuPyCache"
            cache.mkdir(parents=True, exist_ok=True)
        os.environ["CUPY_CACHE_DIR"] = str(cache)

    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    candidates = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
    toolkit_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit_root.is_dir():
        candidates.extend(
            sorted(toolkit_root.glob("v*/bin"), reverse=True)
        )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        try:
            # The returned handle must stay alive for the directory to remain
            # in the DLL search path.
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(candidate)))
            return
        except OSError:
            continue


@lru_cache(maxsize=1)
def cuda_status() -> dict:
    """Return a JSON-safe CUDA status record after a real compute smoke test."""
    status = {
        "available": False,
        "backend": "CPU",
        "device_count": 0,
        "device_name": "",
        "total_memory_mb": None,
        "cupy_version": None,
        "cuda_runtime_version": None,
        "reason": "CUDA was not requested.",
    }
    try:
        _prepare_cuda_environment()
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 1:
            status["reason"] = "No CUDA-capable NVIDIA device was found."
            return status
        device = cp.cuda.Device(0)
        device.use()
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties.get("name", "NVIDIA CUDA GPU")
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        # This is intentionally a real kernel launch, not only an import test.
        probe = cp.arange(256, dtype=cp.float64)
        float(cp.asnumpy(cp.sum(cp.sin(probe))))
        device.synchronize()
        memory = properties.get("totalGlobalMem")
        status.update(
            available=True,
            backend="CUDA (CuPy)",
            device_count=count,
            device_name=str(name),
            total_memory_mb=(None if memory is None else round(float(memory) / 2**20, 1)),
            cupy_version=str(cp.__version__),
            cuda_runtime_version=int(cp.cuda.runtime.runtimeGetVersion()),
            reason="CUDA compute smoke test passed.",
        )
    except Exception as exc:  # optional dependency/device: CPU fallback is valid
        status["reason"] = f"CUDA unavailable; using CPU: {exc}"
    return status


def compute_backend(use_gpu: bool):
    """Return ``(array_module, status)`` without ever making GPU mandatory."""
    if not use_gpu:
        return np, {
            "available": False,
            "backend": "CPU",
            "device_count": 0,
            "device_name": "",
            "total_memory_mb": None,
            "cupy_version": None,
            "cuda_runtime_version": None,
            "reason": "GPU acceleration is disabled for this run.",
        }
    status = dict(cuda_status())
    if not status["available"]:
        return np, status
    import cupy as cp

    return cp, status


def release_gpu_memory() -> None:
    """Release cached CuPy blocks after a large calculation when possible."""
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

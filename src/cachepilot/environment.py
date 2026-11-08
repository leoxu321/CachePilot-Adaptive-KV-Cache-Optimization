"""Environment inspection works even when inference dependencies are absent."""

import ctypes
import importlib.metadata
import os
import platform
import subprocess
import sys


def system_ram_bytes():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "total_phys", "avail_phys", "total_page", "avail_page",
                    "total_virtual", "avail_virtual", "avail_extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.total_phys
        return None
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def inspect_environment():
    versions = {}
    for package in ("cachepilot", "torch", "transformers", "accelerate", "bitsandbytes",
                    "huggingface-hub", "tokenizers", "safetensors", "numpy", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {"os": platform.platform(), "python": sys.version,
              "executable": sys.executable, "machine": platform.machine(),
              "cpu": platform.processor(), "system_ram_bytes": system_ram_bytes(),
              "packages": versions, "cuda_available": False,
              "environment": {k: os.environ.get(k) for k in (
                  "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED",
                  "HF_HUB_OFFLINE", "PYTORCH_CUDA_ALLOC_CONF")}}
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        result["nvidia_smi"] = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["nvidia_smi"] = str(exc)
    try:
        import torch
        result["torch_cuda_runtime"] = torch.version.cuda
        result["cudnn_version"] = torch.backends.cudnn.version()
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            result["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory,
                             "free_memory_bytes": free, "runtime_total_bytes": total,
                             "compute_capability": list(torch.cuda.get_device_capability(0)),
                             "uuid": str(getattr(props, "uuid", "unknown"))}
    except (ImportError, RuntimeError, OSError) as exc:
        result["torch_error"] = str(exc)
    result["missing_inference_packages"] = [p for p in ("torch", "transformers", "accelerate", "bitsandbytes") if versions[p] is None]
    result["inference_ready"] = result["cuda_available"] and not result["missing_inference_packages"]
    result["guidance"] = ("CUDA is ready; native Windows is supported. WSL2 is optional."
        if result["cuda_available"] else
        "Install CUDA PyTorch in the project venv. nvidia-smi alone does not prove PyTorch CUDA support. "
        "Try WSL2 Ubuntu only if native Windows CUDA/bitsandbytes fails; rerun doctor there.")
    if result["missing_inference_packages"]:
        result["guidance"] += " Missing inference packages: " + ", ".join(result["missing_inference_packages"])
    return result


def comparison_environment(env):
    """Stable identity; exclude free VRAM, process paths, and transient utilization."""
    gpu = env.get("gpu", {})
    driver_rows = env.get("nvidia_smi", "").splitlines()
    drivers = [row.split(",")[3].strip() for row in driver_rows if len(row.split(",")) >= 6]
    return {"os": env["os"], "python": env["python"], "cpu": env["cpu"],
            "packages": env["packages"], "cuda_runtime": env.get("torch_cuda_runtime"),
            "cudnn": env.get("cudnn_version"), "drivers": drivers,
            "gpu": {k: gpu.get(k) for k in ("name", "total_memory_bytes", "compute_capability", "uuid")},
            "environment": env["environment"]}

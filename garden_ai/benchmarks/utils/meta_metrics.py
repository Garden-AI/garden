"""Meta-level benchmark metrics utilities.

Shared utilities for collecting hardware info, estimating costs, and extracting
model metadata that can be reused across different benchmark implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List

# GPU hourly cost estimates (USD) - Modal pricing (https://modal.com/pricing)
GPU_HOURLY_COSTS = {
    "B200": 6.25,  # $0.001736/sec
    "H200": 4.54,  # $0.001261/sec
    "H100": 3.95,  # $0.001097/sec
    "A100-80GB": 2.50,  # $0.000694/sec (80GB variant)
    "A100": 2.10,  # $0.000583/sec (40GB variant)
    "L40S": 1.95,  # $0.000542/sec
    "A10": 1.10,  # $0.000306/sec
    "L4": 0.80,  # $0.000222/sec
    "T4": 0.59,  # $0.000164/sec
    "default": 2.00,  # Fallback for unknown GPUs
}

# Model name inference from package names
MODEL_PACKAGE_NAMES = {
    "mace": "MACE",
    "mattersim": "MatterSim",
    "sevennet": "SevenNet",
    "chgnet": "CHGNet",
    "equiformer": "EquiformerV2",
    "orb": "Orb",
    "m3gnet": "M3GNet",
    "alignn": "ALIGNN",
}


def get_hardware_info() -> Dict[str, Any]:
    """Collect hardware information about the execution environment.

    Returns:
        Dictionary containing:
        - device_type: "cuda", "mps", or "cpu"
        - num_gpus: Number of GPUs available
        - gpu_names: List of GPU names
        - gpu_memory_gb: Memory of first GPU in GB (if available)
    """
    info = {"device_type": "cpu", "num_gpus": 0, "gpu_names": [], "gpu_memory_gb": None}
    try:
        import torch

        if torch.cuda.is_available():
            info["device_type"] = "cuda"
            info["num_gpus"] = torch.cuda.device_count()
            info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(info["num_gpus"])
            ]
            if info["num_gpus"] > 0:
                props = torch.cuda.get_device_properties(0)
                info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 1)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["device_type"] = "mps"
    except ImportError:
        pass
    return info


def get_gpu_hourly_cost(gpu_name: str) -> float:
    """Estimate hourly cost for a GPU based on its name.

    Args:
        gpu_name: GPU name string (e.g., "NVIDIA A100-SXM4-40GB")

    Returns:
        Estimated hourly cost in USD
    """
    gpu_name_upper = gpu_name.upper()
    for key in GPU_HOURLY_COSTS:
        if key != "default" and key.upper() in gpu_name_upper:
            return GPU_HOURLY_COSTS[key]
    return GPU_HOURLY_COSTS["default"]


def extract_model_info(model_packages: str | List[str]) -> Dict[str, Any]:
    """Extract model info from package specification.

    Args:
        model_packages: Package name(s) used to install the model

    Returns:
        Dictionary containing:
        - model_name: Inferred model name or "unknown"
        - model_packages: List of package names
    """
    packages = model_packages if isinstance(model_packages, list) else [model_packages]
    model_name = "unknown"
    for pkg in packages:
        pkg_lower = pkg.lower()
        for key, name in MODEL_PACKAGE_NAMES.items():
            if key in pkg_lower:
                model_name = name
                break
        if model_name != "unknown":
            break
    return {"model_name": model_name, "model_packages": packages}


def calculate_run_metadata(
    hardware_info: Dict[str, Any],
    model_info: Dict[str, Any],
    total_elapsed: float,
    num_workers: int,
    num_structures_total: int,
    num_structures_processed: int,
) -> Dict[str, Any]:
    """Calculate run metadata including timing, cost, and hardware info.

    Args:
        hardware_info: Output from get_hardware_info()
        model_info: Output from extract_model_info()
        total_elapsed: Total benchmark runtime in seconds
        num_workers: Number of worker processes used
        num_structures_total: Total structures in dataset
        num_structures_processed: Structures processed in this run

    Returns:
        Complete run_metadata dictionary
    """
    throughput = num_structures_total / total_elapsed if total_elapsed > 0 else 0

    # Calculate cost estimate
    gpu_hourly_cost = (
        get_gpu_hourly_cost(hardware_info["gpu_names"][0])
        if hardware_info["gpu_names"]
        else 0
    )
    total_gpu_hours = (total_elapsed / 3600) * num_workers
    total_cost = total_gpu_hours * gpu_hourly_cost
    cost_per_1k = (
        (total_cost / num_structures_total) * 1000 if num_structures_total > 0 else 0
    )

    return {
        "model": model_info,
        "hardware": hardware_info,
        "timing": {
            "total_seconds": round(total_elapsed, 2),
            "throughput_per_second": round(throughput, 3),
            "num_workers": num_workers,
        },
        "cost": {
            "gpu_hourly_rate_usd": gpu_hourly_cost,
            "total_gpu_hours": round(total_gpu_hours, 4),
            "estimated_cost_usd": round(total_cost, 4),
            "estimated_cost_per_1000_structures_usd": round(cost_per_1k, 4),
        },
        "dataset": {
            "num_structures_total": num_structures_total,
            "num_structures_processed": num_structures_processed,
        },
    }

#!/usr/bin/env python
"""
Resource snapshot for Docker runs.
Reports CPU, memory, disk, cgroup limits, and Intel XPU availability.
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path

import psutil


def _parse_cpuset(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    raw = path.read_text().strip()
    if not raw:
        return None
    cpus: list[int] = []
    for part in raw.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def _cgroup_memory_limit() -> int | None:
    # cgroup v2
    v2_path = Path("/sys/fs/cgroup/memory.max")
    if v2_path.exists():
        raw = v2_path.read_text().strip()
        if raw and raw != "max":
            try:
                return int(raw)
            except ValueError:
                pass
    # cgroup v1
    v1_path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1_path.exists():
        try:
            val = int(v1_path.read_text().strip())
            if val > 0 and val < 2**60:  # ignore "no limit" huge defaults
                return val
        except ValueError:
            pass
    return None


def _format_bytes(num: int) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if num < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


def gpu_info() -> dict:
    info = {
        "torch": None,
        "xpu_available": False,
        "xpu_count": 0,
        "xpu_names": [],
        "smoke": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            info["xpu_available"] = True
            info["xpu_count"] = torch.xpu.device_count()
            info["xpu_names"] = [
                torch.xpu.get_device_name(i) for i in range(info["xpu_count"])
            ]
            try:
                x = torch.randn(1, device="xpu:0")
                info["smoke"] = float(x.sum().item())
            except Exception as exc:  # pragma: no cover - best effort
                info["smoke"] = f"smoke check failed: {exc}"
    except Exception as exc:  # pragma: no cover - best effort
        info["torch"] = f"unavailable ({exc})"
    return info


def resource_report() -> dict:
    vm = psutil.virtual_memory()
    disk_root = Path("/data") if Path("/data").exists() else Path("/workspace")
    disk = shutil.disk_usage(disk_root)
    cpuset = _parse_cpuset(Path("/sys/fs/cgroup/cpuset.cpus.effective")) or _parse_cpuset(
        Path("/sys/fs/cgroup/cpuset/cpuset.cpus")
    )
    cg_mem = _cgroup_memory_limit()

    return {
        "platform": platform.platform(),
        "cpu": {
            "logical": psutil.cpu_count(logical=True),
            "physical": psutil.cpu_count(logical=False),
            "cpuset": cpuset,
        },
        "memory": {
            "total": vm.total,
            "cgroup_limit": cg_mem,
        },
        "disk": {
            "mount": str(disk_root),
            "total": disk.total,
            "free": disk.free,
        },
        "env": {
            "ZE_AFFINITY_MASK": os.environ.get("ZE_AFFINITY_MASK"),
            "ZE_ENABLE_PCI_ID_DEVICE_ORDER": os.environ.get(
                "ZE_ENABLE_PCI_ID_DEVICE_ORDER"
            ),
        },
        "gpu": gpu_info(),
    }


def print_report(data: dict) -> None:
    def fmt(obj):
        if obj is None:
            return "n/a"
        if isinstance(obj, int):
            return str(obj)
        return obj

    cpu = data["cpu"]
    mem = data["memory"]
    disk = data["disk"]
    gpu = data["gpu"]
    env = data["env"]

    print("=== Supereight Resource Report ===")
    print(f"Platform       : {data['platform']}")
    print(
        f"CPU            : logical={fmt(cpu['logical'])} physical={fmt(cpu['physical'])} cpuset={cpu['cpuset'] or 'unrestricted'}"
    )
    if mem["cgroup_limit"]:
        print(
            f"Memory         : { _format_bytes(mem['total']) } (cgroup limit {_format_bytes(mem['cgroup_limit'])})"
        )
    else:
        print(f"Memory         : { _format_bytes(mem['total']) }")
    print(
        f"Disk           : mount={disk['mount']} free={_format_bytes(disk['free'])} / total={_format_bytes(disk['total'])}"
    )
    print(
        f"Env            : ZE_AFFINITY_MASK={env['ZE_AFFINITY_MASK'] or 'unset'}, ZE_ENABLE_PCI_ID_DEVICE_ORDER={env['ZE_ENABLE_PCI_ID_DEVICE_ORDER'] or 'unset'}"
    )
    print(
        f"Intel XPU      : available={gpu['xpu_available']} count={gpu['xpu_count']} names={gpu['xpu_names']}"
    )
    if gpu["smoke"] is not None:
        print(f"XPU smoke      : {gpu['smoke']}")
    if isinstance(gpu["torch"], str):
        print(f"Torch          : {gpu['torch']}")
    else:
        print(f"Torch          : {gpu['torch']}")


def main() -> int:
    data = resource_report()
    print_report(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Resource discovery and parallelism planning for Supereight.

Usage:
    python -m supereight.resources
    python - <<'PY'
    from supereight.resources import inspect_system, print_human, plan_parallelism
    info = inspect_system()
    print_human(info)
    print(plan_parallelism(info))
    PY
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

import psutil


def _try_torch_xpu():
    out = {
        "available": False,
        "count": 0,
        "names": [],
        "version": None,
        "error": None,
    }
    try:
        import torch  # type: ignore

        out["version"] = torch.__version__
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            out["available"] = True
            out["count"] = torch.xpu.device_count()
            out["names"] = [torch.xpu.get_device_name(i) for i in range(out["count"])]
    except Exception as exc:  # pragma: no cover - best effort
        out["error"] = str(exc)
    return out


def _cgroup_cpuset():
    for path in (
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpuset/cpuset.cpus",
    ):
        try:
            raw = open(path).read().strip()
        except FileNotFoundError:
            continue
        if not raw:
            continue
        cpus: List[int] = []
        for part in raw.split(","):
            if "-" in part:
                start, end = part.split("-", 1)
                cpus.extend(range(int(start), int(end) + 1))
            else:
                cpus.append(int(part))
        return sorted(set(cpus))
    return None


def _cgroup_mem_limit() -> Optional[int]:
    for path in (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    ):
        try:
            raw = open(path).read().strip()
        except FileNotFoundError:
            continue
        if not raw or raw == "max":
            continue
        try:
            val = int(raw)
            if val > 0 and val < 2**60:
                return val
        except ValueError:
            continue
    return None


@dataclass
class SystemInfo:
    platform: str
    cpu_logical: Optional[int]
    cpu_physical: Optional[int]
    cpu_cpuset: Optional[list]
    mem_total: Optional[int]
    mem_available: Optional[int]
    mem_cgroup_limit: Optional[int]
    xpu: dict


def inspect_system() -> SystemInfo:
    vm = psutil.virtual_memory()
    return SystemInfo(
        platform=platform.platform(),
        cpu_logical=psutil.cpu_count(logical=True),
        cpu_physical=psutil.cpu_count(logical=False),
        cpu_cpuset=_cgroup_cpuset(),
        mem_total=vm.total,
        mem_available=vm.available,
        mem_cgroup_limit=_cgroup_mem_limit(),
        xpu=_try_torch_xpu(),
    )


def plan_parallelism(info: SystemInfo, workers_override: Optional[int] = None) -> dict:
    # Default: one worker per XPU; fallback to CPU threads-1 (cap at 8)
    if info.xpu.get("available") and info.xpu.get("count", 0) > 0:
        count = info.xpu["count"]
        workers = workers_override or count
        devices = [{"type": "xpu", "index": i, "ze_affinity_mask": str(i)} for i in range(workers)]
        backend = "intel_xpu"
    else:
        logical = info.cpu_logical or 1
        # respect cpuset if present
        if info.cpu_cpuset:
            logical = min(logical, len(info.cpu_cpuset))
        workers = workers_override or max(1, min(8, logical - 1))
        devices = [{"type": "cpu", "index": i} for i in range(workers)]
        backend = "cpu"
    return {"backend": backend, "workers": workers, "devices": devices}


def _format_bytes(num: Optional[int]) -> str:
    if num is None:
        return "n/a"
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if num < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


def print_human(info: SystemInfo, plan: Optional[dict] = None) -> None:
    print("=== Supereight Doctor ===")
    print(f"Platform        : {info.platform}")
    print(
        f"CPU             : logical={info.cpu_logical} physical={info.cpu_physical} cpuset={info.cpu_cpuset or 'unrestricted'}"
    )
    mem_lim = (
        f" (cgroup limit {_format_bytes(info.mem_cgroup_limit)})"
        if info.mem_cgroup_limit
        else ""
    )
    print(
        f"Memory          : total={_format_bytes(info.mem_total)} available={_format_bytes(info.mem_available)}{mem_lim}"
    )
    xpu = info.xpu
    print(
        f"Intel XPU       : available={xpu.get('available')} count={xpu.get('count')} names={xpu.get('names')}"
    )
    print(f"Torch           : {xpu.get('version') or 'n/a'} (error={xpu.get('error')})")
    if plan:
        print(f"Backend         : {plan['backend']}")
        print(f"Workers         : {plan['workers']}")
        print(f"Devices         : {plan['devices']}")


def main(argv=None) -> int:
    info = inspect_system()
    plan = plan_parallelism(info)
    data = {"system": asdict(info), "plan": plan}
    # human
    print_human(info, plan)
    # json
    print("\n--- JSON ---")
    json.dump(data, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

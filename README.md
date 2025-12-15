# Super-Eight Home Movie Optimisation (Docker/XPU)

Fast subtitle and enhancement pipeline with Intel GPU/CPU acceleration, now packaged for Docker + Jupyter.

## Quick start (Docker)
Prereqs: Docker (with Compose), Intel GPU drivers (if using GPUs), access to `/dev/dri` on Linux hosts.

1) Build the image (Intel optimised PyTorch + Jupyter base):
```bash
docker compose build supereight
```
2) Put source videos in `./data/inputs` on the host (ignored by git).
3) Run the full pipeline (scene detect → face export → enhance → audio → subtitles):
```bash
docker compose run --rm supereight
```
Results land in `./data/processed` (scenes, faces, subtitles, enhanced clips).

### Jupyter for interactive work
```bash
docker compose up supereight-jupyter
```
Then open `http://localhost:8888` (token disabled inside the compose service; secure separately if needed).

### Verify GPU inside the container
```bash
docker compose run --rm supereight python -m supereight.resources
```
Expected: shows `Intel XPU available=True` with device names; also prints JSON summary.

### Service layout
- `supereight`: CLI runner (default command: `/workspace/scripts/run_pipeline.sh`).
- `supereight-jupyter`: Jupyter Lab on port 8888.

### Config knobs
Environment variables (can be overridden via `docker compose run -e NAME=value ...`):
- `SUPEREIGHT_INPUT` (default `/data/inputs`)
- `SUPEREIGHT_OUTPUT` (default `/data/processed`)
- `SUPEREIGHT_SCENES_FOLDER` (default `/data/processed/scenes`)
- `SUPEREIGHT_MODEL_CACHE` (default `/models`)
- `ZE_AFFINITY_MASK` (pin a process to a specific Intel GPU if desired)
- `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1` (stable device ordering)
 - `WORKERS` (override worker count; default is one per XPU, or CPU cores-1 if no GPU)

### Resource reporting
Every container start prints a resource snapshot (CPU, memory, disk, detected Intel XPUs, torch version). If XPUs are present, a small smoke tensor is executed on `xpu:0`.

## Notes on acceleration & parallelism
- Uses Intel’s `intel-optimized-pytorch` image with XPU support; falls back to CPU if no GPU is exposed to the container.
- Device selection is automatic via `torch.xpu.device_count()`. Multi-GPU runs pin workers using `ZE_AFFINITY_MASK` conventions inside `process_parallel.py`.
- CPU-only hosts still run, using multiple workers up to detected cgroup / cpuset limits.

## Legacy Windows entrypoints
The original PowerShell wrappers remain in `scripts/` for native Windows runs, but Docker is the recommended path going forward.

## Smoke test in Docker
Minimal check (runs resource doctor to validate container wiring and XPU availability):
```bash
docker compose run --rm supereight \
  /workspace/scripts/docker_entrypoint.sh \
  python -m supereight.resources
```

## Windows via Docker Desktop + WSL2
- Enable WSL2 integration in Docker Desktop.
- Ensure Intel GPU passthrough is enabled (DX11/WSL2 GPU sharing) and `/dev/dri` is visible inside Linux containers; otherwise the stack will run on CPU fallback.
- Place inputs under the repo path (e.g., `./data/inputs`) so they are volume-mounted into the container.

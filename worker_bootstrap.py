"""
Worker bootstrap for multiprocessing-based GPU affinity.

CRITICAL: This module sets environment variables BEFORE importing any GPU libraries.
This ensures proper GPU affinity pinning on Intel Arc and other hardware.

Design principles:
1. Set ZE_AFFINITY_MASK before any torch/GPU imports
2. Each worker process gets its own isolated environment
3. Log all environment settings for debugging
"""
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

# Set up logging before anything else
LOG = logging.getLogger("worker_bootstrap")


def set_gpu_affinity(device: Dict[str, Any], backend: str) -> None:
    """
    Set GPU affinity environment variables BEFORE importing GPU libraries.

    Args:
        device: Device descriptor with 'index', 'name', 'type'
        backend: Backend name ('intel_arc', 'cuda', 'cpu')
    """
    pid = os.getpid()
    device_index = device.get("index", 0)
    device_name = device.get("name", device.get("type", "unknown"))

    if backend == "intel_arc":
        # Set ZE_AFFINITY_MASK BEFORE any GPU imports
        # CRITICAL FIX: Pin to entire card, not tile on card 0
        # Format: <card_index> pins to all tiles on that card
        # - device_index=0 → ZE_AFFINITY_MASK=0 (all tiles on card 0)
        # - device_index=1 → ZE_AFFINITY_MASK=1 (all tiles on card 1)
        #
        # WRONG (old): "0.{device_index}" → "0.0" and "0.1" (both on card 0!)
        # RIGHT (new): "{device_index}" → "0" and "1" (separate cards)
        affinity_mask = str(device_index)
        os.environ["ZE_AFFINITY_MASK"] = affinity_mask

        LOG.info(
            "[worker-bootstrap] PID=%d | backend=%s | device_idx=%d (%s) | ZE_AFFINITY_MASK=%s | STATUS=set_before_import",
            pid,
            backend,
            device_index,
            device_name,
            affinity_mask,
        )

        # VERIFICATION: Check that affinity was honored (after model loads)
        # This will be called after model.to(device) in _load_model_for_device
        os.environ["_VERIFY_GPU_AFFINITY"] = "1"  # Signal to verify later
    elif backend == "cuda":
        # For CUDA, use CUDA_VISIBLE_DEVICES
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)

        LOG.info(
            "[worker-bootstrap] PID=%d | backend=%s | device_idx=%d (%s) | CUDA_VISIBLE_DEVICES=%s | STATUS=set_before_import",
            pid,
            backend,
            device_index,
            device_name,
            device_index,
        )
    else:
        LOG.info(
            "[worker-bootstrap] PID=%d | backend=%s (cpu) | no affinity needed",
            pid,
            backend,
        )


def worker_main(device: Dict[str, Any], scene_list: List[Path], backend: str, task: str, cli_args, progress_queue=None) -> None:
    """
    Main entry point for worker process.

    This function:
    1. Sets GPU affinity BEFORE any imports
    2. Imports GPU libraries AFTER affinity is set
    3. Processes assigned scenes
    4. Sends progress updates via Queue

    Args:
        device: Device descriptor
        scene_list: List of scenes to process
        backend: Backend name
        task: Task name ('subtitles', 'faces', etc.)
        cli_args: CLI arguments namespace
        progress_queue: Optional multiprocessing.Queue for progress updates
    """
    # Step 1: Set affinity BEFORE imports
    set_gpu_affinity(device, backend)

    # ========== PATCH #2: REGISTER CLEANUP HANDLERS ==========
    import signal
    import atexit

    # Store cleanup state in module-level var (accessible from signal handlers)
    _cleanup_state = {
        "model": None,
        "backend": backend,
        "device": device,
        "cleaned": False,
    }

    def cleanup_on_exit():
        """Cleanup handler called on normal exit or SIGTERM."""
        if _cleanup_state["cleaned"]:
            return  # Already cleaned

        LOG.info("[worker-cleanup] PID=%d | Running cleanup handler", os.getpid())

        model = _cleanup_state.get("model")
        if model is not None:
            try:
                # Import cleanup function (avoid circular import)
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "process_parallel_cleanup",
                    Path(__file__).parent / "process_parallel.py"
                )
                pp = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(pp)

                pp._cleanup_gpu_resources(model, backend, device)
                LOG.info("[worker-cleanup] PID=%d | GPU resources cleaned successfully", os.getpid())
            except Exception as exc:
                LOG.error("[worker-cleanup] PID=%d | Cleanup failed: %s", os.getpid(), exc)

        _cleanup_state["cleaned"] = True

    def signal_handler(signum, frame):
        """Handle SIGTERM gracefully."""
        LOG.warning("[worker-cleanup] PID=%d | Received signal %d, cleaning up...", os.getpid(), signum)
        cleanup_on_exit()
        sys.exit(1)  # Exit with error code to signal termination

    # Register handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C too
    atexit.register(cleanup_on_exit)

    LOG.info("[worker-bootstrap] PID=%d | Cleanup handlers registered", os.getpid())
    # ========== END PATCH #2 ==========

    # Step 2: Set up logging in the worker process
    # Each worker process needs its own logging configuration
    # Use the same configure_logging function from process_parallel to ensure file logging works
    import logging
    import importlib.util
    
    # Import process_parallel module (but don't execute GPU imports yet)
    # We need to configure logging first, but we also need configure_logging from the module
    # So we'll import it, configure logging, then the module will be fully loaded
    process_parallel_path = Path(__file__).parent / "process_parallel.py"
    spec = importlib.util.spec_from_file_location(
        "process_parallel_worker",
        process_parallel_path
    )
    process_parallel = importlib.util.module_from_spec(spec)
    
    # Configure logging BEFORE executing the module (to avoid GPU imports)
    # But we need configure_logging from the module... so we'll configure it after import
    # Actually, we can configure logging manually here to match the main process
    log_file = getattr(cli_args, 'log_file', None)
    verbosity = getattr(cli_args, 'verbose', 0)
    
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Console handler (respects verbosity)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)  # Root logger accepts all, handlers filter
    
    # File handler (logs all levels if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Step 3: NOW it's safe to import process_parallel module
    # The module-level imports will happen AFTER we set ZE_AFFINITY_MASK
    # Import as late as possible to ensure env is set first
    # Note: We import the module, not individual functions, to get fresh module state
    spec.loader.exec_module(process_parallel)

    # Set the global _cli_args so process_scene has access to it
    process_parallel._cli_args = cli_args

    # Step 4: Process scenes with properly configured GPU affinity
    LOG.info(
        "[worker-process] PID=%d | Processing %d scenes on %s",
        os.getpid(),
        len(scene_list),
        device.get("name", "device"),
    )

    # Pass progress_queue to process_batch
    process_parallel.process_batch(device, scene_list, backend, task, progress_queue)

    # ========== PATCH #2: STORE MODEL FOR CLEANUP ==========
    # Store model reference so signal handlers can access it
    if hasattr(process_parallel, '_worker_model'):
        _cleanup_state["model"] = process_parallel._worker_model
    # ========== END PATCH #2 ==========

    LOG.info(
        "[worker-process] PID=%d | Completed %d scenes",
        os.getpid(),
        len(scene_list),
    )

"""
Dual-GPU scene processing template (subtitles first) with live resource/ETA display.

Set ZE_AFFINITY_MASK to pin each worker to an Intel Arc GPU.
Usage:
  python process_parallel.py
Optional (Windows GPU telemetry): pip install wmi
"""

from pathlib import Path
from threading import Event
from threading import Lock
import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
import urllib.request
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import psutil
import importlib.util
import platform
import multiprocessing

# Optional: Windows GPU telemetry via WMI (works on recent Intel Arc drivers)
try:
    import wmi  # type: ignore

    _HAS_WMI = True
except Exception as e:
    _HAS_WMI = False
    _WMI_ERROR = str(e)  # Store error for debugging


DEFAULT_SCENES_FOLDER = Path(
    os.environ.get("SUPEREIGHT_SCENES_FOLDER", "/data/processed/scenes")
).resolve()
MODEL_CACHE = Path.home() / ".cache" / "connor_family_models"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
# SCRFD: Better accuracy than YuNet, optimized for Arc GPUs
FACE_DETECT_MODEL_NAME = "scrfd_2.5g_bnkps.onnx"
FACE_DETECT_MODEL_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_scrfd/scrfd_2.5g_bnkps.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_scrfd/scrfd_2.5g_bnkps.onnx",
    # Mirrors (Hugging Face)
    "https://huggingface.co/OwlMaster/AllFilesRope/resolve/d783e61585b3d83a85c91ca8a3b299e8ade94d72/scrfd_2.5g_bnkps.onnx?download=true",
]
# InsightFace ArcFace r100: Much better recognition accuracy than SFace (512-dim vs 128-dim)
# Try multiple sources - will fall back to SFace if ArcFace unavailable
FACE_RECOG_MODEL_NAME = "arcface_r100.onnx"
FACE_RECOG_MODEL_URLS = [
    # Try ONNX Model Zoo first
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcface_r100.onnx",
    # Fallback to SFace (still good, just not as accurate)
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    # Mirrors (Hugging Face)
    "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx?download=true",
]
# Optional checksum verification for downloaded models
MODEL_CHECKSUMS = {
    FACE_RECOG_MODEL_NAME: "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    FACE_DETECT_MODEL_NAME: "bc24bb349491481c3ca793cf89306723162c280cb284c5a5e49df3760bf5c2ce",
    "face_recognition_sface_2021dec.onnx": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    # Add known hashes for other models when available
}

# Track which model we're actually using
_RECOG_MODEL_DIM = 128  # Default to SFace dimensions
FACE_GALLERY: Dict[str, List[float]] = {}
_progress_lock = Lock()
_processed = 0
LOG = logging.getLogger("process_parallel")
_cli_args = None
_worker_model = None  # PATCH #2: Store model for signal handler cleanup
# Cache for codec availability checks
_CODEC_CACHE: Dict[str, bool] = {}


def _detect_gpus_wmi():
    """Lightweight GPU inventory via WMI (Windows only)."""
    if not _HAS_WMI:
        LOG.debug("WMI not available - install with: pip install wmi")
        return []
    try:
        c = wmi.WMI()
        gpus = []
        for idx, gpu in enumerate(c.Win32_VideoController()):
            name = (gpu.Name or "").strip()
            if not name:
                continue
            lower = name.lower()
            # Filter out virtual/remote display adapters
            if any(
                virtual in lower
                for virtual in [
                    "microsoft remote",
                    "remote display",
                    "virtual",
                    "vmware",
                    "virtualbox",
                ]
            ):
                LOG.debug("Skipping virtual adapter: %s", name)
                continue
            if "intel" in lower:
                vendor = "intel"
            elif "nvidia" in lower:
                vendor = "nvidia"
            else:
                vendor = "other"
            gpus.append({"index": idx, "name": name, "vendor": vendor})
            LOG.debug("Detected GPU %d: %s (vendor: %s)", idx, name, vendor)
        return gpus
    except Exception as e:
        LOG.warning("WMI GPU detection failed: %s", e)
        return []


def _detect_gpus_torch():
    """Fallback GPU detection via torch (CUDA/XPU) when WMI is unavailable or incomplete."""
    gpus = []
    try:
        import importlib
        torch = importlib.import_module("torch")
    except Exception as exc:
        LOG.debug("Torch not available for GPU detection: %s", exc)
        return gpus

    # CUDA
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(idx)
                gpus.append({"index": idx, "name": name, "vendor": "nvidia"})
    except Exception as exc:
        LOG.debug("CUDA detection failed: %s", exc)

    # Intel XPU (IPEX)
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            for idx in range(torch.xpu.device_count()):
                name = f"Intel XPU {idx}"
                gpus.append({"index": idx, "name": name, "vendor": "intel"})
    except Exception as exc:
        LOG.debug("XPU detection failed: %s", exc)

    return gpus


def detect_hardware():
    """Discover available CPU/GPU resources and return a simple descriptor."""
    cpu_count = psutil.cpu_count(logical=True) or 1
    gpus = _detect_gpus_wmi()
    if not gpus:
        LOG.debug("WMI GPU detection empty, falling back to torch-based probe")
        gpus = _detect_gpus_torch()
    LOG.debug("Raw GPU detection: %d GPUs found", len(gpus))

    # Filter out integrated GPUs (UHD Graphics, Iris, etc.) - only use discrete GPUs
    discrete_gpus = [
        g
        for g in gpus
        if not any(
            igpu in g["name"].lower()
            for igpu in ["uhd graphics", "iris", "hd graphics", "xe graphics"]
        )
    ]
    LOG.debug("After filtering integrated GPUs: %d discrete GPUs", len(discrete_gpus))

    # Filter Intel Arc GPUs (discrete only)
    intel_arc = [
        g
        for g in discrete_gpus
        if g["vendor"] == "intel" and "arc" in g["name"].lower()
    ]
    LOG.debug("Intel Arc GPUs: %d found", len(intel_arc))

    # Re-index Arc GPUs starting from 0 (for ZE_AFFINITY_MASK)
    for idx, gpu in enumerate(intel_arc):
        gpu["ze_index"] = idx  # Zero-based index for Level Zero
        LOG.debug(
            "Arc GPU %d: WMI index %d -> ze_index %d (%s)",
            idx,
            gpu["index"],
            gpu["ze_index"],
            gpu["name"],
        )

    return {
        "cpu_count": cpu_count,
        "gpus": discrete_gpus,  # Only discrete GPUs
        "intel_arc": intel_arc,
        "nvidia": [g for g in discrete_gpus if g["vendor"] == "nvidia"],
    }


def select_backend(hw):
    """
    Pick a processing backend based on discovered hardware.

    - Prefer Intel Arc (Level Zero) since the pipeline is tuned for it.
    - Fall back to CUDA (placeholder for future users).
    - Otherwise, use CPU workers.
    """
    if hw["intel_arc"]:
        # Use ze_index (0-based) for ZE_AFFINITY_MASK, keep original index for logging
        devices = [
            {
                "type": "intel_arc",
                "index": g.get("ze_index", g["index"]),  # Use ze_index if available
                "wmi_index": g["index"],  # Keep original WMI index for reference
                "name": g["name"],
            }
            for g in hw["intel_arc"]
        ]
        return {"backend": "intel_arc", "devices": devices}
    if hw["nvidia"]:
        devices = [
            {"type": "cuda", "index": g["index"], "name": g["name"]}
            for g in hw["nvidia"]
        ]
        return {"backend": "cuda", "devices": devices}

    # CPU fallback
    cpu_workers = max(1, min(4, (hw["cpu_count"] or 1) - 1))  # keep one core free
    devices = [{"type": "cpu", "index": i, "name": "CPU"} for i in range(cpu_workers)]
    return {"backend": "cpu", "devices": devices}


def suggest_libraries(backend):
    """Return a short list of suggested packages for the chosen backend."""
    if backend["backend"] == "intel_arc":
        return [
            "pip install intel-extension-for-pytorch torch==2.4.0+cpu -f https://download.pytorch.org/whl/cpu",
            "pip install openvino-dev",  # optional if using OpenVINO execution providers
            "pip install wmi",  # telemetry
        ]
    if backend["backend"] == "cuda":
        return [
            "pip install torch --index-url https://download.pytorch.org/whl/cu121",
            "pip install openai-whisper",
        ]
    return ["pip install openai-whisper"]


def _has_pkg(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _check_requirements(task: str) -> bool:
    missing = []
    if task == "subtitles":
        if not _has_pkg("whisper"):
            missing.append("openai-whisper")
    if task in ("faces", "faces_export"):
        if not _has_pkg("cv2"):
            missing.append("opencv-python")
        if not _has_pkg("numpy"):
            missing.append("numpy")
    if task == "faces_cluster":
        if not _has_pkg("cv2"):
            missing.append("opencv-python")
        if not _has_pkg("numpy"):
            missing.append("numpy")
        if not _has_pkg("sklearn"):
            missing.append("scikit-learn")
    if task in ("video_enhance", "audio_enhance"):
        if shutil.which("ffmpeg") is None:
            missing.append("ffmpeg (binary)")
    # WMI is optional (only for GPU monitoring), not required for any task
    # if platform.system().lower() == "windows" and not _HAS_WMI:
    #     missing.append("wmi (pip install wmi)")

    if missing:
        LOG.error("Missing requirements for task '%s': %s", task, ", ".join(missing))
        LOG.error(
            "Install as needed, e.g.: pip install %s",
            " ".join(m for m in missing if "(" not in m),
        )
        return False
    return True


def _partition_scenes(scenes, worker_count):
    """
    Duration-aware work distribution using longest-processing-time (LPT) algorithm.

    Falls back to round-robin if FFprobe unavailable or fails.
    """
    # ========== PATCH #3: DURATION-AWARE LOAD BALANCING ==========
    # Try to get video durations via FFprobe
    import subprocess
    import json

    scene_durations = []
    for scene in scenes:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_entries", "format=duration",
                    str(scene),
                ],
                capture_output=True,
                text=True,
                timeout=5,  # 5s timeout per probe
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                duration = float(data.get("format", {}).get("duration", 0))
            else:
                duration = 0  # Unknown, will be sorted last
        except Exception:
            duration = 0  # Fallback to 0 on any error

        scene_durations.append((scene, duration))

    # LPT Algorithm: Sort by duration (descending), assign to least-loaded worker
    scene_durations.sort(key=lambda x: x[1], reverse=True)

    buckets = [[] for _ in range(worker_count)]
    bucket_loads = [0.0] * worker_count  # Track total duration per bucket

    for scene, duration in scene_durations:
        # Assign to worker with smallest current load
        min_idx = bucket_loads.index(min(bucket_loads))
        buckets[min_idx].append(scene)
        bucket_loads[min_idx] += duration if duration > 0 else 60.0  # Assume 60s for unknown

    # Log load distribution
    LOG.info("[scheduler] Load distribution (LPT algorithm):")
    for idx, (bucket, load) in enumerate(zip(buckets, bucket_loads)):
        LOG.info(
            "  Worker %d: %d scenes, estimated %.1f min",
            idx,
            len(bucket),
            load / 60,
        )

    # ========== END PATCH #3 ==========
    return buckets


def build_worker_plan(scenes, backend):
    """Map scenes to workers based on available devices."""
    devices = backend["devices"]
    if not devices:
        return []
    buckets = _partition_scenes(scenes, len(devices))
    return [
        {"device": device, "scenes": bucket} for device, bucket in zip(devices, buckets)
    ]


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def filter_ffmpeg_errors(stderr: str, context: str = "") -> Tuple[List[str], List[str]]:
    """
    Filter out common FFmpeg decode warnings from corrupted video streams.
    Logs filtered decode errors at DEBUG level for debugging.

    Args:
        stderr: FFmpeg stderr output
        context: Context string for logging (e.g., video filename)

    Returns:
        Tuple of (actual_errors, filtered_decode_warnings):
        - actual_errors: List of real error lines (not decode-related)
        - filtered_decode_warnings: List of filtered decode warnings (for summary)
    """
    if not stderr:
        return [], []

    filtered = []
    filtered_decode_warnings = []
    decode_warnings = [
        "error while decoding mb",
        "invalid nal unit size",
        "missing picture in access unit",
        "number of bands",
        "exceeds limit",
        "channel element",
        "is not allocated",
        "prediction is not allowed",
        "reserved bit set",
        "non-existing pps",
        "reference",
        "top block unavailable",
        "error submitting packet to decoder",  # AAC/H264 decode errors
        "error in spectral data",  # AAC decode errors
        "esc overflow",  # AAC decode errors
        "invalid data found when processing input",  # Generic decode errors
        "decoder:",  # Any decoder-related errors
        "decoding",  # Any decoding errors
    ]

    for line in stderr.splitlines():
        line_lower = line.lower()
        # Check if this is a decode warning
        is_decode_warning = any(skip in line_lower for skip in decode_warnings)

        if is_decode_warning:
            filtered_decode_warnings.append(line)
            # Log at DEBUG level for debugging purposes
            if context:
                LOG.debug("[FFmpeg decode warning] %s: %s", context, line.strip())
            else:
                LOG.debug("[FFmpeg decode warning] %s", line.strip())
        else:
            filtered.append(line)

    # Further filter: only return lines with "error" that aren't decode-related
    error_lines = [
        l
        for l in filtered
        if "error" in l.lower()
        and "decoder" not in l.lower()
        and "decoding" not in l.lower()
        and "invalid data" not in l.lower()
    ]

    # Log summary if there were many decode warnings
    if filtered_decode_warnings and len(filtered_decode_warnings) > 5:
        LOG.debug(
            "[FFmpeg] %s: Filtered %d decode warnings (corrupted stream, processing continues)",
            context or "FFmpeg",
            len(filtered_decode_warnings),
        )

    return error_lines, filtered_decode_warnings


def is_scene_done(scene_path: Path, task: str, output_dir: Optional[Path] = None, scenes_folder: Optional[Path] = None) -> bool:
    """
    Check if scene has been successfully processed (checkpoint exists).

    Args:
        scene_path: Path to scene video file
        task: Task name ('subtitles', 'faces', etc.)
        output_dir: Optional output directory for checkpoint markers (if None, checks next to source video)
        scenes_folder: Scenes folder root (needed to calculate relative path for output_dir)

    Returns:
        True if .done marker exists, False otherwise
    """
    # Checkpoint marker: .{stem}.{task}.done
    # Example: .tape1_scene_0001.subtitles.done
    marker_name = f".{scene_path.stem}.{task}.done"
    if output_dir is not None and scenes_folder is not None:
        # Check in run-specific output directory, preserving subfolder structure
        try:
            rel_path = scene_path.relative_to(scenes_folder)
            marker_path = output_dir / rel_path.parent / marker_name
        except ValueError:
            # If scene_path is not under scenes_folder, just use parent directory name
            marker_path = output_dir / scene_path.parent.name / marker_name
    else:
        # Check next to source video (original behavior)
        marker_path = scene_path.parent / marker_name
    return marker_path.exists()


def mark_scene_done(scene_path: Path, task: str, output_dir: Optional[Path] = None, scenes_folder: Optional[Path] = None) -> None:
    """
    Mark scene as successfully processed (create checkpoint).

    Args:
        scene_path: Path to scene video file
        task: Task name
        output_dir: Optional output directory for checkpoint markers (if None, writes next to source video)
        scenes_folder: Scenes folder root (needed to calculate relative path for output_dir)
    """
    marker_name = f".{scene_path.stem}.{task}.done"
    if output_dir is not None and scenes_folder is not None:
        # Write to run-specific output directory, preserving subfolder structure
        try:
            rel_path = scene_path.relative_to(scenes_folder)
            marker_path = output_dir / rel_path.parent / marker_name
        except ValueError:
            # If scene_path is not under scenes_folder, just use parent directory name
            marker_path = output_dir / scene_path.parent.name / marker_name
        marker_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Write next to source video (original behavior)
        marker_path = scene_path.parent / marker_name

    # Atomic creation with metadata
    import json
    metadata = {
        "scene": scene_path.name,
        "task": task,
        "timestamp": datetime.now().isoformat(),
        "pid": os.getpid(),
    }

    tmp_path = marker_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Atomic rename
    tmp_path.rename(marker_path)
    LOG.debug("[checkpoint] Marked done: %s", marker_path.name)


def write_srt(scene_path: Path, segments, output_dir: Optional[Path] = None, scenes_folder: Optional[Path] = None) -> None:
    """
    Write SRT file atomically with checkpoint on success.
    
    Args:
        scene_path: Path to source video file
        segments: Transcription segments
        output_dir: Optional output directory (if None, writes next to source video)
        scenes_folder: Scenes folder root (needed to calculate relative path for output_dir)
    """
    if output_dir is not None and scenes_folder is not None:
        # Write to run-specific output directory, preserving subfolder structure
        # e.g., scenes/tape 1/video.mp4 -> run-dir/tape 1/video.srt
        try:
            rel_path = scene_path.relative_to(scenes_folder)
            srt_path = output_dir / rel_path.with_suffix(".srt")
        except ValueError:
            # If scene_path is not under scenes_folder, just use filename
            srt_path = output_dir / scene_path.name.replace(scene_path.suffix, ".srt")
        srt_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Write next to source video (original behavior)
        srt_path = scene_path.with_suffix(".srt")
    tmp_path = srt_path.with_suffix(".srt.tmp")

    # Write to temp file first
    with open(tmp_path, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(f"{segment['id'] + 1}\n")
            f.write(
                f"{format_time(segment['start'])} --> {format_time(segment['end'])}\n"
            )
            f.write(f"{segment['text'].strip()}\n\n")

    # Atomic rename (replace existing target if present to avoid WinError 183)
    if srt_path.exists():
        srt_path.unlink()
    tmp_path.rename(srt_path)
    LOG.debug("[write_srt] Wrote %s", srt_path.name)


def _ensure_model(name: str, urls) -> Path:
    target = MODEL_CACHE / name
    expected_hash = MODEL_CHECKSUMS.get(name)

    def _sha256(path: Path) -> str:
        import hashlib

        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # Verify existing cache entry
    if target.exists():
        if expected_hash:
            actual = _sha256(target)
            if actual.lower() != expected_hash.lower():
                LOG.warning(
                    "[model-cache] Checksum mismatch for %s (expected %s, got %s); re-downloading",
                    name,
                    expected_hash,
                    actual,
                )
                target.unlink(missing_ok=True)
            else:
                return target
        else:
            LOG.info("[model-cache] No checksum set for %s; using cached copy", name)
            return target

    if isinstance(urls, (str, bytes)):
        urls = [urls]
    last_err = None
    for url in urls:
        try:
            LOG.info("Downloading model %s", name)
            tmp_target = target.with_suffix(".tmp")
            urllib.request.urlretrieve(url, tmp_target)  # nosec - trusted model source
            if expected_hash:
                actual = _sha256(tmp_target)
                if actual.lower() != expected_hash.lower():
                    tmp_target.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Checksum mismatch for {name} from {url}: expected {expected_hash}, got {actual}"
                    )
            else:
                actual = _sha256(tmp_target)
                LOG.warning(
                    "[model-cache] No checksum recorded for %s; downloaded hash: %s. Add to MODEL_CHECKSUMS to enable verification.",
                    name,
                    actual,
                )
            tmp_target.rename(target)
            return target
        except Exception as exc:  # pragma: no cover - network path
            last_err = exc
            LOG.warning("Download failed from %s: %s", url, exc)
    raise RuntimeError(f"Could not download model {name}: {last_err}")


def _load_face_embedder() -> Any:
    """Load InsightFace ArcFace r100 model for high-accuracy face recognition.
    Falls back to SFace if ArcFace unavailable.
    """
    global _RECOG_MODEL_DIM
    try:
        import cv2  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Face identification requires: pip install opencv-python"
        ) from exc

    model_path = _ensure_model(FACE_RECOG_MODEL_NAME, FACE_RECOG_MODEL_URLS)

    # Detect which model we got by filename
    if "arcface" in model_path.name.lower():
        _RECOG_MODEL_DIM = 512
        LOG.info("Using ArcFace r100 (512-dim embeddings)")
    else:
        _RECOG_MODEL_DIM = 128
        LOG.info(
            "Using SFace (128-dim embeddings) - ArcFace unavailable, using fallback"
        )

    net = cv2.dnn.readNet(str(model_path))

    # Optimize for Arc GPU
    backend_id = cv2.dnn.DNN_BACKEND_OPENCV
    target_id = (
        cv2.dnn.DNN_TARGET_OPENCL_FP16
        if cv2.ocl.haveOpenCL()
        else cv2.dnn.DNN_TARGET_CPU
    )
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
    net.setPreferableBackend(backend_id)
    net.setPreferableTarget(target_id)

    return net


def _face_embedding(net, face_img) -> Optional[List[float]]:
    """Extract face embedding (ArcFace 512-dim or SFace 128-dim, L2 normalized)."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if face_img.size == 0:
        return None

    # ArcFace expects 112x112 input, normalized to [-1, 1] range
    # SFace expects 112x112 input, normalized to [0, 1] range
    if _RECOG_MODEL_DIM == 512:
        # ArcFace normalization
        blob = cv2.dnn.blobFromImage(
            face_img,
            1.0 / 127.5,
            (112, 112),
            (127.5, 127.5, 127.5),
            swapRB=True,
            crop=True,
        )
    else:
        # SFace normalization
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0 / 255.0, (112, 112), (0, 0, 0), swapRB=True, crop=True
        )

    net.setInput(blob)
    emb = net.forward()
    emb = emb.reshape(-1)
    # L2 normalize
    norm = float(np.linalg.norm(emb))
    if norm == 0:
        return None
    return (emb / norm).tolist()


def _cosine(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _resolve_run_folder(base: Path, run_name: Optional[str]) -> Path:
    name = run_name or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    folder = base / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def check_vram_available(device_index: int, required_gb: float, backend: str) -> Tuple[bool, float]:
    """
    Check if GPU has enough VRAM for model loading.

    Returns:
        (is_available, free_gb): Tuple of whether VRAM is sufficient and how much is free
    """
    if backend != "intel_arc":
        return (True, 0.0)  # Only check for Intel Arc GPUs

    try:
        import intel_extension_for_pytorch as ipex
        if not ipex.xpu.is_available():
            LOG.warning("[vram-check] Intel XPU not available")
            return (False, 0.0)

        # Get total memory first (required for accurate free memory calculation)
        try:
            mem_total = ipex.xpu.get_device_properties(device_index).total_memory
        except Exception:
            mem_total = 16.0 * (1024**3)  # Assume 16GB for Arc A770 (in bytes)

        # Get allocated memory
        mem_allocated = ipex.xpu.memory_allocated(device_index)

        # Calculate free memory: total - allocated
        # NOTE: mem_reserved can be 0 on first load, so we use total - allocated
        mem_free = mem_total - mem_allocated
        mem_free_gb = mem_free / (1024**3)
        mem_total_gb = mem_total / (1024**3)
        mem_allocated_gb = mem_allocated / (1024**3)

        is_available = mem_free_gb >= required_gb

        LOG.info(
            "[vram-check] GPU %d | Total: %.2f GB | Used: %.2f GB | Free: %.2f GB | Need: %.2f GB | OK: %s",
            device_index,
            mem_total_gb,
            mem_allocated_gb,
            mem_free_gb,
            required_gb,
            "✓" if is_available else "✗",
        )

        return (is_available, mem_free_gb)

    except Exception as e:
        LOG.warning("[vram-check] Failed to check VRAM on GPU %d: %s", device_index, e)
        # Fail closed: treat as unknown/insufficient to force a smaller model
        return (False, 0.0)


def select_model_for_vram(free_vram_gb: float) -> Tuple[str, float]:
    """
    Select appropriate Whisper model size based on available VRAM.

    Returns:
        (model_name, estimated_vram_gb): Selected model and its estimated VRAM usage
    """
    # Whisper model VRAM estimates (empirical):
    # - large: ~10GB
    # - medium: ~5GB
    # - small: ~2GB
    # - base: ~1GB
    # - tiny: ~0.5GB

    if free_vram_gb >= 6.0:
        return ("medium", 5.0)  # Default: medium for quality
    elif free_vram_gb >= 3.0:
        return ("small", 2.0)
    elif free_vram_gb >= 1.5:
        return ("base", 1.0)
    else:
        return ("tiny", 0.5)  # Last resort


def _load_model_for_device(device, backend: str):
    """
    Load a whisper model tuned for the target device.
    Prefers Intel Arc GPU execution (xpu or DirectML) when available.
    """
    try:
        import whisper
    except ImportError:
        raise RuntimeError("whisper not installed. Run: pip install openai-whisper")

    # ========== PATCH #1: SET SEEDS FOR REPRODUCIBILITY ==========
    # Set all random seeds before model loading for deterministic outputs
    # This ensures Whisper transcription is reproducible across runs
    import random
    import numpy as np

    # Use device_index as seed offset to ensure different workers get different
    # randomness, but same worker always gets same sequence
    base_seed = 42  # Configurable via CLI in future
    device_index_for_seed = device.get("index", 0)
    worker_seed = base_seed + device_index_for_seed

    random.seed(worker_seed)
    np.random.seed(worker_seed)

    # For torch (if available)
    try:
        import torch
        torch.manual_seed(worker_seed)
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            # Intel XPU seed (IPEX)
            torch.xpu.manual_seed_all(worker_seed)
    except ImportError:
        pass  # torch not required for all tasks

    LOG.info(
        "[seed] PID=%d | device_idx=%s | Seeded RNGs with worker_seed=%d for deterministic outputs",
        os.getpid(),
        device_index_for_seed,
        worker_seed,
    )
    # ========== END PATCH #1 ==========

    # Preserve any GPU affinity set by worker_bootstrap (especially for Intel Arc)
    os.environ.setdefault("ZE_AFFINITY_MASK", str(device.get("index", 0)))
    load_kwargs: Dict[str, Any] = {}
    fallback_reason = None
    post_move = None  # Optional post-load device move (e.g., DML)
    final_device_desc = "cpu"

    # Log model load start (for performance tracking and testing)
    pid = os.getpid()
    device_index = device.get("index", "N/A")
    device_name = device.get("name", device.get("type", "unknown"))

    # VRAM SAFETY CHECK: Select model size based on available VRAM
    model_name = "medium"  # Default preference
    if backend == "intel_arc":
        vram_ok, free_vram_gb = check_vram_available(device_index, required_gb=5.0, backend=backend)
        if not vram_ok:
            # Auto-downgrade model size (or when VRAM is unknown)
            model_name, estimated_vram = select_model_for_vram(free_vram_gb)
            LOG.warning(
                "[vram-safety] GPU %d low/unknown VRAM (%.2f GB free), downgrading to '%s' model (needs ~%.1f GB)",
                device_index,
                free_vram_gb,
                model_name,
                estimated_vram,
            )

    if backend == "intel_arc":
        # Try Intel XPU backend first (requires IPEX/oneAPI), then DirectML.
        try:
            import torch  # type: ignore

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                # Set ZE_AFFINITY_MASK before the device is touched.
                # CRITICAL FIX: Pin to entire card, not tile on card 0
                # Format: <card_index> pins to all tiles on that card
                # - device_index=0 → ZE_AFFINITY_MASK=0 (all tiles on card 0)
                # - device_index=1 → ZE_AFFINITY_MASK=1 (all tiles on card 1)
                ze_idx = device.get("index", 0)
                os.environ["ZE_AFFINITY_MASK"] = str(ze_idx)
                load_kwargs = {"device": f"xpu:{device['index']}"}
                final_device_desc = load_kwargs["device"]
            else:
                # Prefer loading on CPU, then moving to DirectML to avoid map_location pickle issues.
                import torch_directml as tdm  # type: ignore

                load_kwargs = {"device": "cpu"}
                post_move = tdm.device(device["index"])
                final_device_desc = f"dml:{device['index']}"
        except Exception as exc:
            # Fall back to CPU but keep going.
            load_kwargs = {"device": "cpu"}
            fallback_reason = f"intel_arc backend requested but GPU packages unavailable ({exc})"
            final_device_desc = "cpu"
    elif backend == "cuda":
        load_kwargs = {"device": f"cuda:{device['index']}"}
        final_device_desc = load_kwargs["device"]
    else:
        load_kwargs = {"device": "cpu"}
        final_device_desc = "cpu"

    if backend == "intel_arc" and load_kwargs.get("device") == "cpu":
        LOG.warning(
            "Falling back to CPU for intel_arc backend (missing Intel XPU or DirectML). "
            "Install GPU packages, e.g.: pip install intel-extension-for-pytorch torch==2.4.0+cpu -f https://download.pytorch.org/whl/cpu; pip install torch-directml. Reason: %s",
            fallback_reason or "unknown",
        )

    # Load on the chosen device
    model = whisper.load_model(model_name, **load_kwargs)

    # Optional post-move (e.g., CPU -> DML)
    if post_move is not None:
        try:
            model = model.to(post_move)
        except Exception as exc:
            LOG.warning(
                "Post-move to %s failed for %s (device_idx=%s): %s; staying on CPU",
                final_device_desc,
                device_name,
                device_index,
                exc,
            )
            final_device_desc = "cpu"

    # Log contract: exactly one [model-load] per worker PID (final device)
    LOG.info(
        "[model-load] PID=%d | device_idx=%s (%s) | model=%s | backend=%s | kwargs=%s",
        pid,
        device_index,
        device_name,
        model_name,
        backend,
        {"device": final_device_desc},
    )

    LOG.info(
        "[model-ready] PID=%d | model=%s loaded successfully",
        pid,
        model_name,
    )

    # VERIFICATION: Check GPU affinity was honored
    if backend == "intel_arc" and os.environ.get("_VERIFY_GPU_AFFINITY") == "1":
        try:
            import torch
            if hasattr(model, "parameters"):
                actual_device = str(next(model.parameters()).device)
                expected_device = f"xpu:{device_index}"

                LOG.info(
                    "[affinity-verify] PID=%d | Expected device: %s | Actual device: %s",
                    pid,
                    expected_device,
                    actual_device,
                )

                if expected_device not in actual_device:
                    LOG.error("⚠️  GPU affinity verification FAILED: model on %s, expected %s", actual_device, expected_device)
        except Exception as e:
            LOG.debug("GPU affinity verification failed: %s", e)

    return model


def _cleanup_gpu_resources(model, backend: str, device: Dict[str, Any]) -> None:
    """
    Clean up GPU resources by releasing model memory and clearing caches.
    
    This helps prevent resource leaks and stuck processes during worker termination.
    """
    try:
        import gc
        
        # Delete model reference
        if model is not None:
            del model
        
        # Force garbage collection
        gc.collect()
        
        # For Intel XPU, try to clear memory cache
        if backend == "intel_arc":
            try:
                import torch
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    device_index = device.get("index", 0)
                    # Clear XPU cache
                    torch.xpu.empty_cache()
                    LOG.debug("[cleanup] Cleared XPU cache for device %d", device_index)
            except Exception as e:
                LOG.debug("[cleanup] Could not clear XPU cache: %s", e)
        
        # For CUDA, clear CUDA cache
        elif backend == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    device_index = device.get("index", 0)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(device_index)
                    LOG.debug("[cleanup] Cleared CUDA cache for device %d", device_index)
            except Exception as e:
                LOG.debug("[cleanup] Could not clear CUDA cache: %s", e)
        
        # Force another garbage collection after cache clearing
        gc.collect()
        
    except Exception as e:
        LOG.debug("[cleanup] Error during GPU resource cleanup: %s", e)


def transcribe_scene(scene_path: Path, model, output_dir: Optional[Path] = None, scenes_folder: Optional[Path] = None) -> None:
    """
    Transcribe audio from a video file using Whisper.

    Args:
        scene_path: Path to video file
        model: Whisper model
        output_dir: Optional output directory for SRT files (if None, writes next to source video)
        scenes_folder: Scenes folder root (needed to calculate relative path for output_dir)

    Raises:
        RuntimeError: If the video file has no audio stream or FFmpeg fails
        Exception: For any other transcription errors
    """
    try:
        LOG.debug(f"[transcribe_scene] Starting transcription for {scene_path.name}")
        LOG.debug(f"[transcribe_scene] Model device: {model.device if hasattr(model, 'device') else 'unknown'}")

        result = model.transcribe(str(scene_path))

        LOG.debug(f"[transcribe_scene] Transcription complete for {scene_path.name}, writing SRT")
        write_srt(scene_path, result["segments"], output_dir, scenes_folder)
        LOG.debug(f"[transcribe_scene] SRT written successfully for {scene_path.name}")

    except RuntimeError as e:
        # FFmpeg error (e.g., no audio stream)
        error_msg = str(e)
        if "Failed to load audio" in error_msg:
            # Extract just the key error info, not the entire FFmpeg output
            if "Output file does not contain any stream" in error_msg:
                raise RuntimeError(f"Video file has no audio stream: {scene_path.name}") from e
            elif "does not exist" in error_msg or "No such file" in error_msg:
                raise RuntimeError(f"File not found: {scene_path.name}") from e
            else:
                # Generic FFmpeg failure - re-raise with simplified message
                raise RuntimeError(f"Failed to extract audio from {scene_path.name}: {error_msg.split('Error')[0]}") from e
        else:
            # Some other RuntimeError - re-raise as-is
            raise
    except AttributeError as e:
        # Special case: Check if this is the "'TypeError' object has no attribute 'shape'" error
        import traceback
        tb = traceback.format_exc()

        if "'TypeError' object has no attribute" in str(e):
            LOG.error(f"[transcribe_scene] CRITICAL BUG DETECTED: TypeError object being used as data")
            LOG.error(f"[transcribe_scene] This indicates a bug in whisper or Intel XPU backend")
            LOG.error(f"[transcribe_scene] Full traceback:\n{tb}")

            # Try to find the TypeError in the traceback
            if "TypeError:" in tb:
                typeerror_lines = [line for line in tb.split('\n') if 'TypeError' in line]
                LOG.error(f"[transcribe_scene] Related TypeError: {typeerror_lines}")

        raise RuntimeError(f"Internal transcription error for {scene_path.name}: {e}") from e
    except Exception as e:
        # Catch any other exception and add context
        import traceback
        LOG.error(f"[transcribe_scene] Unexpected error for {scene_path.name}: {type(e).__name__}: {e}")
        LOG.error(f"[transcribe_scene] Full traceback:\n{traceback.format_exc()}")
        raise


def _detect_faces_scrfd(detector_net, frame, min_score: float):
    """Detect faces using SCRFD model via OpenCV DNN."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    h, w = frame.shape[:2]
    # SCRFD expects input size that's a multiple of 32
    input_size = (int((w + 31) // 32 * 32), int((h + 31) // 32 * 32))

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, input_size, (0, 0, 0), swapRB=True, crop=False
    )
    detector_net.setInput(blob)
    outputs = detector_net.forward()

    # SCRFD outputs: [batch, num_detections, 15] where 15 = [x, y, w, h, score, 5 landmarks x2]
    detections = []
    if len(outputs.shape) == 3:
        outputs = outputs[0]  # Remove batch dimension

    scale_x = w / input_size[0]
    scale_y = h / input_size[1]

    for det in outputs:
        x, y, w_det, h_det, score = det[:5]
        if score < min_score:
            continue
        # Scale back to original image size
        x = float(x * scale_x)
        y = float(y * scale_y)
        w_det = float(w_det * scale_x)
        h_det = float(h_det * scale_y)
        box = [x, y, x + w_det, y + h_det]
        detections.append({"box": box, "score": float(score)})

    return detections


def _run_face_tracking(scene_path: Path, frame_stride: int, min_score: float) -> Path:
    """
    High-accuracy face detection/tracking using SCRFD + ArcFace r100.
    Optimized for Intel Arc GPUs with OpenCL acceleration.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Face detection requires: pip install opencv-python numpy"
        ) from exc

    detect_model_path = _ensure_model(FACE_DETECT_MODEL_NAME, FACE_DETECT_MODEL_URLS)
    embedder = _load_face_embedder()

    # Load SCRFD detector
    detector_net = cv2.dnn.readNet(str(detect_model_path))
    backend_id = cv2.dnn.DNN_BACKEND_OPENCV
    target_id = (
        cv2.dnn.DNN_TARGET_OPENCL_FP16
        if cv2.ocl.haveOpenCL()
        else cv2.dnn.DNN_TARGET_CPU
    )
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
    detector_net.setPreferableBackend(backend_id)
    detector_net.setPreferableTarget(target_id)

    cap = cv2.VideoCapture(str(scene_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {scene_path}")

    tracks: List[Dict[str, Any]] = []
    next_id = 0
    results = []

    def iou(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter_area / float(area_a + area_b - inter_area)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        # Detect faces with SCRFD
        face_dets = _detect_faces_scrfd(detector_net, frame, min_score)

        detections = []
        for det in face_dets:
            # Extract face crop for embedding
            x1, y1, x2, y2 = map(int, det["box"])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            face_crop = frame[y1:y2, x1:x2]
            emb = _face_embedding(embedder, face_crop)
            detections.append({"box": det["box"], "score": det["score"], "emb": emb})

        # tracking with IOU + embedding similarity
        for det in detections:
            best_track = None
            best_iou = 0.0
            best_cos = 0.0
            for tr in tracks:
                ov = iou(det["box"], tr["box"])
                cos = (
                    _cosine(det["emb"], tr["emb"])
                    if det["emb"] is not None and tr.get("emb")
                    else 0.0
                )
                score = ov * 0.6 + cos * 0.4
                if score > best_iou:
                    best_iou = score
                    best_track = tr
                    best_cos = cos
            if best_track and best_iou > 0.35:
                best_track["box"] = det["box"]
                best_track["emb"] = det["emb"] or best_track.get("emb")
                best_track["last_frame"] = frame_idx
                det["id"] = best_track["id"]
                det["match_score"] = best_cos
            else:
                det["id"] = next_id
                tracks.append(
                    {
                        "id": next_id,
                        "box": det["box"],
                        "emb": det["emb"],
                        "last_frame": frame_idx,
                    }
                )
                next_id += 1

        # optional naming from gallery
        if FACE_GALLERY:
            for det in detections:
                if det.get("emb") is None:
                    continue
                best_name = None
                best_sim = 0.0
                for name, gallery_emb in FACE_GALLERY.items():
                    sim = _cosine(det["emb"], gallery_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name
                if best_name and best_sim >= _cli_args.id_threshold:
                    det["name"] = best_name
                    det["name_score"] = best_sim

        if detections:
            results.append({"frame": frame_idx, "detections": detections})
        frame_idx += 1

    cap.release()
    out_path = scene_path.with_suffix(".faces.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"video": scene_path.name, "frames": results, "tracks": tracks},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return out_path


def _assess_face_quality(img) -> float:
    """Assess face image quality (blur, brightness, contrast). Returns 0-1 score."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if img.size == 0:
        return 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Laplacian variance (blur detection)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(1.0, laplacian_var / 100.0)  # Normalize

    # Brightness (avoid too dark/bright)
    mean_brightness = np.mean(gray)
    brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5

    # Contrast (std dev)
    contrast_score = min(1.0, np.std(gray) / 64.0)

    # Combined quality score
    quality = blur_score * 0.5 + brightness_score * 0.25 + contrast_score * 0.25
    return float(quality)


def _cluster_faces(
    run_folder: Path,
    similarity_threshold: float,
    min_samples: int,
    method: str = "dbscan",
    quality_threshold: float = 0.3,
    n_passes: int = 1,
) -> Path:
    """
    Cluster face crops by identity using ArcFace r100 embeddings and DBSCAN.
    Regroups faces from person_#### folders into cluster_#### folders.
    Uses 512-dim ArcFace embeddings for superior accuracy vs SFace.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        from sklearn.cluster import DBSCAN  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Face clustering requires: pip install opencv-python numpy scikit-learn"
        ) from exc

    LOG.info("Loading face embeddings from %s", run_folder)
    net = _load_face_embedder()

    # Set up GPU acceleration if available
    backend_id = cv2.dnn.DNN_BACKEND_OPENCV
    target_id = (
        cv2.dnn.DNN_TARGET_OPENCL_FP16
        if cv2.ocl.haveOpenCL()
        else cv2.dnn.DNN_TARGET_CPU
    )
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
    net.setPreferableBackend(backend_id)
    net.setPreferableTarget(target_id)

    face_files = []
    embeddings = []
    quality_scores = []

    # Collect all face images from person_#### folders with quality filtering
    LOG.info("Loading and assessing face quality...")
    for person_dir in sorted(run_folder.iterdir()):
        if not person_dir.is_dir() or not person_dir.name.startswith("person_"):
            continue
        for img_file in sorted(person_dir.glob("*.jpg")):
            img = cv2.imread(str(img_file))
            if img is None or img.size == 0:
                continue

            # Assess quality
            quality = _assess_face_quality(img)
            if quality < quality_threshold:
                continue  # Skip low-quality faces

            emb = _face_embedding(net, img)
            if emb is not None:
                face_files.append((person_dir.name, img_file))
                embeddings.append(emb)
                quality_scores.append(quality)

    if len(embeddings) < 2:
        LOG.warning(
            "Not enough faces to cluster (found %d after quality filtering)",
            len(embeddings),
        )
        return run_folder

    LOG.info("Loaded %d faces (quality >= %.2f)", len(embeddings), quality_threshold)

    if len(embeddings) < 2:
        LOG.warning("Not enough faces to cluster (found %d)", len(embeddings))
        return run_folder

    LOG.info(
        "Computing clusters from %d faces (threshold=%.3f, min_samples=%d)",
        len(embeddings),
        similarity_threshold,
        min_samples,
    )

    # Convert to numpy array for DBSCAN
    X = np.array(embeddings)

    # Compute pairwise similarities for diagnostics (use all faces for better stats)
    from sklearn.metrics.pairwise import cosine_similarity

    # For very large sets, sample for stats but use all for clustering
    if len(embeddings) > 2000:
        import random

        sample_size = 1000
        sample_indices = random.sample(range(len(embeddings)), sample_size)
        X_sample = X[sample_indices]
        LOG.info(
            "Computing similarity stats on sample of %d faces (using all %d for clustering)",
            sample_size,
            len(embeddings),
        )
        sim_matrix_sample = cosine_similarity(X_sample)
        triu_indices = np.triu_indices(len(sim_matrix_sample), k=1)
        similarities = sim_matrix_sample[triu_indices]
    else:
        LOG.info("Computing similarity stats on all %d faces", len(embeddings))
        sim_matrix = cosine_similarity(X)
        triu_indices = np.triu_indices(len(sim_matrix), k=1)
        similarities = sim_matrix[triu_indices]

    LOG.info(
        "Similarity stats: min=%.3f, max=%.3f, mean=%.3f, median=%.3f, std=%.3f",
        float(np.min(similarities)),
        float(np.max(similarities)),
        float(np.mean(similarities)),
        float(np.median(similarities)),
        float(np.std(similarities)),
    )

    # Show distribution percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    LOG.info(
        "Similarity percentiles: %s",
        ", ".join([f"p{p}={np.percentile(similarities, p):.3f}" for p in percentiles]),
    )

    # Choose clustering method
    if method == "hierarchical":
        from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
        from scipy.spatial.distance import squareform
        from sklearn.metrics import silhouette_score

        LOG.info("Using hierarchical clustering with %d passes", n_passes)

        # Compute distance matrix (1 - similarity)
        from sklearn.metrics.pairwise import cosine_distances

        dist_matrix = cosine_distances(X)
        dist_condensed = squareform(dist_matrix, checks=False)

        # Try multiple linkage methods and select best
        linkage_methods = ["ward", "average", "complete"] if n_passes > 1 else ["ward"]
        best_labels = None
        best_score = -1
        best_method = None
        best_linkage = None

        for linkage_method in linkage_methods:
            try:
                LOG.info("Trying linkage method: %s", linkage_method)
                linkage_matrix = linkage(dist_condensed, method=linkage_method)

                # Try multiple cut points
                cut_points = []
                distance_threshold = 1.0 - similarity_threshold
                cut_points.append(("distance", distance_threshold))

                # Try reasonable cluster counts
                for n_clust in [10, 15, 20, 25, 30, 50]:
                    if n_clust < len(embeddings):
                        cut_points.append(("maxclust", n_clust))

                for criterion, value in cut_points:
                    labels_test = fcluster(linkage_matrix, value, criterion=criterion)
                    n_clust_test = len(set(labels_test))

                    if n_clust_test < 2 or n_clust_test >= len(embeddings) * 0.9:
                        continue

                    # Evaluate with silhouette score (sample for large sets)
                    sample_size_sil = min(1000, len(embeddings))
                    if len(embeddings) > sample_size_sil:
                        import random

                        sil_indices = random.sample(
                            range(len(embeddings)), sample_size_sil
                        )
                        sil_score = silhouette_score(
                            X[sil_indices], labels_test[sil_indices], metric="cosine"
                        )
                    else:
                        sil_score = silhouette_score(X, labels_test, metric="cosine")

                    LOG.info(
                        "  %s=%.3f -> %d clusters, silhouette=%.3f",
                        criterion,
                        value,
                        n_clust_test,
                        sil_score,
                    )

                    if sil_score > best_score:
                        best_score = sil_score
                        best_labels = labels_test
                        best_method = criterion
                        best_linkage = linkage_matrix
                        LOG.info("    ^ New best (silhouette=%.3f)", sil_score)

            except Exception as exc:
                LOG.warning("Linkage method %s failed: %s", linkage_method, exc)
                continue

        if best_labels is None:
            # Fallback to simple ward linkage
            LOG.warning("All linkage methods failed, using simple ward")
            linkage_matrix = linkage(dist_condensed, method="ward")
            distance_threshold = 1.0 - similarity_threshold
            best_labels = fcluster(
                linkage_matrix, distance_threshold, criterion="distance"
            )
            best_linkage = linkage_matrix

        labels = best_labels
        LOG.info(
            "Selected: %s method, %d clusters, silhouette=%.3f",
            best_method,
            len(set(labels)),
            best_score,
        )

        # Save dendrogram visualization
        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt

            plt.figure(figsize=(20, 10))
            dendrogram(
                best_linkage,
                truncate_mode="level",
                p=15,
                show_leaf_counts=True,
                leaf_rotation=90,
                leaf_font_size=8,
            )
            plt.title(
                f"Face Clustering Dendrogram - {best_method} method, {len(set(labels))} clusters (silhouette={best_score:.3f})"
            )
            plt.xlabel("Face Index")
            plt.ylabel("Distance")
            dendro_path = run_folder / "clustered" / "dendrogram.png"
            dendro_path.parent.mkdir(exist_ok=True)
            plt.savefig(dendro_path, dpi=200, bbox_inches="tight")
            plt.close()
            LOG.info("Saved dendrogram to %s", dendro_path)
        except Exception as exc:
            LOG.warning("Could not save dendrogram: %s", exc)

        n_clusters = len(set(labels))
        n_noise = 0  # Hierarchical doesn't have noise
        # Adjust labels to start from 0
        unique_labels = sorted(set(labels))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])

    else:
        # DBSCAN: eps is cosine distance threshold (1 - similarity)
        # cosine similarity 0.75 -> distance 0.25
        eps = 1.0 - similarity_threshold
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(X)
        labels = clustering.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

    LOG.info("Found %d clusters, %d noise points", n_clusters, n_noise)

    # Warn if everything is in one cluster
    if n_clusters == 1:
        if n_noise == 0:
            LOG.warning(
                "All faces clustered into a single group at threshold %.2f. This suggests:",
                similarity_threshold,
            )
            LOG.warning("  1. Faces may be very similar (family members, same person)")
            LOG.warning("  2. Threshold may need to be >0.90 (try 0.92-0.95)")
            LOG.warning("  3. Consider hierarchical clustering or manual review")
        else:
            LOG.warning("Only 1 cluster found with %d noise points. Consider:", n_noise)
            LOG.warning("  - Increasing threshold to 0.92-0.95 for stricter matching")
            LOG.warning(
                "  - Using hierarchical clustering (--cluster-method hierarchical)"
            )
            LOG.warning("  - Manual review of noise folder to identify distinct people")

    # Warn if everything is in one cluster
    if n_clusters == 1 and n_noise == 0:
        LOG.warning(
            "All faces clustered into a single group. Try increasing --cluster-similarity (e.g., 0.80-0.85) for stricter matching."
        )

    # Create clustered folders and initial cluster map
    clustered_folder = run_folder / "clustered"
    clustered_folder.mkdir(exist_ok=True)

    cluster_map = {}
    for (person_name, img_file), label in zip(face_files, labels):
        if label == -1:
            cluster_name = "noise"
        else:
            cluster_name = f"cluster_{label:04d}"

        cluster_dir = clustered_folder / cluster_name
        cluster_dir.mkdir(exist_ok=True)

        # Copy file with original name preserved
        dest = cluster_dir / img_file.name
        shutil.copy2(img_file, dest)

        if cluster_name not in cluster_map:
            cluster_map[cluster_name] = []
        cluster_map[cluster_name].append(str(img_file.relative_to(run_folder)))

    # Post-process: merge very similar clusters (optional, for hierarchical)
    if method == "hierarchical" and n_clusters > 10:
        LOG.info("Post-processing: checking for very similar clusters to merge...")

        cluster_centroids = {}
        cluster_indices = {}
        for idx, label in enumerate(labels):
            if label not in cluster_indices:
                cluster_indices[label] = []
            cluster_indices[label].append(idx)

        # Compute cluster centroids
        for label, indices in cluster_indices.items():
            cluster_embeddings = X[indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)  # Normalize
            cluster_centroids[label] = centroid

        # Find clusters to merge (similarity > 0.95)
        merge_threshold = 0.95
        merged = set()
        merge_map = {}
        cluster_list = sorted(cluster_centroids.keys())

        for i, label1 in enumerate(cluster_list):
            if label1 in merged:
                continue
            for label2 in cluster_list[i + 1 :]:
                if label2 in merged:
                    continue
                sim = float(
                    np.dot(cluster_centroids[label1], cluster_centroids[label2])
                )
                if sim > merge_threshold:
                    merge_map[label2] = label1
                    merged.add(label2)
                    LOG.info(
                        "Merging cluster %d into %d (similarity=%.3f)",
                        label2,
                        label1,
                        sim,
                    )

        # Apply merges
        if merge_map:
            labels_merged = labels.copy()
            for old_label, new_label in merge_map.items():
                labels_merged[labels == old_label] = new_label

            # Recreate cluster folders with merged labels
            # Remove old cluster directories (they may have files, so use rmtree)
            for old_cluster_dir in clustered_folder.glob("cluster_*"):
                if old_cluster_dir.is_dir():
                    shutil.rmtree(old_cluster_dir)

            cluster_map = {}
            for (person_name, img_file), label in zip(face_files, labels_merged):
                cluster_name = f"cluster_{label:04d}"
                cluster_dir = clustered_folder / cluster_name
                cluster_dir.mkdir(exist_ok=True)
                dest = cluster_dir / img_file.name
                shutil.copy2(img_file, dest)
                if cluster_name not in cluster_map:
                    cluster_map[cluster_name] = []
                cluster_map[cluster_name].append(str(img_file.relative_to(run_folder)))

            labels = labels_merged
            n_clusters = len(set(labels))
            LOG.info("After merging: %d clusters", n_clusters)

    # Save clustering report
    report_path = clustered_folder / "clustering_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_faces": len(embeddings),
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "similarity_threshold": similarity_threshold,
                "min_samples": min_samples,
                "quality_threshold": quality_threshold,
                "method": method,
                "n_passes": n_passes if method == "hierarchical" else 1,
                "clusters": {k: len(v) for k, v in cluster_map.items()},
                "cluster_map": cluster_map,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    LOG.info("Clustered faces saved to %s", clustered_folder)
    LOG.info(
        "Review cluster_#### folders and rename to person names, then delete 'noise' folder if needed"
    )
    return clustered_folder


def _export_faces(
    scene_path: Path,
    frame_stride: int,
    min_score: float,
    out_root: Path,
    run_name: str,
    crop_size: int,
    max_per_track: int,
) -> Path:
    """
    Export face crops grouped by track to facilitate manual naming.
    Creates run folder with subfolders person_XXXX containing face crops.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Face export requires: pip install opencv-python numpy"
        ) from exc

    detect_model_path = _ensure_model(FACE_DETECT_MODEL_NAME, FACE_DETECT_MODEL_URLS)

    # Load SCRFD detector with GPU acceleration
    detector_net = cv2.dnn.readNet(str(detect_model_path))
    backend_id = cv2.dnn.DNN_BACKEND_OPENCV
    target_id = (
        cv2.dnn.DNN_TARGET_OPENCL_FP16
        if cv2.ocl.haveOpenCL()
        else cv2.dnn.DNN_TARGET_CPU
    )
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
    detector_net.setPreferableBackend(backend_id)
    detector_net.setPreferableTarget(target_id)

    run_folder = _resolve_run_folder(out_root, run_name)
    cap = cv2.VideoCapture(str(scene_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {scene_path}")

    tracks: List[Dict[str, Any]] = []
    next_id = 0
    track_counts: Dict[int, int] = {}

    def iou(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter_area / float(area_a + area_b - inter_area)

    frame_idx = 0
    manifest = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        # Detect faces with SCRFD
        face_dets = _detect_faces_scrfd(detector_net, frame, min_score)
        detections = [{"box": det["box"], "score": det["score"]} for det in face_dets]

        for det in detections:
            best_track = None
            best_iou = 0.0
            for tr in tracks:
                ov = iou(det["box"], tr["box"])
                if ov > best_iou:
                    best_iou = ov
                    best_track = tr
            if best_track and best_iou > 0.35:
                best_track["box"] = det["box"]
                best_track["last_frame"] = frame_idx
                det["id"] = best_track["id"]
            else:
                det["id"] = next_id
                tracks.append(
                    {"id": next_id, "box": det["box"], "last_frame": frame_idx}
                )
                next_id += 1

            tid = det["id"]
            track_counts.setdefault(tid, 0)
            if track_counts[tid] >= max_per_track:
                continue

            x1, y1, x2, y2 = map(int, det["box"])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            face_crop = frame[y1:y2, x1:x2]
            if crop_size > 0:
                face_crop = cv2.resize(
                    face_crop, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC
                )

            person_dir = run_folder / f"person_{tid:04d}"
            person_dir.mkdir(parents=True, exist_ok=True)
            out_file = person_dir / f"{scene_path.stem}_f{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_file), face_crop)
            track_counts[tid] += 1
            manifest.append(
                {"track_id": tid, "frame": frame_idx, "file": str(out_file)}
            )

        frame_idx += 1

    cap.release()
    manifest_path = run_folder / f"{scene_path.stem}_faces_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video": scene_path.name,
                "tracks": tracks,
                "manifest": manifest,
                "rename_hint": "Rename folders person_#### to actual names",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return manifest_path


def _enhance_video(
    scene_path: Path,
    scale: float,
    denoise: float,
    codec: str,
    crf: int,
    force: bool = False,
) -> Path:
    # Find next available filename with enumeration
    base_name = scene_path.stem + "_enhanced"
    suffix = scene_path.suffix
    out_path = scene_path.with_name(base_name + suffix)

    # If file exists, enumerate: _enhanced(1).mp4, _enhanced(2).mp4, etc.
    if out_path.exists() and not force:
        counter = 1
        while True:
            enumerated_name = f"{base_name}({counter}){suffix}"
            out_path = scene_path.with_name(enumerated_name)
            if not out_path.exists():
                LOG.info(
                    "Enhanced video exists, creating enumerated version: %s",
                    out_path.name,
                )
                break
            counter += 1
    elif out_path.exists() and force:
        LOG.info("Force re-encoding, removing existing: %s", out_path.name)
        out_path.unlink()

    filters = []
    if denoise > 0:
        # hqdn3d: temporal+spatial denoise; tuned modestly to avoid over-blur
        filters.append(f"hqdn3d={denoise}:{denoise}:{denoise*4}:{denoise*4}")
    if scale and abs(scale - 1.0) > 1e-3:
        # Use lanczos for best quality upscaling
        filters.append(f"scale=iw*{scale}:ih*{scale}:flags=lanczos")
        # Add light sharpening after upscale to enhance detail (only if scaling)
        # Unsharp mask: 5:5:1.0:5:5:0.0 (luma only, subtle)
        filters.append("unsharp=5:5:1.0:5:5:0.0")
    vf = ",".join(filters) if filters else "null"

    # Get original video resolution to verify scaling
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(scene_path),
    ]
    orig_w, orig_h = None, None
    try:
        probe_result = subprocess.run(
            probe_cmd, capture_output=True, text=True, check=True, timeout=5
        )
        parts = probe_result.stdout.strip().split(",")
        if len(parts) >= 2:
            orig_w, orig_h = int(parts[0]), int(parts[1])
            LOG.info(
                "[enhance-video] %s: %dx%d -> scale %.2fx -> %dx%d",
                scene_path.name,
                orig_w,
                orig_h,
                scale,
                int(orig_w * scale) if orig_w else 0,
                int(orig_h * scale) if orig_h else 0,
            )
    except Exception as e:
        LOG.warning(
            "[enhance-video] Could not probe resolution for %s: %s", scene_path.name, e
        )

    # Build FFmpeg command based on codec type
    cmd = [
        "ffmpeg",
        "-y",
        "-err_detect",
        "ignore_err",  # Ignore decode errors in corrupted streams
        "-fflags",
        "+genpts+igndts",  # Generate missing timestamps, ignore DTS errors
        "-i",
        str(scene_path),
        "-vf",
        vf,
        "-c:v",
        codec,
    ]

    # Codec-specific parameters
    if codec == "hevc_qsv":
        # Intel Quick Sync Video (QSV) parameters
        # QSV uses global_quality instead of CRF (scale: 1-100, lower = better quality)
        # CRF 12-16 maps roughly to global_quality 18-23
        qsv_quality = max(
            18, min(28, int(23 - (crf - 12) * 0.5))
        )  # Map CRF to QSV quality
        cmd.extend(
            [
                "-global_quality",
                str(qsv_quality),
                "-preset",
                (
                    "slow" if crf <= 14 else "medium"
                ),  # QSV preset: veryfast, faster, fast, medium, slow, slower, veryslow
                "-look_ahead",
                "1",  # Enable look-ahead for better quality
            ]
        )
        LOG.info(
            "[enhance-video] Using Intel QSV encoding (quality=%d, preset=%s)",
            qsv_quality,
            "slow" if crf <= 14 else "medium",
        )
    elif codec == "h264_nvenc":
        # NVIDIA NVENC parameters
        cmd.extend(
            [
                "-crf",
                str(crf),
                "-preset",
                "slow" if crf <= 14 else "medium",
                "-tune",
                "hq",  # High quality tuning for NVENC
            ]
        )
        LOG.info("[enhance-video] Using NVIDIA NVENC encoding")
    else:
        # libx264 (CPU) parameters
        cmd.extend(
            [
                "-crf",
                str(crf),
                "-preset",
                "slow" if crf <= 14 else "medium",
                "-tune",
                "film",
                "-profile:v",
                "high",
                "-level",
                "4.2",
            ]
        )
        LOG.info("[enhance-video] Using CPU encoding (libx264)")

    cmd.extend(
        [
            "-pix_fmt",
            "yuv420p",  # Ensure proper pixel format
            "-c:a",
            "copy",
            str(out_path),
        ]
    )

    LOG.debug("[enhance-video] FFmpeg command: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        # Check if QSV/NVENC failed due to hardware unavailability
        stderr_lower = proc.stderr.lower()
        is_hw_error = any(
            err in stderr_lower
            for err in [
                "no qsv device",
                "qsv not available",
                "failed to initialize",
                "no nvenc device",
                "nvenc not available",
                "no device found",
            ]
        )

        if is_hw_error and codec in ("hevc_qsv", "h264_nvenc"):
            # Fall back to CPU encoding - rebuild command with libx264
            LOG.warning(
                "[enhance-video] GPU encoding failed, falling back to CPU (libx264)"
            )
            # Remove QSV/NVENC specific params and add libx264 params
            cmd = [
                "ffmpeg",
                "-y",
                "-err_detect",
                "ignore_err",
                "-fflags",
                "+genpts+igndts",
                "-i",
                str(scene_path),
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                "slow" if crf <= 14 else "medium",
                "-tune",
                "film",
                "-profile:v",
                "high",
                "-level",
                "4.2",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                str(out_path),
            ]
            LOG.info("[enhance-video] Retrying with CPU encoding (libx264)")
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if proc.returncode != 0:
                error_lines, decode_warnings = filter_ffmpeg_errors(
                    proc.stderr, context=scene_path.name
                )
                if error_lines:
                    error_msg = "\n".join(error_lines[:10])
                    LOG.error(
                        "[enhance-video] FFmpeg failed (CPU fallback) for %s: %s",
                        scene_path.name,
                        error_msg,
                    )
                    raise RuntimeError(
                        f"ffmpeg video enhance failed (CPU fallback) ({proc.returncode}): {error_msg}"
                    )
                else:
                    # Only decode warnings, but still failed - log full stderr
                    error_msg = proc.stderr[:400]
                    LOG.error(
                        "[enhance-video] FFmpeg failed (CPU fallback) for %s (decode warnings only): %s",
                        scene_path.name,
                        error_msg,
                    )
                    raise RuntimeError(
                        f"ffmpeg video enhance failed (CPU fallback) ({proc.returncode}): {error_msg}"
                    )
            return out_path

        # Filter out common decode warnings before reporting error
        error_lines, decode_warnings = filter_ffmpeg_errors(
            proc.stderr, context=scene_path.name
        )
        if error_lines:
            error_msg = "\n".join(error_lines[:10])
            LOG.error(
                "[enhance-video] FFmpeg failed for %s: %s", scene_path.name, error_msg
            )
            raise RuntimeError(
                f"ffmpeg video enhance failed ({proc.returncode}): {error_msg}"
            )
        else:
            # Only decode warnings, but still failed - log full stderr
            error_msg = proc.stderr[:400]
            LOG.error(
                "[enhance-video] FFmpeg failed for %s (decode warnings only): %s",
                scene_path.name,
                error_msg,
            )
            raise RuntimeError(
                f"ffmpeg video enhance failed ({proc.returncode}): {error_msg}"
            )
    return out_path


def _enhance_audio(scene_path: Path, denoise_db: float, bitrate: str) -> Path:
    out_path = scene_path.with_name(
        scene_path.stem + "_audiodenoise" + scene_path.suffix
    )
    if out_path.exists():
        LOG.info("Enhanced audio already exists, skipping: %s", out_path.name)
        return out_path

    afftdn = f"afftdn=nf={denoise_db}"
    cmd = [
        "ffmpeg",
        "-y",
        "-err_detect",
        "ignore_err",  # Ignore decode errors in corrupted streams
        "-fflags",
        "+genpts+igndts",  # Generate missing timestamps, ignore DTS errors
        "-i",
        str(scene_path),
        "-af",
        afftdn,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        bitrate,
        str(out_path),
    ]
    LOG.info("[enhance-audio] %s", scene_path.name)
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        # Filter out common decode warnings before reporting error
        error_lines, decode_warnings = filter_ffmpeg_errors(
            proc.stderr, context=scene_path.name
        )
        if error_lines:
            error_msg = "\n".join(error_lines[:10])
            LOG.error(
                "[enhance-audio] FFmpeg failed for %s: %s", scene_path.name, error_msg
            )
            raise RuntimeError(
                f"ffmpeg audio enhance failed ({proc.returncode}): {error_msg}"
            )
        else:
            # Only decode warnings, but still failed - log full stderr
            error_msg = proc.stderr[:400]
            LOG.error(
                "[enhance-audio] FFmpeg failed for %s (decode warnings only): %s",
                scene_path.name,
                error_msg,
            )
            raise RuntimeError(
                f"ffmpeg audio enhance failed ({proc.returncode}): {error_msg}"
            )
    return out_path


def _check_codec_available(codec: str) -> bool:
    """Check if a codec is available in FFmpeg (cached)."""
    if codec in _CODEC_CACHE:
        return _CODEC_CACHE[codec]

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        available = codec in result.stdout
        _CODEC_CACHE[codec] = available
        if available:
            LOG.debug("Codec '%s' is available", codec)
        else:
            LOG.debug("Codec '%s' is not available", codec)
        return available
    except Exception as e:
        LOG.debug("Failed to check codec '%s': %s", codec, e)
        _CODEC_CACHE[codec] = False
        return False


def _pick_video_codec(
    user_codec: str, backend_name: str, hw: Dict[str, Any] = None
) -> str:
    """
    Select the best video codec based on hardware hierarchy.

    Hierarchy (best to worst):
    1. hevc_qsv (Intel Arc QSV - best quality, hardware accelerated)
    2. h264_qsv (Intel Arc QSV - H.264 variant)
    3. h264_nvenc (NVIDIA NVENC - hardware accelerated)
    4. libx264 (CPU - fallback, best compatibility)

    Args:
        user_codec: User-specified codec (or "auto"/"auto_gpu")
        backend_name: Detected backend name
        hw: Hardware descriptor (optional, for better detection)

    Returns:
        Best available codec name
    """
    # If user explicitly specified a codec, use it (but warn if unavailable)
    if user_codec and user_codec.lower() not in ("auto", "auto_gpu"):
        if not _check_codec_available(user_codec):
            LOG.warning(
                "Requested codec '%s' not available, falling back to auto-selection",
                user_codec,
            )
        else:
            return user_codec

    # Hardware hierarchy: Intel Arc QSV > NVIDIA NVENC > CPU
    codec_priority = []

    # Check for Intel Arc GPUs (prefer HEVC, fallback to H.264)
    if backend_name == "intel_arc" or (hw and hw.get("intel_arc")):
        if _check_codec_available("hevc_qsv"):
            codec_priority.append(("hevc_qsv", "Intel Arc QSV (HEVC)"))
        elif _check_codec_available("h264_qsv"):
            codec_priority.append(("h264_qsv", "Intel Arc QSV (H.264)"))

    # Check for NVIDIA GPUs
    if backend_name == "cuda" or (hw and hw.get("nvidia")):
        if _check_codec_available("h264_nvenc"):
            codec_priority.append(("h264_nvenc", "NVIDIA NVENC"))
        elif _check_codec_available("hevc_nvenc"):
            codec_priority.append(("hevc_nvenc", "NVIDIA NVENC (HEVC)"))

    # CPU fallback (always available)
    codec_priority.append(("libx264", "CPU (libx264)"))

    # Select the best available codec
    selected_codec, selected_name = codec_priority[0]
    LOG.info(
        "Codec selection: %s (from %d options)", selected_name, len(codec_priority)
    )

    return selected_codec


def process_scene(scene_path: Path, model, task: str, args, skip_done: bool = True) -> bool:
    """
    Dispatch scene processing based on task type.

    Args:
        scene_path: Path to scene file
        model: Loaded model (for subtitles task)
        task: Task name
        args: CLI arguments
        skip_done: If True, skip already-processed scenes (default: True)

    Returns:
        True if processed successfully, False if skipped or failed
    """
    # Get output directory and scenes folder from args if specified
    output_dir = getattr(args, 'run_output_dir', None)
    scenes_folder = getattr(args, 'scenes_folder', None)
    if scenes_folder:
        scenes_folder = Path(scenes_folder)
    
    # CHECKPOINT: Check if already done
    if skip_done and is_scene_done(scene_path, task, output_dir, scenes_folder):
        LOG.info("[skip] Already processed: %s (task=%s)", scene_path.name, task)
        return False

    # Process the scene
    success = False
    if task == "subtitles":
        transcribe_scene(scene_path, model, output_dir, scenes_folder)
        success = True
    elif task == "faces":
        _run_face_tracking(scene_path, args.frame_stride, args.min_face_score)
        success = True
    elif task == "faces_export":
        _export_faces(
            scene_path,
            args.frame_stride,
            args.min_face_score,
            args.faces_outdir,
            args.run_name,
            args.face_crop_size,
            args.face_max_per_track,
        )
        success = True
    elif task == "video_enhance":
        hw = getattr(args, "hardware", None)
        codec = _pick_video_codec(args.video_codec, args.backend_name, hw)
        LOG.info(
            "[video_enhance] Selected codec: %s (backend: %s)", codec, args.backend_name
        )
        _enhance_video(
            scene_path,
            args.enhance_scale,
            args.enhance_denoise,
            codec,
            args.enhance_crf,
            force=False,
        )
        success = True
    elif task == "audio_enhance":
        _enhance_audio(scene_path, args.audio_denoise, args.audio_bitrate)
        success = True
    else:
        raise NotImplementedError(f"Task '{task}' not implemented")

    # CHECKPOINT: Mark as done after successful processing
    if success:
        mark_scene_done(scene_path, task, output_dir, scenes_folder)

    return success


def process_batch(device, scene_list, backend: str, task: str, progress_queue=None) -> None:
    if not scene_list:
        return

    # Log worker identity for debugging GPU affinity
    import threading
    worker_id = threading.current_thread().name
    pid = os.getpid()
    device_index = device.get("index", "N/A")
    device_name = device.get("name", device.get("type", "unknown"))
    ze_mask = os.environ.get("ZE_AFFINITY_MASK", "not set")

    LOG.info(
        "[worker-start] PID=%d | thread=%s | backend=%s | device_idx=%s (%s) | ZE_AFFINITY_MASK=%s",
        pid,
        worker_id,
        backend,
        device_index,
        device_name,
        ze_mask,
    )

    model = _load_model_for_device(device, backend) if task == "subtitles" else None

    # ========== PATCH #2: STORE MODEL FOR CLEANUP ==========
    # Store model in module-level variable so signal handlers can access it
    global _worker_model
    _worker_model = model
    # ========== END PATCH #2 ==========

    label = device["name"] if device.get("name") else device.get("type", "device")

    # Stats tracking
    stats = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
    }

    for scene in scene_list:
        # Only log at INFO for new processing, DEBUG for skipped
        LOG.debug("[%s] %s: %s", label, task, scene.name)
        try:
            was_processed = process_scene(scene, model, task, _cli_args, skip_done=True)
            if was_processed:
                stats["processed"] += 1
                # Log successful processing at INFO level
                LOG.info("[%s] Processed: %s", label, scene.name)
            else:
                stats["skipped"] += 1
                # Skipped files already logged at INFO in process_scene
        except Exception as exc:
            LOG.error("[%s] Failed: %s -> %s", label, scene, exc)
            stats["failed"] += 1
        finally:
            # Send progress update via Queue (multiprocessing-safe)
            if progress_queue is not None:
                try:
                    progress_queue.put({
                        "type": "progress",
                        "scene": scene.name,
                        "pid": pid,
                        "device": device_name,
                    }, block=False)
                except Exception as e:
                    LOG.debug("Failed to send progress update: %s", e)

    # Log final stats
    LOG.info(
        "[worker-done] PID=%d | Processed: %d | Skipped: %d | Failed: %d",
        pid,
        stats["processed"],
        stats["skipped"],
        stats["failed"],
    )

    # Cleanup GPU resources before worker termination
    if model is not None:
        try:
            LOG.info("[worker-cleanup] PID=%d | Cleaning up GPU resources for model", pid)
            _cleanup_gpu_resources(model, backend, device)
            LOG.info("[worker-cleanup] PID=%d | GPU resources cleaned up successfully", pid)
        except Exception as exc:
            LOG.warning("[worker-cleanup] PID=%d | GPU cleanup failed (non-fatal): %s", pid, exc)


def gather_scenes(scenes_folder: Path, limit: Optional[int] = None, filter_variants: bool = True):
    """
    Gather scene files from the scenes folder.
    
    Args:
        scenes_folder: Folder containing scene files
        limit: Optional limit on number of scenes
        filter_variants: If True, skip duplicate/variant files (enhanced, denoised, RIFE, etc.)
    
    Returns:
        List of scene file paths
    """
    scenes = []
    if not scenes_folder.exists():
        return scenes

    # Patterns to identify variant/duplicate files that should be skipped
    # These are typically intermediate processing outputs, not original scenes
    variant_patterns = [
        r'_enhanced',           # Enhanced versions
        r'_audiodenoise',       # Audio denoised versions
        r'-2x-RIFE',            # RIFE frame interpolation variants
        r'-8x-RIFE',            # RIFE frame interpolation variants
        r'\.rife\.',            # RIFE processed files
        r'\.realesrgan\.',      # RealESRGAN upscaled files
        r'_comparison\.',       # Comparison videos
        r'_mp4_CFR\.',          # Constant frame rate conversions
        r'\.old\.',             # Old/backup files
        r'\(1\)', r'\(2\)', r'\(3\)', r'\(4\)',  # Enumerated duplicates
        r'\.libplacebo\.',      # Libplacebo processed
    ]
    
    def is_variant_file(path: Path) -> bool:
        """Check if a file is a variant/duplicate that should be skipped."""
        if not filter_variants:
            return False
        name_lower = path.name.lower()
        return any(re.search(pattern, name_lower) for pattern in variant_patterns)

    # Allow either flat scene files or subfolders of scenes (original structure).
    flat_files = sorted(
        p for p in scenes_folder.iterdir() 
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not is_variant_file(p)
    )
    scenes.extend(flat_files)

    for video_folder in sorted(p for p in scenes_folder.iterdir() if p.is_dir()):
        folder_scenes = [
            p for p in video_folder.iterdir() 
            if p.suffix.lower() in VIDEO_EXTS and not is_variant_file(p)
        ]
        scenes.extend(sorted(folder_scenes))

    # Apply limit if specified
    if limit is not None and limit > 0:
        scenes = scenes[:limit]

    return scenes


def _query_gpu_wmi():
    if not _HAS_WMI:
        return []
    try:
        c = wmi.WMI(namespace="root\\cimv2")
        engines = c.Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine()
        mems = c.Win32_PerfFormattedData_GPUPerformanceCounters_GPUMemory()
        mem_by_adapter = {}
        for m in mems:
            key = (m.AdapterLUID, m.NodeLUID)
            mem_by_adapter[key] = (m.LocalUsage, m.LocalMemoryTotal)

        usage = []
        for e in engines:
            if "3D" not in e.Name:
                continue
            key = (e.AdapterLUID, e.NodeLUID)
            mem_usage = mem_by_adapter.get(key, (None, None))
            usage.append(
                {
                    "adapter": e.AdapterLUID,
                    "node": e.NodeLUID,
                    "util": float(e.UtilizationPercentage),
                    "mem_used": mem_usage[0],
                    "mem_total": mem_usage[1],
                }
            )
        return usage
    except Exception:
        return []


def _fmt_gpu_rows(rows):
    out = []
    for r in rows:
        mem = ""
        if r["mem_used"] is not None and r["mem_total"]:
            mem = f"{r['mem_used']} / {r['mem_total']} MB"
        out.append(f"GPU{r['node']} {r['util']:4.1f}% VRAM {mem}")
    return "; ".join(out) if out else "GPU util unavailable (install 'wmi'?)"


def monitor(total_scenes: int, stop_event: Event, interval: int) -> None:
    start = time.time()
    while not stop_event.wait(interval):
        with _progress_lock:
            done = _processed
        elapsed = time.time() - start
        avg = elapsed / done if done else 0
        remaining = total_scenes - done
        eta = avg * remaining if done else 0
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        gpu_rows = _fmt_gpu_rows(_query_gpu_wmi())
        LOG.info(
            "[monitor] %s/%s | elapsed %.1fm | eta %.1fm | CPU %.1f%% | RAM %.1f%% (%.1fGB/%.1fGB) | %s",
            done,
            total_scenes,
            elapsed / 60,
            eta / 60,
            cpu,
            mem.percent,
            mem.used / 1e9,
            mem.total / 1e9,
            gpu_rows,
        )


def main(args) -> int:
    # Handle faces_cluster as a special case (doesn't process scenes)
    if args.task == "faces_cluster":
        if not _check_requirements("faces_cluster"):
            return 1

        # Cluster all runs in a directory if requested
        if args.cluster_all_runs:
            if not args.cluster_run_folder:
                # Default to faces_export directory
                scenes_folder = Path(args.scenes_folder)
                faces_export_dir = scenes_folder.parent / "faces_export"
                if not faces_export_dir.exists():
                    LOG.error(
                        "Faces export directory does not exist: %s", faces_export_dir
                    )
                    return 1
            else:
                faces_export_dir = Path(args.cluster_run_folder)

            # Find all run folders (directories starting with "run-")
            run_folders = [
                d
                for d in faces_export_dir.iterdir()
                if d.is_dir() and d.name.startswith("run-")
            ]
            if not run_folders:
                LOG.warning("No run folders found in %s", faces_export_dir)
                return 1

            LOG.info(
                "Clustering %d run folders in %s", len(run_folders), faces_export_dir
            )
            for run_folder in sorted(run_folders):
                clustered_dir = run_folder / "clustered"
                # Skip if already clustered (unless force)
                if clustered_dir.exists():
                    if args.force_recluster:
                        LOG.info(
                            "Force re-clustering %s (removing existing clusters)",
                            run_folder.name,
                        )
                        shutil.rmtree(clustered_dir)
                    else:
                        LOG.info(
                            "Skipping %s (already clustered, use --force-recluster to overwrite)",
                            run_folder.name,
                        )
                        continue
                LOG.info("Clustering %s", run_folder.name)
                try:
                    _cluster_faces(
                        run_folder,
                        args.cluster_similarity,
                        args.cluster_min_samples,
                        args.cluster_method,
                        args.cluster_quality_threshold,
                        args.cluster_n_passes,
                    )
                except Exception as exc:
                    LOG.error("Failed to cluster %s: %s", run_folder.name, exc)
            return 0

        # Cluster a specific run folder
        if not args.cluster_run_folder:
            LOG.error(
                "faces_cluster task requires --cluster-run-folder (or use --cluster-all-runs)"
            )
            return 1
        run_folder = Path(args.cluster_run_folder)
        if not run_folder.exists():
            LOG.error("Run folder does not exist: %s", run_folder)
            return 1
        clustered_dir = run_folder / "clustered"
        if clustered_dir.exists() and args.force_recluster:
            LOG.info("Force re-clustering (removing existing clusters)")
            shutil.rmtree(clustered_dir)
        elif clustered_dir.exists():
            LOG.warning(
                "Run folder already clustered. Use --force-recluster to overwrite."
            )
        _cluster_faces(
            run_folder,
            args.cluster_similarity,
            args.cluster_min_samples,
            args.cluster_method,
            args.cluster_quality_threshold,
            args.cluster_n_passes,
        )
        return 0
    global _cli_args
    _cli_args = args
    scenes_folder = Path(args.scenes_folder)
    if args.faces_outdir:
        args.faces_outdir = Path(args.faces_outdir)
    else:
        args.faces_outdir = scenes_folder.parent / "faces_export"
    if not args.run_name:
        args.run_name = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    
    # Set up run-specific output directory (opt-in via --run-dir)
    args.run_output_dir = Path(args.run_dir) if args.run_dir else None

    # Create run output directory if it doesn't exist
    if args.run_output_dir:
        args.run_output_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("Run output directory: %s", args.run_output_dir)
    else:
        LOG.info("Run output directory: using source scene folders (checkpoint reuse enabled)")

    if not _check_requirements(args.task):
        return 1

    hw = detect_hardware()
    backend = select_backend(hw)
    args.backend_name = backend["backend"]
    args.hardware = hw  # Store hardware descriptor for codec selection

    # Show detailed hardware info
    gpu_names = [g["name"] for g in hw["gpus"]] if hw["gpus"] else ["none"]
    if not _HAS_WMI:
        error_msg = getattr(globals(), "_WMI_ERROR", "unknown error")
        LOG.warning(
            "WMI not available - GPU detection limited. Install with: pip install wmi (error: %s)",
            error_msg,
        )
    LOG.info(
        "Hardware detected: CPU %s | GPUs %s | backend=%s",
        hw["cpu_count"],
        gpu_names,
        backend["backend"],
    )
    if hw["intel_arc"]:
        LOG.info(
            "Intel Arc GPUs detected: %d device(s) - %s",
            len(hw["intel_arc"]),
            ", ".join(
                [
                    f"{g['name']} (ze_index={g.get('ze_index', 'N/A')})"
                    for g in hw["intel_arc"]
                ]
            ),
        )
    suggestions = suggest_libraries(backend)
    if suggestions:
        LOG.info("Suggested installs for this hardware:")
        for s in suggestions:
            LOG.info("  %s", s)

    # If a specific scene file is requested, process only that file
    if args.scene_file:
        scene_path = Path(args.scene_file)
        if not scene_path.exists():
            LOG.error("Scene file not found: %s", scene_path)
            return 1
        if scene_path.suffix.lower() not in VIDEO_EXTS:
            LOG.error("Scene file is not a video: %s", scene_path)
            return 1
        scenes = [scene_path]
        LOG.info("Processing single scene file: %s", scene_path.name)
    else:
        # Count all files before filtering for logging
        all_files = []
        if scenes_folder.exists():
            for p in scenes_folder.rglob("*"):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    all_files.append(p)
        
        scenes = gather_scenes(scenes_folder, limit=args.limit, filter_variants=True)
        
        # Log filtering statistics
        filtered_count = len(all_files) - len(scenes) if all_files else 0
        if filtered_count > 0:
            LOG.info("Filtered out %d variant/duplicate files (enhanced, denoised, RIFE, etc.)", filtered_count)
            LOG.info("Processing %d base scene files (out of %d total files)", len(scenes), len(all_files))
    
    if not scenes:
        LOG.warning("No scenes found in %s", scenes_folder)
        return 1

    plan = build_worker_plan(scenes, backend)
    if not plan:
        LOG.error("No eligible hardware detected; nothing to do.")
        return 1

    total = len(scenes)
    LOG.info(
        "Total scenes: %s | workers: %s",
        total,
        [p["device"].get("name", p["device"]["type"]) for p in plan],
    )

    if args.dry_run:
        LOG.info("Dry-run requested; exiting before processing.")
        return 0

    # Use spawn context for Windows compatibility and clean process separation
    mp_ctx = multiprocessing.get_context("spawn")

    # Create progress queue for IPC (no shared mutable globals!)
    progress_queue = mp_ctx.Queue()

    progress_log_handle = None
    if getattr(args, "progress_log", None):
        try:
            progress_log_handle = open(args.progress_log, "a", encoding="utf-8")
        except Exception as exc:
            LOG.error("Could not open progress log %s: %s", args.progress_log, exc)
            progress_log_handle = None

    def _write_progress(payload: Dict[str, Any]) -> None:
        if not progress_log_handle:
            return
        try:
            payload_with_ts = {"ts": time.time(), **payload}
            progress_log_handle.write(json.dumps(payload_with_ts) + "\n")
            progress_log_handle.flush()
        except Exception as exc:
            LOG.debug("Failed to write progress log: %s", exc)

    # Import worker_bootstrap for multiprocessing workers
    import worker_bootstrap

    workers = [
        mp_ctx.Process(
            target=worker_bootstrap.worker_main,
            args=(p["device"], p["scenes"], backend["backend"], args.task, args, progress_queue),
        )
        for p in plan
        if p["scenes"]
    ]

    LOG.info("Starting %d worker processes...", len(workers))

    for w in workers:
        w.start()

    # Monitor progress via Queue (non-blocking read)
    import queue
    processed_count = 0
    start_time = time.time()

    while any(w.is_alive() for w in workers):
        try:
            # Non-blocking read with short timeout
            msg = progress_queue.get(timeout=0.5)
            if msg.get("type") == "progress":
                processed_count += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / processed_count if processed_count > 0 else 0
                remaining = total - processed_count
                eta = avg_time * remaining if processed_count > 0 else 0

                # Log progress updates less frequently (every 10 files or every 60 seconds)
                should_log = (processed_count % 10 == 0) or (elapsed % 60 < 0.5)
                if should_log:
                    LOG.info(
                        "[progress] PID=%d scene=%s (%d/%d) | elapsed=%.1fs eta=%.1fs",
                        msg.get("pid"),
                        msg.get("scene"),
                        processed_count,
                        total,
                        elapsed,
                        eta,
                    )
                else:
                    LOG.debug(
                        "[progress] PID=%d scene=%s (%d/%d) | elapsed=%.1fs eta=%.1fs",
                        msg.get("pid"),
                        msg.get("scene"),
                        processed_count,
                        total,
                        elapsed,
                        eta,
                    )
                _write_progress(
                    {
                        "type": "progress",
                        "pid": msg.get("pid"),
                        "scene": msg.get("scene"),
                        "done": processed_count,
                        "total": total,
                        "elapsed_s": elapsed,
                        "eta_s": eta,
                    }
                )
        except queue.Empty:
            continue

    # Drain any remaining messages after workers finish
    while not progress_queue.empty():
        try:
            msg = progress_queue.get(timeout=0.1)
            if msg.get("type") == "progress":
                processed_count += 1
                LOG.info(
                    "[progress] PID=%d scene=%s (%d/%d)",
                    msg.get("pid"),
                    msg.get("scene"),
                    processed_count,
                    total,
                )
                _write_progress(
                    {
                        "type": "progress",
                        "pid": msg.get("pid"),
                        "scene": msg.get("scene"),
                        "done": processed_count,
                        "total": total,
                    }
                )
        except queue.Empty:
            break

    # Wait for all workers to complete with timeout and logging
    LOG.info("[main] Waiting for %d worker process(es) to complete...", len(workers))
    worker_timeout = 600  # 10 minutes timeout per worker (allows for model download + loading + processing)
    start_join_time = time.time()
    failed_workers = []
    
    for idx, w in enumerate(workers):
        worker_pid = w.pid if hasattr(w, 'pid') else 'unknown'
        LOG.info("[main] Waiting for worker %d/%d (PID=%s) to complete...", idx + 1, len(workers), worker_pid)
        worker_start_time = time.time()
        
        # Check if worker is still alive before joining
        if not w.is_alive():
            LOG.info("[main] Worker %d (PID=%s) already completed", idx + 1, worker_pid)
            w.join(timeout=1.0)  # Quick join to clean up
        else:
            # Join with timeout
            w.join(timeout=worker_timeout)
            worker_elapsed = time.time() - worker_start_time
            
            if w.is_alive():
                LOG.error(
                    "[main] Worker %d (PID=%s) did not complete within %d seconds (elapsed: %.1fs). "
                    "This may indicate a stuck process or slow model loading. "
                    "Consider checking GPU resources or increasing timeout.",
                    idx + 1, worker_pid, worker_timeout, worker_elapsed
                )
                # Try to terminate the worker
                try:
                    LOG.warning("[main] Attempting to terminate worker %d (PID=%s)", idx + 1, worker_pid)
                    w.terminate()
                    w.join(timeout=30.0)  # Give it 30 seconds to terminate gracefully
                    if w.is_alive():
                        LOG.error("[main] Worker %d (PID=%s) did not terminate, may need manual cleanup", idx + 1, worker_pid)
                    else:
                        LOG.info("[main] Worker %d (PID=%s) terminated successfully", idx + 1, worker_pid)
                except Exception as exc:
                    LOG.error("[main] Failed to terminate worker %d (PID=%s): %s", idx + 1, worker_pid, exc)
                failed_workers.append({"pid": worker_pid, "reason": "timeout"})
                continue
            else:
                LOG.info("[main] Worker %d (PID=%s) completed successfully (join time: %.1fs)", idx + 1, worker_pid, worker_elapsed)

        # Inspect exit codes after join/terminate
        if hasattr(w, "exitcode"):
            if w.exitcode is None:
                failed_workers.append({"pid": worker_pid, "reason": "no-exitcode"})
            elif w.exitcode != 0:
                failed_workers.append({"pid": worker_pid, "reason": f"exitcode {w.exitcode}"})
                LOG.error("[main] Worker %d (PID=%s) exited with code %s", idx + 1, worker_pid, w.exitcode)
    
    total_join_time = time.time() - start_join_time
    LOG.info("[main] All worker joins completed in %.1f seconds", total_join_time)

    _write_progress(
        {
            "type": "summary",
            "total": total,
            "processed": processed_count,
            "duration_s": time.time() - start_time,
        }
    )

    if progress_log_handle:
        try:
            progress_log_handle.close()
        except Exception:
            pass

    rc = 0
    if failed_workers:
        rc = 1
        LOG.error("[main] %d worker(s) failed: %s", len(failed_workers), failed_workers)

    if processed_count < total:
        rc = 1
        LOG.error("[main] Incomplete processing: processed %d of %d scenes", processed_count, total)

    LOG.info("Processing complete (rc=%d).", rc)
    if rc != 0:
        return rc

    # Auto-cluster faces after export if requested (only if main processing succeeded)
    if rc == 0 and args.task == "faces_export" and args.auto_cluster:
        run_folder = args.faces_outdir / args.run_name
        if run_folder.exists():
            LOG.info("Auto-clustering faces in %s", run_folder)
            if not _check_requirements("faces_cluster"):
                LOG.warning("Clustering requirements not met, skipping auto-cluster")
            else:
                try:
                    _cluster_faces(
                        run_folder,
                        args.cluster_similarity,
                        args.cluster_min_samples,
                        args.cluster_method,
                        args.cluster_quality_threshold,
                        args.cluster_n_passes,
                    )
                except Exception as exc:
                    LOG.error("Auto-clustering failed: %s", exc)

    LOG.info("Processing complete (rc=%d).", rc)
    return rc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Parallel scene processing")
    parser.add_argument(
        "--scenes-folder",
        default=str(DEFAULT_SCENES_FOLDER),
        help="Folder containing scene clips",
    )
    parser.add_argument(
        "--scene-file",
        default=None,
        help="Process only a specific scene file (full path). If provided, only this file will be processed.",
    )
    parser.add_argument(
        "--task",
        default="subtitles",
        choices=[
            "subtitles",
            "faces",
            "faces_export",
            "faces_cluster",
            "video_enhance",
            "audio_enhance",
        ],
        help="Processing task to run per scene",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=12,
        help="Process every Nth frame for face detection (faces task). Default 12 for better performance.",
    )
    parser.add_argument(
        "--min-face-score",
        type=float,
        default=0.6,
        help="Minimum face confidence (faces task)",
    )
    parser.add_argument(
        "--faces-outdir",
        default=None,
        help="Output root for face crops (faces_export task). Defaults to sibling 'faces_export' next to scenes-folder.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name for face export subfolder (defaults to timestamp).",
    )
    parser.add_argument(
        "--face-crop-size",
        type=int,
        default=256,
        help="Output size for face crops (faces_export task).",
    )
    parser.add_argument(
        "--face-max-per-track",
        type=int,
        default=3,
        help="Max crops to save per track (faces_export task). Default 3 to reduce storage.",
    )
    parser.add_argument(
        "--cluster-run-folder",
        default=None,
        help="Run folder path for faces_cluster task (e.g., faces_export/run-20231211-123456)",
    )
    parser.add_argument(
        "--cluster-similarity",
        type=float,
        default=0.90,
        help="Cosine similarity threshold for clustering (faces_cluster task, 0.0-1.0). Default 0.90 for strict matching with ArcFace (higher = stricter matching)",
    )
    parser.add_argument(
        "--cluster-min-samples",
        type=int,
        default=5,
        help="Minimum samples per cluster (faces_cluster task). Default 5 for more reliable clusters.",
    )
    parser.add_argument(
        "--auto-cluster",
        action="store_true",
        help="Automatically cluster faces after faces_export completes",
    )
    parser.add_argument(
        "--cluster-all-runs",
        action="store_true",
        help="Cluster all run folders in faces_export directory (faces_cluster task)",
    )
    parser.add_argument(
        "--force-recluster",
        action="store_true",
        help="Force re-clustering even if clustered folder already exists (faces_cluster task)",
    )
    parser.add_argument(
        "--cluster-method",
        default="hierarchical",
        choices=["dbscan", "hierarchical"],
        help="Clustering method: dbscan (density-based) or hierarchical (shows dendrogram). Default hierarchical for better accuracy.",
    )
    parser.add_argument(
        "--cluster-quality-threshold",
        type=float,
        default=0.3,
        help="Minimum face quality score (0.0-1.0) to include in clustering (filters blur/poor lighting)",
    )
    parser.add_argument(
        "--cluster-n-passes",
        type=int,
        default=3,
        help="Number of clustering passes for hierarchical: 1=ward only, 3=tries ward/average/complete and selects best",
    )
    parser.add_argument(
        "--enhance-scale",
        type=float,
        default=2.0,
        help="Video upscale factor (video_enhance task). Default 2.0 (2x upscale). Use 1.0 for no scaling.",
    )
    parser.add_argument(
        "--enhance-denoise",
        type=float,
        default=1.0,
        help="Denoise strength for video (hqdn3d) (video_enhance task). Default 1.0 for balanced quality.",
    )
    parser.add_argument(
        "--enhance-crf",
        type=int,
        default=16,
        help="CRF quality for video encode (video_enhance task, lower=better quality, 16-28 typical). Default 16 for high quality.",
    )
    parser.add_argument(
        "--video-codec",
        default="auto",
        help="Video codec for enhanced output (auto, auto_gpu, libx264, hevc_qsv, h264_nvenc). 'auto' selects best codec based on detected hardware.",
    )
    parser.add_argument(
        "--audio-denoise",
        type=float,
        default=-25.0,
        help="Noise floor (dB) for afftdn (audio_enhance task)",
    )
    parser.add_argument(
        "--audio-bitrate",
        default="192k",
        help="Audio bitrate for re-encode (audio_enhance task)",
    )
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=10,
        help="Seconds between monitor updates",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List work plan and exit without processing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (use -vv for debug)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to log file (logs all levels to file for debugging, console respects verbosity)",
    )
    parser.add_argument(
        "--progress-log",
        type=str,
        default=None,
        help="Optional path to JSONL progress log (one line per progress event)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to first N scenes (useful for testing)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional run-specific output directory (creates timestamped directory if not specified). Outputs (SRT, .done files) will be written here instead of next to source videos.",
    )
    return parser.parse_args(argv)


def configure_logging(verbosity: int, log_file: Optional[str] = None) -> None:
    """
    Configure logging with optional file output.

    Args:
        verbosity: Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG)
        log_file: Optional path to log file (logs all levels to file, respects verbosity for console)
    """
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
        LOG.info("Logging to file: %s", log_file)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    cli_args = parse_args()
    configure_logging(cli_args.verbose, log_file=getattr(cli_args, "log_file", None))
    sys.exit(main(cli_args))

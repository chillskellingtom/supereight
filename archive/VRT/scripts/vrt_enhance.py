"""
VRT (Video Restoration Transformer) integration for video enhancement.

This script integrates VRT models for:
- Video Super-Resolution
- Video Deblurring  
- Video Denoising
- Video Frame Interpolation
- Space-Time Super-Resolution

Usage:
    python vrt_enhance.py --task videosr --input video.mp4 --output output.mp4
"""
import argparse
import cv2
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import torch
import numpy as np

# Initialize oneAPI environment for Intel GPU support
def _init_oneapi_env():
    """Initialize oneAPI environment variables for Intel GPU support."""
    # Windows: optionally run Intel oneAPI setvars.bat to populate PATH/DLL search paths.
    # Linux/WSL: do not attempt to run setvars.bat (it won't exist); rely on pip/apt installs.
    if os.name == "nt":
        oneapi_setvars = Path(r"C:\Program Files (x86)\Intel\oneAPI\setvars.bat")
        if oneapi_setvars.exists():
            # Run setvars.bat and capture environment
            result = subprocess.run(
                f'"{oneapi_setvars}" && set',
                shell=True,
                capture_output=True,
                text=True
            )
            # Parse environment variables from output
            for line in result.stdout.splitlines():
                if '=' in line and not line.startswith('::'):
                    try:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
                    except Exception:
                        pass
            LOG.debug("oneAPI environment initialized (Windows)")
        else:
            LOG.warning("oneAPI setvars.bat not found - Intel GPU may not work on Windows")
    
    # Set Intel GPU environment variables
    os.environ.setdefault("ZE_AFFINITY_MASK", "0,1")  # Use both GPUs
    os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")
    os.environ.setdefault("SYCL_CACHE_DIR", str(Path.home() / ".cache" / "intel_gpu_cache"))

# Add VRT to path
def _resolve_vrt_path() -> Path:
    """
    Resolve VRT repository path.

    Priority:
    1) $VRT_PATH environment variable
    2) ./VRT next to this script (repo-local vendor checkout)
    3) ./VRT in current working directory
    4) ~/VRT (common manual clone location)
    5) (Windows-only) C:\\Users\\latch\\VRT legacy path
    """
    env_path = os.environ.get("VRT_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    candidates = [
        Path(__file__).parent / "VRT",
        Path.cwd() / "VRT",
        Path.home() / "VRT",
    ]
    if os.name == "nt":
        candidates.append(Path(r"C:\Users\latch\VRT"))

    for p in candidates:
        if p.exists():
            return p

    # Show the most actionable expected location first
    raise RuntimeError(
        "VRT repository not found. "
        "Clone it into a `VRT/` folder next to this script, or set environment variable VRT_PATH."
    )


VRT_PATH = _resolve_vrt_path()

# Insert VRT path at the beginning of sys.path BEFORE any imports
sys.path.insert(0, str(VRT_PATH))

try:
    from models.network_vrt import VRT as net
    from utils import utils_image as util
    from data.dataset_video_test import SingleVideoRecurrentTestDataset
    from torch.utils.data import DataLoader
except ImportError as e:
    raise RuntimeError(f"Failed to import VRT modules. Make sure VRT is installed and requirements are met: {e}\n"
                      f"Install with: pip install -r {VRT_PATH / 'requirements.txt'}")

LOG = logging.getLogger("vrt_enhance")

# Initialize oneAPI environment at module load
_init_oneapi_env()

# VRT Model configurations
VRT_MODELS = {
    "videosr_reds_6frames": {
        "task": "001_VRT_videosr_bi_REDS_6frames",
        "model_file": "001_VRT_videosr_bi_REDS_6frames.pth",
        "url": "https://github.com/JingyunLiang/VRT/releases/download/v0.0/001_VRT_videosr_bi_REDS_6frames.pth",
        "tile": [40, 128, 128],
        "tile_overlap": [2, 20, 20],
        "description": "Video Super-Resolution (REDS, 6 frames, bicubic)"
    },
    "videosr_reds_16frames": {
        "task": "002_VRT_videosr_bi_REDS_16frames",
        "model_file": "002_VRT_videosr_bi_REDS_16frames.pth",
        "url": "https://github.com/JingyunLiang/VRT/releases/download/v0.0/002_VRT_videosr_bi_REDS_16frames.pth",
        "tile": [40, 128, 128],
        "tile_overlap": [2, 20, 20],
        "description": "Video Super-Resolution (REDS, 16 frames, bicubic) - HIGH QUALITY"
    },
    "videosr_vimeo": {
        "task": "003_VRT_videosr_bi_Vimeo_7frames",
        "model_file": "003_VRT_videosr_bi_Vimeo_7frames.pth",
        "url": "https://github.com/JingyunLiang/VRT/releases/download/v0.0/003_VRT_videosr_bi_Vimeo_7frames.pth",
        "tile": [32, 128, 128],
        "tile_overlap": [2, 20, 20],
        "description": "Video Super-Resolution (Vimeo, 7 frames, bicubic)"
    },
    "videosr_vimeo_bd": {
        "task": "004_VRT_videosr_bd_Vimeo_7frames",
        "model_file": "004_VRT_videosr_bd_Vimeo_7frames.pth",
        "url": "https://github.com/JingyunLiang/VRT/releases/download/v0.0/004_VRT_videosr_bd_Vimeo_7frames.pth",
        "tile": [32, 128, 128],
        "tile_overlap": [2, 20, 20],
        "description": "Video Super-Resolution (Vimeo, blur-downsampling)"
    },
    "videodeblur_dvd": {
        "task": "005_VRT_videodeblurring_DVD",
        "model_file": "005_VRT_videodeblurring_DVD.pth",
        "url": "https://github.com/JingyunLiang/VRT/releases/download/v0.0/005_VRT_videodeblurring_DVD.pth",
        "tile": [12, 256, 256],
        "tile_overlap": [2, 20, 20],
        "description": "Video Deblurring (DVD dataset)"
    },
    "videodeblur_gopro": {
        "task": "006_VRT_videodeblurring_GoPro",
        "model_file": "006_VRT_videodeblurring_GoPro.pth",
        "url": "https://github.com/JingyunLiang/VRT/releases/download/v0.0/006_VRT_videodeblurring_GoPro.pth",
        "tile": [18, 192, 192],
        "tile_overlap": [2, 20, 20],
        "description": "Video Deblurring (GoPro dataset)"
    },
    "videodeblur_reds": {
        "task": "007_VRT_videodeblurring_REDS",
        "model_file": "007_VRT_videodeblurring_REDS.pth",
        "url": "https://github.com/JingyunLiang/VRT/releases/download/v0.0/007_VRT_videodeblurring_REDS.pth",
        "tile": [12, 256, 256],
        "tile_overlap": [2, 20, 20],
        "description": "Video Deblurring (REDS dataset)"
    },
    "videodenoise": {
        "task": "008_VRT_videodenoising_DAVIS",
        "model_file": "008_VRT_videodenoising_DAVIS.pth",
        "url": "https://github.com/JingyunLiang/VRT/releases/download/v0.0/008_VRT_videodenoising_DAVIS.pth",
        "tile": [12, 256, 256],
        "tile_overlap": [2, 20, 20],
        "description": "Video Denoising (DAVIS dataset)",
        "sigma": 10  # Default noise level
    },
    "videofi": {
        "task": "009_VRT_videofi_Vimeo_4frames",
        "model_file": "009_VRT_videofi_Vimeo_4frames.pth",
        "url": "https://github.com/JingyunLiang/VRT/releases/download/v0.0/009_VRT_videofi_Vimeo_4frames.pth",
        "tile": [0, 0, 0],
        "tile_overlap": [0, 0, 0],
        "description": "Video Frame Interpolation (Vimeo)"
    }
}

MODEL_CACHE = Path.home() / ".cache" / "vrt_models"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)


def download_model(model_key: str) -> Path:
    """Download VRT model if not cached."""
    if model_key not in VRT_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(VRT_MODELS.keys())}")
    
    config = VRT_MODELS[model_key]
    model_path = MODEL_CACHE / config["model_file"]
    
    if model_path.exists():
        LOG.info("Using cached model: %s", model_path)
        return model_path
    
    LOG.info("Downloading model: %s", config["description"])
    LOG.info("URL: %s", config["url"])
    
    import urllib.request
    try:
        urllib.request.urlretrieve(config["url"], model_path)
        LOG.info("Downloaded model to: %s", model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")
    
    return model_path


def prepare_model(model_key: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    """Load and prepare VRT model. Returns (model, config_dict)."""
    config = VRT_MODELS[model_key]
    model_path = download_model(model_key)
    task = config["task"]
    
    # Load model based on task (matching VRT main_test_vrt.py exactly)
    if task == '001_VRT_videosr_bi_REDS_6frames':
        model = net(upscale=4, img_size=[6,64,64], window_size=[6,8,8], 
                    depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
                    indep_reconsts=[11,12], 
                    embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
                    num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], 
                    pa_frames=2, deformable_groups=12)
        scale = 4
    elif task == '002_VRT_videosr_bi_REDS_16frames':
        model = net(upscale=4, img_size=[16,64,64], window_size=[8,8,8], 
                    depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
                    indep_reconsts=[11,12], 
                    embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
                    num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], 
                    pa_frames=6, deformable_groups=24)
        scale = 4
    elif task in ['003_VRT_videosr_bi_Vimeo_7frames', '004_VRT_videosr_bd_Vimeo_7frames']:
        model = net(upscale=4, img_size=[8,64,64], window_size=[8,8,8], 
                    depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
                    indep_reconsts=[11,12], 
                    embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
                    num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], 
                    pa_frames=4, deformable_groups=16)
        scale = 4
    elif task in ['005_VRT_videodeblurring_DVD', '006_VRT_videodeblurring_GoPro', '007_VRT_videodeblurring_REDS']:
        model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], 
                    depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[9,10], 
                    embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], 
                    pa_frames=2, deformable_groups=16)
        scale = 1
    elif task == '008_VRT_videodenoising_DAVIS':
        model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], 
                    depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[9,10], 
                    embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], 
                    pa_frames=2, deformable_groups=16,
                    nonblind_denoising=True)
        scale = 1
    elif task == '009_VRT_videofi_Vimeo_4frames':
        model = net(upscale=1, out_chans=3, img_size=[4,192,192], window_size=[4,8,8], 
                    depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[], 
                    embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], 
                    pa_frames=0)
        scale = 1
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'], strict=True)
    elif 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    
    model.eval()
    model = model.to(device)
    
    model_config = {
        'scale': scale,
        'window_size': model.window_size if hasattr(model, 'window_size') else None,
        'nonblind_denoising': getattr(model, 'nonblind_denoising', False)
    }
    
    LOG.info("Loaded model: %s", config["description"])
    return model, model_config


def extract_frames(video_path: Path, output_dir: Path) -> Tuple[int, int, int, float]:
    """Extract frames from video using FFmpeg. Returns (frame_count, width, height, fps)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video info
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0",
        str(video_path)
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    parts = result.stdout.strip().split(",")
    width, height = int(parts[0]), int(parts[1])
    fps_parts = parts[2].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 25.0
    
    # Extract frames
    frame_pattern = str(output_dir / "frame_%06d.png")
    cmd = [
        "ffmpeg", "-y",
        "-err_detect", "ignore_err",
        "-fflags", "+genpts+igndts",
        "-i", str(video_path),
        "-vsync", "0",  # Preserve original frame timings
        frame_pattern
    ]
    
    LOG.info("Extracting frames from %s...", video_path.name)
    subprocess.run(cmd, check=True, capture_output=True)
    
    frame_count = len(list(output_dir.glob("frame_*.png")))
    LOG.info("Extracted %d frames (%.2f fps, %dx%d)", frame_count, fps, width, height)
    return frame_count, width, height, fps


def reassemble_video(frame_dir: Path, output_path: Path, fps: float, width: int, height: int) -> None:
    """Reassemble frames into video using FFmpeg."""
    frame_pattern = str(frame_dir / "frame_%06d.png")
    
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.2",
        str(output_path)
    ]
    
    LOG.info("Reassembling video: %s", output_path.name)
    subprocess.run(cmd, check=True, capture_output=True)
    LOG.info("Video saved: %s", output_path)


def process_video_vrt(
    video_path: Path,
    model_key: str,
    output_path: Path,
    sigma: int = 10,
    tile: Optional[list] = None,
    tile_overlap: Optional[list] = None,
    use_gpu: bool = True
) -> None:
    """Process video through VRT model using VRT's main_test_vrt.py."""
    config = VRT_MODELS[model_key]
    task = config["task"]
    
    # Check for Intel GPU support
    device_type = "cpu"
    if use_gpu:
        try:
            import intel_extension_for_pytorch as ipex
            if ipex.xpu.is_available():
                device_type = "xpu"
                device_count = ipex.xpu.device_count()
                LOG.info("Intel GPU (XPU) available: %d device(s)", device_count)
                for i in range(device_count):
                    LOG.info("  Device %d: %s", i, ipex.xpu.get_device_name(i))
            else:
                LOG.warning("Intel GPU (XPU) not available, falling back to CPU")
        except ImportError:
            LOG.warning("Intel Extension for PyTorch not installed. Install with: pip install intel-extension-for-pytorch")
            LOG.warning("Falling back to CPU (this will be slow)")
        except Exception as e:
            LOG.warning("Error checking Intel GPU: %s. Falling back to CPU", e)
    
    # Use provided tile settings or defaults
    if tile is None:
        tile = config.get("tile", [40, 128, 128])
    if tile_overlap is None:
        tile_overlap = config.get("tile_overlap", [2, 20, 20])
    
    # Create temp directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        frames_lq_dir = temp_path / "frames_lq"
        frames_restored_dir = temp_path / "frames_restored"
        frames_restored_dir.mkdir()
        
        # Extract frames
        frame_count, width, height, fps = extract_frames(video_path, frames_lq_dir)
        
        # VRT expects a subdirectory structure: folder/subfolder/frames.png
        # Create a subdirectory with the video name
        video_subdir = frames_lq_dir / video_path.stem
        video_subdir.mkdir(exist_ok=True)
        
        # Move frames into the subdirectory
        for frame_file in sorted(frames_lq_dir.glob("frame_*.png")):
            frame_file.rename(video_subdir / frame_file.name)
        
        # Prepare VRT command
        vrt_script = VRT_PATH / "main_test_vrt.py"
        if not vrt_script.exists():
            raise RuntimeError(f"VRT main script not found: {vrt_script}")
        
        results_dir = temp_path / "results" / task
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment for Intel GPU and CPU optimization
        env = os.environ.copy()
        
        # Set Intel GPU environment variables (for oneAPI/Level Zero)
        # Use both GPUs: "0,1" or just first: "0"
        env["ZE_AFFINITY_MASK"] = "0,1"  # Use both Intel Arc A770 GPUs
        env["SYCL_CACHE_PERSISTENT"] = "1"
        env["SYCL_CACHE_DIR"] = str(Path.home() / ".cache" / "intel_gpu_cache")
        
        # Optimize CPU usage
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        env["OMP_NUM_THREADS"] = str(cpu_count)
        env["MKL_NUM_THREADS"] = str(cpu_count)
        env["NUMEXPR_NUM_THREADS"] = str(cpu_count)
        
        LOG.info("Environment configured for Intel GPU acceleration")
        LOG.info("  ZE_AFFINITY_MASK: %s (using both GPUs)", env["ZE_AFFINITY_MASK"])
        LOG.info("  CPU threads: %d", cpu_count)
        
        # Log model loading information
        model_path = MODEL_CACHE / config["model_file"]
        if model_path.exists():
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            LOG.info("Model: %s (%.2f MB)", config["model_file"], model_size_mb)
        else:
            LOG.info("Model: %s (will download)", config["model_file"])
        
        cmd = [
            sys.executable,
            str(vrt_script),
            "--task", task,
            "--folder_lq", str(frames_lq_dir),  # Parent directory containing subdirectories
            "--tile"] + [str(t) for t in tile] + [
            "--tile_overlap"] + [str(t) for t in tile_overlap] + [
            "--save_result"
        ]
        
        if 'denoise' in model_key:
            cmd.extend(["--sigma", str(sigma)])
        
        LOG.info("Running VRT processing on %s...", device_type.upper())
        LOG.info("Processing %d frames...", frame_count)
        LOG.debug("Command: %s", " ".join(cmd))
        
        # Run VRT with environment and real-time progress monitoring
        LOG.info("="*60)
        LOG.info("Starting VRT inference...")
        LOG.info("="*60)
        
        process = subprocess.Popen(
            cmd,
            cwd=str(VRT_PATH),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor progress in real-time
        last_progress = 0
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            
            # Parse progress from VRT output
            # VRT outputs: "Testing {:20s} ({:2d}/{})"
            if "Testing" in line and "/" in line:
                try:
                    # Extract progress: "Testing video_name (5/10)"
                    import re
                    match = re.search(r'\((\d+)/(\d+)\)', line)
                    if match:
                        current = int(match.group(1))
                        total = int(match.group(2))
                        progress_pct = (current / total) * 100
                        if progress_pct - last_progress >= 5:  # Log every 5%
                            LOG.info("Progress: %d/%d frames (%.1f%%)", current, total, progress_pct)
                            last_progress = progress_pct
                except:
                    pass
            
            # Log important messages
            if any(keyword in line.lower() for keyword in ["error", "warning", "using", "device", "gpu", "xpu", "model"]):
                if "error" in line.lower():
                    LOG.error("VRT: %s", line)
                elif "warning" in line.lower():
                    LOG.warning("VRT: %s", line)
                else:
                    LOG.info("VRT: %s", line)
        
        process.wait()
        result = subprocess.CompletedProcess(
            process.args,
            process.returncode,
            "",  # stdout already consumed
            ""   # stderr merged with stdout
        )
        
        if result.returncode != 0:
            LOG.error("="*60)
            LOG.error("VRT processing failed with exit code: %d", result.returncode)
            LOG.error("="*60)
            # Try to get stderr if available
            if hasattr(result, 'stderr') and result.stderr:
                LOG.error("STDERR: %s", result.stderr)
            raise RuntimeError(f"VRT processing failed with code {result.returncode}")
        
        LOG.info("="*60)
        LOG.info("✓ VRT inference complete")
        LOG.info("="*60)
        
        # Find output frames (VRT saves to results/task/folder_name/)
        output_folders = list(results_dir.glob("*"))
        if not output_folders:
            raise RuntimeError(f"No output found in {results_dir}")
        
        # Get frames from the output folder
        output_folder = output_folders[0]
        restored_frames = sorted(output_folder.glob("*.png"))
        
        if not restored_frames:
            raise RuntimeError(f"No restored frames found in {output_folder}")
        
        LOG.info("Found %d restored frames", len(restored_frames))
        
        # Copy frames to our output directory with proper naming
        for idx, frame_path in enumerate(restored_frames, 1):
            dest = frames_restored_dir / f"frame_{idx:06d}.png"
            shutil.copy2(frame_path, dest)
        
        # Get output dimensions from first restored frame
        first_frame = cv2.imread(str(frames_restored_dir / "frame_000001.png"))
        out_height, out_width = first_frame.shape[:2]
        
        # Reassemble video
        reassemble_video(frames_restored_dir, output_path, fps, out_width, out_height)


def create_comparison(original: Path, enhanced: Path, output: Path) -> None:
    """Create side-by-side comparison video."""
    LOG.info("Creating side-by-side comparison...")
    
    # Get video info
    probe_orig = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(original)],
        capture_output=True, text=True
    )
    orig_w, orig_h = map(int, probe_orig.stdout.strip().split(","))
    
    probe_enh = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(enhanced)],
        capture_output=True, text=True
    )
    enh_w, enh_h = map(int, probe_enh.stdout.strip().split(","))
    
    # Scale both to same height for comparison
    comp_height = min(orig_h, enh_h)
    orig_scale_w = int(orig_w * comp_height / orig_h / 2) * 2  # Make even
    enh_scale_w = int(enh_w * comp_height / enh_h / 2) * 2
    total_width = orig_scale_w + enh_scale_w
    
    # Create side-by-side with labels
    vf = (
        f"scale={orig_scale_w}:{comp_height}[orig];"
        f"scale={enh_scale_w}:{comp_height}[enh];"
        f"[orig]pad=iw+{enh_scale_w}:ih:0:0:black[left];"
        f"[left][enh]overlay={orig_scale_w}:0[out];"
        f"[out]drawtext=text='Original':fontsize=24:fontcolor=white:x=10:y=10:box=1:boxcolor=black@0.5,"
        f"drawtext=text='VRT Enhanced':fontsize=24:fontcolor=white:x={orig_scale_w+10}:y=10:box=1:boxcolor=black@0.5"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(original),
        "-i", str(enhanced),
        "-filter_complex", vf,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "copy",
        "-shortest",
        str(output)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    LOG.info("Comparison saved: %s", output)


def main():
    parser = argparse.ArgumentParser(description="VRT Video Enhancement")
    parser.add_argument("--task", type=str, required=True,
                       choices=list(VRT_MODELS.keys()),
                       help="VRT task to perform")
    parser.add_argument("--input", type=str, required=True,
                       help="Input video path")
    parser.add_argument("--output", type=str, required=True,
                       help="Output video path")
    parser.add_argument("--comparison", type=str, default=None,
                       help="Optional: Create side-by-side comparison video")
    parser.add_argument("--sigma", type=int, default=10,
                       help="Noise level for denoising (10, 20, 30, 40, 50)")
    parser.add_argument("--tile", type=int, nargs=3, default=None,
                       help="Tile size [t h w] (default: model default)")
    parser.add_argument("--tile-overlap", type=int, nargs=3, default=None,
                       help="Tile overlap [t h w] (default: model default)")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                       help="Verbosity level")
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.WARNING
    if args.verbose >= 1:
        level = logging.INFO
    if args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        LOG.error("Input video not found: %s", input_path)
        return 1
    
    LOG.info("Processing: %s -> %s", input_path.name, output_path.name)
    LOG.info("Task: %s - %s", args.task, VRT_MODELS[args.task]["description"])
    
    try:
        process_video_vrt(
            input_path,
            args.task,
            output_path,
            sigma=args.sigma,
            tile=args.tile,
            tile_overlap=args.tile_overlap
        )
        
        if args.comparison:
            create_comparison(input_path, output_path, Path(args.comparison))
        
        LOG.info("Processing complete!")
        return 0
    except Exception as e:
        LOG.error("Processing failed: %s", e, exc_info=args.verbose >= 2)
        return 1


if __name__ == "__main__":
    sys.exit(main())


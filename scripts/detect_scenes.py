"""
Batch scene detection for family videos.

Prereqs:
  pip install scenedetect[opencv] openai-whisper
Usage:
  python detect_scenes.py --input <folder> --output <folder>
"""
from pathlib import Path
import argparse
import logging
import os
import subprocess
import sys


DEFAULT_INPUT = Path(os.environ.get("SUPEREIGHT_INPUT", "/data/inputs")).resolve()
DEFAULT_OUTPUT = Path(os.environ.get("SUPEREIGHT_OUTPUT", "/data/processed")).resolve()
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
LOG = logging.getLogger("detect_scenes")


def ensure_dirs(output_base: Path) -> tuple[Path, Path]:
    scenes_folder = output_base / "scenes"
    subs_folder = output_base / "subtitles"
    output_base.mkdir(parents=True, exist_ok=True)
    scenes_folder.mkdir(parents=True, exist_ok=True)
    subs_folder.mkdir(parents=True, exist_ok=True)
    return scenes_folder, subs_folder


def find_videos(input_folder: Path):
    return sorted(
        p for p in input_folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def detect(video_path: Path, scenes_folder: Path, suppress_errors: bool = True) -> int:
    output_dir = scenes_folder / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "scenedetect",
        "--input", str(video_path),
        "--output", str(output_dir),
        "detect-adaptive",
        "split-video",
    ]
    
    # Add ffmpeg error suppression for corrupted video files
    if suppress_errors:
        # Set environment variable to suppress ffmpeg decode errors
        env = os.environ.copy()
        env["FFMPEG_ERROR_LEVEL"] = "error"  # Only show errors, not warnings
        # Pass ffmpeg args through scenedetect's ffmpeg wrapper
        # Note: scenedetect doesn't directly expose ffmpeg args, but we can suppress stderr
        LOG.info("[scene-detect] %s", video_path.name)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        stdout, stderr = proc.communicate()
        
        # Filter out common decode warnings but keep actual errors
        if stderr:
            filtered_stderr = []
            decode_warnings = []
            decode_warning_patterns = [
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
                is_decode_warning = any(skip in line_lower for skip in decode_warning_patterns)
                
                if is_decode_warning:
                    decode_warnings.append(line)
                    # Log at DEBUG level for debugging purposes
                    LOG.debug("[FFmpeg decode warning] %s: %s", video_path.name, line.strip())
                else:
                    filtered_stderr.append(line)
            
            # Log summary if there were many decode warnings
            if decode_warnings and len(decode_warnings) > 5:
                LOG.debug("[FFmpeg] %s: Filtered %d decode warnings (corrupted stream, processing continues)", 
                         video_path.name, len(decode_warnings))
            
            if filtered_stderr:
                # Only log if there are real errors (not decode-related)
                # Filter out any remaining decode/decoder errors
                error_lines = [
                    l for l in filtered_stderr 
                    if "error" in l.lower() 
                    and "decoder" not in l.lower() 
                    and "decoding" not in l.lower()
                    and "invalid data" not in l.lower()
                ]
                if error_lines:
                    LOG.warning("FFmpeg errors for %s: %s", video_path.name, "\n".join(error_lines[:5]))
                    # Also log full context at DEBUG level for debugging
                    LOG.debug("[FFmpeg full stderr] %s: %s", video_path.name, "\n".join(filtered_stderr[:20]))
        
        return proc.returncode
    else:
        LOG.info("[scene-detect] %s", video_path.name)
        return subprocess.call(cmd)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Batch scene detection")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Folder with source videos")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Base output folder")
    parser.add_argument("--dry-run", action="store_true", help="List planned videos and exit")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase log verbosity (use -vv for debug)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to log file (logs all levels to file for debugging, console respects verbosity)",
    )
    return parser.parse_args(argv)


def configure_logging(verbosity: int, log_file: str = None) -> None:
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
    console_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)  # Root logger accepts all, handlers filter
    
    # File handler (logs all levels if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        LOG.info("Logging to file: %s", log_file)


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose, log_file=getattr(args, 'log_file', None))

    input_folder = Path(args.input)
    output_base = Path(args.output)
    scenes_folder, _ = ensure_dirs(output_base)

    videos = find_videos(input_folder)
    if not videos:
        LOG.warning("No videos found in %s", input_folder)
        return 1

    LOG.info("Found %d videos", len(videos))
    if args.dry_run:
        for v in videos:
            LOG.info("Would process: %s", v.name)
        return 0

    failures = 0
    for video in videos:
        rc = detect(video, scenes_folder)
        if rc != 0:
            LOG.error("!! scenedetect failed for %s (rc=%s)", video.name, rc)
            failures += 1

    LOG.info("Scene detection finished. Videos: %d, failures: %d", len(videos), failures)
    return failures


if __name__ == "__main__":
    sys.exit(main())


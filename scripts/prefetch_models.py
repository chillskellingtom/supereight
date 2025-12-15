#!/usr/bin/env python
"""
Prefetch core ONNX models into /models (or SUPEREIGHT_MODEL_CACHE) for offline runs.
"""
from pathlib import Path
import hashlib
import urllib.request
import sys

FACE_DETECT_MODEL_NAME = "scrfd_2.5g_bnkps.onnx"
FACE_DETECT_MODEL_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_scrfd/scrfd_2.5g_bnkps.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_scrfd/scrfd_2.5g_bnkps.onnx",
    "https://huggingface.co/OwlMaster/AllFilesRope/resolve/d783e61585b3d83a85c91ca8a3b299e8ade94d72/scrfd_2.5g_bnkps.onnx?download=true",
]
FACE_RECOG_MODEL_NAME = "arcface_r100.onnx"
FACE_RECOG_MODEL_URLS = [
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcface_r100.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx?download=true",
]

MODEL_CHECKSUMS = {
    FACE_RECOG_MODEL_NAME: "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    FACE_DETECT_MODEL_NAME: "bc24bb349491481c3ca793cf89306723162c280cb284c5a5e49df3760bf5c2ce",
    "face_recognition_sface_2021dec.onnx": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
}


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_first(name: str, urls, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            print(f"Downloading {name} from {url}...")
            urllib.request.urlretrieve(url, target)
            if name in MODEL_CHECKSUMS:
                digest = sha256sum(target)
                if digest != MODEL_CHECKSUMS[name]:
                    print(f"Checksum mismatch for {name}: {digest} != {MODEL_CHECKSUMS[name]}")
                    continue
            return True
        except Exception as exc:
            print(f"Failed {url}: {exc}")
    return False


def main() -> int:
    model_root = Path(
        Path().anchor if False else Path("/models")
    )  # container default
    model_root = Path(
        Path("/models")
    )
    ok1 = download_first(FACE_DETECT_MODEL_NAME, FACE_DETECT_MODEL_URLS, model_root / FACE_DETECT_MODEL_NAME)
    ok2 = download_first(FACE_RECOG_MODEL_NAME, FACE_RECOG_MODEL_URLS, model_root / FACE_RECOG_MODEL_NAME)
    if not ok1 or not ok2:
        return 1
    print("Prefetch complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

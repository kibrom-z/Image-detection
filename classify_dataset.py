"""
Classify images as 'face' vs 'no_face' based on OpenCV Haar face detection,
then MOVE files into dataset folders.

Rule (per user request):
- face: at least 1 detected face with size >= 80x80 (w>=80 and h>=80)
- no_face: otherwise

Default input:  data/image/
Default output: data/dataset/{face,no_face}/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Tuple

import cv2

import sys
from pathlib import Path as _Path

# allow importing from src/utils when running from repo root
sys.path.insert(0, str(_Path(__file__).parent / "src"))
from utils.face_backends import detect_faces_haar, detect_faces_dnn  # type: ignore


def iter_images(input_dir: Path, recursive: bool) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if recursive:
        it = input_dir.rglob("*")
    else:
        it = input_dir.glob("*")
    for p in it:
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def has_face_of_min_size(
    image_path: Path,
    method: str,
    min_face_size: Tuple[int, int],
    scale_factor: float,
    min_neighbors: int,
    conf_threshold: float,
    model_dir: Path,
) -> Tuple[bool, int]:
    """
    Returns (is_face, qualifying_face_count).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    if method == "haar":
        boxes = detect_faces_haar(
            img,
            min_size=min_face_size,
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
        )
    elif method == "dnn":
        boxes = detect_faces_dnn(
            img,
            min_size=min_face_size,
            conf_threshold=conf_threshold,
            model_dir=model_dir,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    qualifying = len(boxes)

    return (qualifying >= 1), qualifying


def safe_move(src: Path, dest_dir: Path) -> Path:
    """
    Move src file into dest_dir, avoiding name collisions by appending _N.
    Returns the final destination path.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if not dest.exists():
        shutil.move(str(src), str(dest))
        return dest

    stem, suffix = src.stem, src.suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            shutil.move(str(src), str(candidate))
            return candidate
        i += 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move images into face/no_face folders based on face detection."
    )
    parser.add_argument(
        "--input-dir",
        default="data/image",
        help="Directory containing images to classify (default: data/image)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/dataset",
        help="Output dataset directory (default: data/dataset)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input directory recursively",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        nargs=2,
        default=[80, 80],
        metavar=("WIDTH", "HEIGHT"),
        help="Minimum face size (default: 80 80)",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Haar detectMultiScale scaleFactor (default: 1.1)",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Haar detectMultiScale minNeighbors (default: 5)",
    )
    parser.add_argument(
        "--method",
        choices=["haar", "dnn"],
        default="dnn",
        help="Detector backend (default: dnn). Use haar for faster but less accurate.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="DNN confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--model-dir",
        default="data/models/opencv_face_detector",
        help="Where to store/load DNN model files (default: data/models/opencv_face_detector)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without moving files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: input directory does not exist: {input_dir}")
        return 2

    output_dir = Path(args.output_dir)
    face_dir = output_dir / "face"
    no_face_dir = output_dir / "no_face"

    min_face_size = (int(args.min_face_size[0]), int(args.min_face_size[1]))
    model_dir = Path(args.model_dir)

    images = list(iter_images(input_dir, args.recursive))
    if not images:
        print(f"No images found in: {input_dir}")
        return 0

    moved_face = 0
    moved_no_face = 0
    errors = 0

    print(f"Classifying {len(images)} image(s) from {input_dir} ...")
    print(f"Rule: face if >=1 face with size >= {min_face_size[0]}x{min_face_size[1]}")
    print(f"Method: {args.method}")
    if args.dry_run:
        print("[DRY RUN] No files will be moved.")

    for img_path in images:
        try:
            is_face, count = has_face_of_min_size(
                img_path,
                method=args.method,
                min_face_size=min_face_size,
                scale_factor=args.scale_factor,
                min_neighbors=args.min_neighbors,
                conf_threshold=args.conf_threshold,
                model_dir=model_dir,
            )
            dest = face_dir if is_face else no_face_dir
            label = "face" if is_face else "no_face"

            if args.dry_run:
                print(f"[{label}] {img_path.name} (qualifying_faces={count}) -> {dest}")
            else:
                final_path = safe_move(img_path, dest)
                print(f"[{label}] {img_path.name} (qualifying_faces={count}) -> {final_path}")

            if is_face:
                moved_face += 1
            else:
                moved_no_face += 1
        except Exception as e:
            errors += 1
            print(f"[error] {img_path}: {e}")

    print(
        f"Done. face={moved_face}, no_face={moved_no_face}, errors={errors}. "
        f"Output: {output_dir}"
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


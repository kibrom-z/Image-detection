"""
Capture images from webcam and optionally classify them into face/no_face.

Controls:
- SPACE: save current frame as JPEG (and classify if enabled)
- Q: quit
"""

import argparse
import datetime
from pathlib import Path
from typing import Tuple

import cv2

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).parent / "src"))
from utils.face_backends import detect_faces_haar, detect_faces_dnn  # type: ignore

def has_face_in_frame(
    frame,
    method: str,
    min_face_size: Tuple[int, int],
    scale_factor: float,
    min_neighbors: int,
    conf_threshold: float,
    model_dir: Path,
) -> bool:
    """Return True if the frame has at least one face meeting the size threshold."""
    if method == "haar":
        boxes = detect_faces_haar(
            frame,
            min_size=min_face_size,
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
        )
    elif method == "dnn":
        boxes = detect_faces_dnn(
            frame,
            min_size=min_face_size,
            conf_threshold=conf_threshold,
            model_dir=model_dir,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return len(boxes) > 0


def save_frame(frame, dest_dir: Path, prefix: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = dest_dir / f"{prefix}_{timestamp}.jpg"
    cv2.imwrite(str(filename), frame)
    return filename


def main():
    parser = argparse.ArgumentParser(
        description="Capture webcam frames; optionally classify into face/no_face."
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/image",
        help="Directory for uncategorized captures (default: data/image)",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="If set, classify captures into face/no_face folders under data/dataset/",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/dataset",
        help="Base dataset dir when classify is enabled (default: data/dataset)",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        nargs=2,
        default=[80, 80],
        metavar=("WIDTH", "HEIGHT"),
        help="Minimum face size to qualify as 'face' (default: 80 80)",
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.classify:
        dataset_dir = Path(args.dataset_dir)
        face_dir = dataset_dir / "face"
        no_face_dir = dataset_dir / "no_face"

        min_face_size = (int(args.min_face_size[0]), int(args.min_face_size[1]))
        model_dir = Path(args.model_dir)
    else:
        face_dir = no_face_dir = min_face_size = model_dir = None

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam index {args.camera_index}. Try a different index.")
        return

    print("Webcam started. Press SPACE to save a frame, Q to quit.")
    if args.classify:
        print(
            f"Classification ON -> face if >=1 face size >= {args.min_face_size[0]}x{args.min_face_size[1]}."
        )
        print(f"Method: {args.method}")

    saved_raw = 0
    saved_face = 0
    saved_no_face = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read from webcam.")
                break

            cv2.imshow("Webcam Capture (SPACE = save, Q = quit)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):  # Spacebar to save
                if args.classify:
                    is_face = has_face_in_frame(
                        frame,
                        method=args.method,
                        min_face_size=min_face_size,
                        scale_factor=args.scale_factor,
                        min_neighbors=args.min_neighbors,
                        conf_threshold=args.conf_threshold,
                        model_dir=model_dir,
                    )
                    target_dir = face_dir if is_face else no_face_dir
                    saved_path = save_frame(frame, target_dir, prefix="webcam")
                    if is_face:
                        saved_face += 1
                    else:
                        saved_no_face += 1
                    print(f"[OK] Saved to {target_dir.name}: {saved_path.name}")
                else:
                    saved_path = save_frame(frame, output_dir, prefix="webcam")
                    saved_raw += 1
                    print(f"[OK] Saved: {saved_path.name}")

            elif key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if args.classify:
        print(
            f"Done. Saved face={saved_face}, no_face={saved_no_face} into {Path(args.dataset_dir)}."
        )
    else:
        print(f"Done. Saved {saved_raw} image(s) to {output_dir}.")


if __name__ == "__main__":
    main()

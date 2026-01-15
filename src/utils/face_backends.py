"""
Face detector backends.

- Haar cascade (fast, less accurate)
- OpenCV DNN ResNet-10 SSD (more accurate)

The DNN model is downloaded on-demand into `data/models/opencv_face_detector/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import urllib.request

import cv2
import numpy as np


Box = Tuple[int, int, int, int]  # x, y, w, h


@dataclass(frozen=True)
class DnnModelPaths:
    prototxt: Path
    caffemodel: Path


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # urllib is in stdlib (works without extra deps)
    urllib.request.urlretrieve(url, str(dest))


def ensure_opencv_dnn_face_model(model_dir: Path = Path("data/models/opencv_face_detector")) -> DnnModelPaths:
    """
    Ensures the OpenCV DNN face model files exist locally, downloading them if missing.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    prototxt = model_dir / "deploy.prototxt"
    caffemodel = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

    # Known public sources (may change over time). If download fails, user can place files manually.
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn_models/res10_300x300_ssd_iter_140000.caffemodel"

    if not prototxt.exists():
        _download(prototxt_url, prototxt)
    if not caffemodel.exists():
        _download(caffemodel_url, caffemodel)

    return DnnModelPaths(prototxt=prototxt, caffemodel=caffemodel)


def detect_faces_haar(
    bgr_image: np.ndarray,
    min_size: Tuple[int, int] = (80, 80),
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
) -> List[Box]:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def detect_faces_dnn(
    bgr_image: np.ndarray,
    min_size: Tuple[int, int] = (80, 80),
    conf_threshold: float = 0.5,
    model_dir: Path = Path("data/models/opencv_face_detector"),
) -> List[Box]:
    """
    Returns bounding boxes from OpenCV DNN SSD face detector.
    """
    paths = ensure_opencv_dnn_face_model(model_dir=model_dir)

    net = cv2.dnn.readNetFromCaffe(str(paths.prototxt), str(paths.caffemodel))
    (h, w) = bgr_image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(bgr_image, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()

    boxes: List[Box] = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < conf_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        bw = x2 - x1
        bh = y2 - y1
        if bw >= min_size[0] and bh >= min_size[1]:
            boxes.append((int(x1), int(y1), int(bw), int(bh)))

    return boxes


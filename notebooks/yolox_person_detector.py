"""YOLOX-s (ONNX, Apache-2.0) used as a person detector. Notebook 07 explains each step."""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

_session = ort.InferenceSession(str(Path(__file__).parent.parent / "models/yolox_s.onnx"))

# YOLOX grids: cell (gx, gy) and stride s for the 8,400 predictions (80² + 40² + 20²)
_grids, _strides = [], []
for s in (8, 16, 32):
    n = 640 // s
    gy, gx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    _grids.append(np.stack([gx, gy], -1).reshape(-1, 2)); _strides.append(np.full((n * n, 1), s))
GRID, STRIDE = np.concatenate(_grids), np.concatenate(_strides)
assert len(GRID) == 8400


def area(b):
    return (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])


def intersection(a, b):
    """Intersection area between each box in a (n×4) and each box in b (m×4) → n×m."""
    a, b = a[:, None], b[None]
    return (np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None) *
            np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None))


def nms(boxes, scores, max_iou):
    """Non-maximum suppression: keep the highest score, drop boxes that overlap it too much, repeat."""
    order, kept = np.argsort(-scores), []
    while len(order):
        i, order = order[0], order[1:]
        kept.append(i)
        inter = intersection(boxes[i:i + 1], boxes[order])[0]
        order = order[inter / (area(boxes[i]) + area(boxes[order]) - inter) <= max_iou]
    return kept


def detect_persons(img, threshold=0.3, max_iou=0.45):
    """Person boxes (x1, y1, x2, y2) in pixels of the original photo, and their person score."""
    r = min(640 / img.width, 640 / img.height)
    canvas = np.full((640, 640, 3), 114, np.uint8)
    scaled = np.asarray(img.resize((round(img.width * r), round(img.height * r)), Image.BILINEAR))
    canvas[:scaled.shape[0], :scaled.shape[1]] = scaled[..., ::-1]               # RGB → BGR
    out = _session.run(None, {"images": canvas.transpose(2, 0, 1)[None].astype("float32")})[0][0]
    center = (out[:, :2] + GRID) * STRIDE
    side = np.exp(out[:, 2:4]) * STRIDE
    score = out[:, 4] * out[:, 5]                                                # objectness × P(person)
    k = score >= threshold
    boxes = np.concatenate([center[k] - side[k] / 2, center[k] + side[k] / 2], 1)
    kept = nms(boxes, score[k], max_iou)
    return boxes[kept] / r, score[k][kept]

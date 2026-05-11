import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGES_PER_CLASS = 500

SUBSET_CLASSES = [
    "pizza",
    "sushi",
    "ramen",
    "hamburger",
    "steak",
    "ice_cream",
    "waffles",
    "tacos",
    "fried_rice",
    "caesar_salad",
]

ROOT = Path(__file__).parent
FOOD101_DIR = ROOT / "trainingset" / "food-101" / "images"
OUTPUT_DIR = ROOT / "trainingset" / "food-101-cropped" / "images"
ONNX_PATH = ROOT / "onnx" / "yolov8n.onnx"
TRAIN_JSON = ROOT / "trainingset" / "food-101" / "meta" / "train.json"

YOLO_INPUT_SIZE = 640
YOLO_CONF = 0.25
NMS_IOU_THRESHOLD = 0.45

MIN_AREA_FRAC = 0.01
PAD = 10


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def preprocess_yolo(img: Image.Image):
    orig_w, orig_h = img.size
    resized = img.resize((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    blob = arr[np.newaxis, ...]
    scale_x = orig_w / YOLO_INPUT_SIZE
    scale_y = orig_h / YOLO_INPUT_SIZE
    return blob, scale_x, scale_y, orig_w, orig_h


def iou(box_a, box_b):
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_threshold):
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        order = order[1:]
        if len(order) == 0:
            break
        ious = np.array([iou(boxes[i], boxes[j]) for j in order])
        order = order[ious <= iou_threshold]
    return keep


def detect_boxes(yolo_output, scale_x, scale_y, orig_w, orig_h):
    preds = yolo_output[0].T  # [8400, 84]
    boxes_xywh = preds[:, :4]
    confidences = preds[:, 4:].max(axis=1)

    mask = confidences > YOLO_CONF
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]

    if len(boxes_xywh) == 0:
        return []

    cx = boxes_xywh[:, 0]
    cy = boxes_xywh[:, 1]
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]

    x1 = np.clip((cx - w / 2) * scale_x, 0, orig_w)
    y1 = np.clip((cy - h / 2) * scale_y, 0, orig_h)
    x2 = np.clip((cx + w / 2) * scale_x, 0, orig_w)
    y2 = np.clip((cy + h / 2) * scale_y, 0, orig_h)

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    min_area = orig_w * orig_h * MIN_AREA_FRAC
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (
        boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    )
    valid = areas >= min_area
    boxes_xyxy = boxes_xyxy[valid]
    confidences = confidences[valid]

    if len(boxes_xyxy) == 0:
        return []

    keep = nms(boxes_xyxy, confidences, NMS_IOU_THRESHOLD)
    return boxes_xyxy[keep].tolist()


def best_box(boxes, orig_w, orig_h):
    best = None
    best_area = 0
    for x1, y1, x2, y2 in boxes:
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
    return best


def main():
    print("Loading YOLO ONNX session...")
    session = ort.InferenceSession(str(ONNX_PATH))
    input_name = session.get_inputs()[0].name

    with open(TRAIN_JSON) as f:
        train_data = json.load(f)

    total_saved = 0
    total_fallback = 0

    for cls in SUBSET_CLASSES:
        all_images = train_data.get(cls, [])[:IMAGES_PER_CLASS]

        out_dir = OUTPUT_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        fallback = 0

        for img_path_str in all_images:
            img_path = FOOD101_DIR / f"{img_path_str}.jpg"
            if not img_path.exists():
                print(f"  missing: {img_path}")
                continue

            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size

            blob, scale_x, scale_y, orig_w, orig_h = preprocess_yolo(img)
            yolo_out = session.run(None, {input_name: blob})[0]
            boxes = detect_boxes(yolo_out, scale_x, scale_y, orig_w, orig_h)

            if boxes:
                result = best_box(boxes, orig_w, orig_h)

                if result is None:
                    crop = img
                    fallback += 1
                else:
                    x1, y1, x2, y2 = result
                    crop_w = x2 - x1
                    crop_h = y2 - y1
                    if (
                        crop_w < 20
                        or crop_h < 20
                        or max(crop_w, crop_h) / min(crop_w, crop_h) > 5
                    ):
                        crop = img
                        fallback += 1
                    else:
                        x1 = max(0, x1 - PAD)
                        y1 = max(0, y1 - PAD)
                        x2 = min(orig_w, x2 + PAD)
                        y2 = min(orig_h, y2 + PAD)
                        crop = img.crop((x1, y1, x2, y2))
            else:
                crop = img
                fallback += 1

            out_path = out_dir / Path(img_path_str).with_suffix(".jpg").name
            crop.save(out_path, "JPEG", quality=95)
            saved += 1

        print(f"  {cls}: {saved} saved, {fallback} fallback (no detection)")
        total_saved += saved
        total_fallback += fallback
    print(f"\nDone. {total_saved} total cropped saved, {total_fallback} fallbacks.")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

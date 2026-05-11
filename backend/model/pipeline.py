import io
from dataclasses import dataclass
from typing import cast

import numpy as np
import onnxruntime as ort  # type: ignore
import torchvision.transforms as T
from PIL import Image
from torch.fx.experimental.sym_node import sizes_strides_methods


@dataclass
class RegionPrediction:
    bbox: list[float]
    dish: str
    confidence: float
    uncertain: bool
    top5: list[dict]


@dataclass
class PipelineResult:
    regions: list[RegionPrediction]
    multi_food: bool
    no_food_found: bool


CONFIDENCE_THRESHOLD = 0.35
YOLO_INPUT_SIZE = 640
YOLO_CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
FOODCNN_INPUT_SIZE = 128
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TTA_TRANSFORMS = [
    T.Compose(
        [
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    ),
    T.Compose(
        [
            T.Resize((128, 128)),
            T.CenterCrop(128),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    ),
    T.Compose(
        [
            T.Resize((144, 144)),
            T.CenterCrop(128),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    ),
    T.Compose(
        [
            T.Resize((144, 144)),
            T.CenterCrop(128),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    ),
    T.Compose(
        [
            T.Resize((128, 128)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    ),
]


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def preprocess_yolo(img: Image.Image) -> tuple[np.ndarray, float, float, int, int]:
    orig_w, orig_h = img.size
    resized = img.resize((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), Image.Resampling.BILINEAR)

    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    blob = arr[np.newaxis, ...]

    scale_x = orig_w / YOLO_INPUT_SIZE
    scale_y = orig_h / YOLO_INPUT_SIZE

    return blob, scale_x, scale_y, orig_w, orig_h


def decode_yolo_output(
    output: np.ndarray,
    scale_x: float,
    scale_y: float,
    orig_w: int,
    orig_h: int,
) -> list[list[float]]:
    preds = output[0]
    preds = preds.T

    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]

    confidences = class_scores.max(axis=1)

    mask = confidences > YOLO_CONF_THRESHOLD
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]

    if len(boxes_xywh) == 0:
        return []

    cx = boxes_xywh[:, 0]
    cy = boxes_xywh[:, 1]
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]

    x1 = (cx - w / 2) * scale_x
    y1 = (cy - h / 2) * scale_y
    x2 = (cx + w / 2) * scale_x
    y2 = (cy + h / 2) * scale_y

    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(boxes_xyxy, confidences, NMS_IOU_THRESHOLD)

    min_area = (orig_w * orig_h) * 0.01
    boxes_xyxy = boxes_xyxy[keep]
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (
        boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    )
    boxes_xyxy = boxes_xyxy[areas >= min_area]

    return boxes_xyxy.tolist()


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = max(box_a[2], box_b[2])
    iy2 = max(box_a[3], box_b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(int(i))

        if len(order) == 1:
            break

        remaining = order[1:]
        ious = np.array([iou(boxes[i], boxes[j]) for j in remaining])
        order = remaining[ious <= iou_threshold]

    return keep


def classify_crop(
    crop: Image.Image,
    session: ort.InferenceSession,
    input_name: str,
    classes: list[str],
) -> list[dict]:

    all_probs = []

    for transform in TTA_TRANSFORMS:
        tensor = transform(crop).unsqueeze(0).numpy()  # type: ignore
        outputs = session.run(None, {input_name: tensor})[0]
        all_probs.append(softmax(outputs[0]))  # type: ignore

    avg_probs = np.mean(all_probs, axis=0)

    top5_idx = avg_probs.argsort()[::-1][:5]

    return [
        {"dish": classes[i], "confidence": round(float(avg_probs[i]), 4)}
        for i in top5_idx
    ]


def merge_duplicates(regions: list[RegionPrediction]) -> list[RegionPrediction]:
    seen: dict[str, RegionPrediction] = {}
    for region in regions:
        if region.dish not in seen or region.confidence > seen[region.dish].confidence:
            seen[region.dish] = region

    return sorted(seen.values(), key=lambda r: r.confidence, reverse=True)


class FoodPipeline:
    def __init__(
        self,
        yolo_onnx_path: str,
        foodcnn_onnx_path: str,
        classes: list[str],
    ):
        self.yolo_session = ort.InferenceSession(yolo_onnx_path)
        self.yolo_input_name = self.yolo_session.get_inputs()[0].name

        self.foodcnn_session = ort.InferenceSession(foodcnn_onnx_path)
        self.foodcnn_input_name = self.foodcnn_session.get_inputs()[0].name

        self.classes = classes

    def run(self, image_bytes: bytes) -> PipelineResult:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        blob, scale_x, scale_y, orig_w, orig_h = preprocess_yolo(img)

        yolo_output = cast(
            np.ndarray, self.yolo_session.run(None, {self.yolo_input_name: blob})[0]
        )
        boxes = decode_yolo_output(yolo_output, scale_x, scale_y, orig_w, orig_h)
        boxes = boxes[:5]

        if not boxes:
            top5 = classify_crop(
                img, self.foodcnn_session, self.foodcnn_input_name, self.classes
            )
            return PipelineResult(
                regions=[
                    RegionPrediction(
                        bbox=[0, 0, orig_w, orig_h],
                        dish=top5[0]["dish"],
                        confidence=top5[0]["confidence"],
                        uncertain=top5[0]["confidence"] < CONFIDENCE_THRESHOLD,
                        top5=top5,
                    )
                ],
                multi_food=False,
                no_food_found=True,
            )

        regions: list[RegionPrediction] = []

        for x1, y1, x2, y2 in boxes:
            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < 20 or crop_h < 20:
                continue
            if max(crop_w, crop_h) / min(crop_w, crop_h) > 5:
                continue

            PAD = 10
            x1 = max(0, x1 - PAD)
            y1 = max(0, y1 - PAD)
            x2 = min(orig_w, x2 + PAD)
            y2 = min(orig_h, y2 + PAD)

            crop = img.crop((x1, y1, x2, y2))
            top5 = classify_crop(
                crop, self.foodcnn_session, self.foodcnn_input_name, self.classes
            )

            regions.append(
                RegionPrediction(
                    bbox=[round(x1), round(y1), round(x2), round(y2)],
                    dish=top5[0]["dish"],
                    confidence=top5[0]["confidence"],
                    uncertain=top5[0]["confidence"] < CONFIDENCE_THRESHOLD,
                    top5=top5,
                )
            )

        regions = merge_duplicates(regions)
        return PipelineResult(
            regions=regions,
            multi_food=len(regions) > 1,
            no_food_found=False,
        )

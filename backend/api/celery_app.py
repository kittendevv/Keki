import io
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "model"))

from celery import Celery  # type: ignore
from pipeline import FoodPipeline  # type: ignore

celery_app = Celery(
    "keki",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

celery_app.conf.update(
    result_expires=300,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_soft_time_limit=600,
    timezone="UTC",
)

_ONNX_DIR = Path(__file__).parent.parent / "model" / "onnx"
_CLASSES_PATH = (
    Path(__file__).parent.parent
    / "model"
    / "trainingset"
    / "food-101"
    / "meta"
    / "train.json"
)

with open(_CLASSES_PATH) as f:
    _CLASSES = sorted(json.load(f).keys())

_pipeline = FoodPipeline(
    yolo_onnx_path=str(_ONNX_DIR / "yolov8n.onnx"),
    foodcnn_onnx_path=str(_ONNX_DIR / "foodcnn.onnx"),
    classes=_CLASSES,
)


@celery_app.task(bind=True)
def classify_task(self, image_bytes: bytes):
    self.update_state(state="STARTED")

    result = _pipeline.run(image_bytes)

    return {
        "regions": [
            {
                "bbox": r.bbox,
                "dish": r.dish,
                "confidence": r.confidence,
                "uncertain": r.uncertain,
                "top5": r.top5,
            }
            for r in result.regions
        ],
        "multi_food": result.multi_food,
        "no_food_found": result.no_food_found,
    }

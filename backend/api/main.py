import io
import sqlite3
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).parent.parent / "model"))
from dataset import FoodDataset  # type: ignore
from fastapi import FastAPI, File, UploadFile  # type: ignore
from model import FoodCNN  # type: ignore
from PIL import Image

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore

model = FoodCNN(num_classes=101).to(device)
# model.load_state_dict(torch.load("foodcnn.pth", map_location=device))
# model.eval()

import json

classes = []
# with open("../model/trainingset/food-101/meta/train.json") as f:
#    train_data = json.load(f)
# classes = sorted(train_data.keys())

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)  # type: ignore

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)  # type: ignore
        confidence, predicted = torch.max(probs, dim=1)  # type: ignore

        dish = classes[int(predicted.item())]

    return {"dish": dish, "confidence": confidence.item()}


@app.get("/recipe/{dish}")
async def get_recipe(dish: str):
    conn = sqlite3.connect("../db/recipes.db")
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT id, name, ingredients, instructions, source FROM recipes WHERE dish = ?",
        (dish,),
    ).fetchall()
    conn.close()

    if not rows:
        return {"error": "Recipe not found"}

    return {
        "dish": dish,
        "recipe": [
            {
                "id": row["id"],
                "name": row["name"],
                "ingredients": json.loads(row["ingredients"]),
                "instructions": row["instructions"],
                "source": row["source"],
            }
            for row in rows
        ],
    }

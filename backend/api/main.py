import io
import json
import sqlite3
import sys
from pathlib import Path

import httpx
import torch
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).parent.parent / "model"))
from fastapi import FastAPI, File, HTTPException, UploadFile  # type: ignore
from model import FoodCNN  # type: ignore
from PIL import Image

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore

model = FoodCNN(num_classes=101).to(device)
# model.load_state_dict(torch.load("foodcnn.pth", map_location=device))
# model.eval()

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
    top5_confidences, top5_indices = torch.topk(probs, 5, dim=1)  # type: ignore

    top1_confidence = top5_confidences[0][0].item()  # type: ignore

    if top1_confidence < 0.4:
        return {"uncertain": True, "confidence": round(top1_confidence, 4)}

    top5 = [
        {
            "dish": classes[top5_indices[0][i].item()],  # type: ignore
            "confidence": round(top5_confidences[0][i].item(), 4),  # type: ignore
        }
        for i in range(5)
    ]

    return {
        "uncertain": False,
        "dish": top5[0]["dish"],
        "confidence": top5[0]["confidence"],
        "top5": top5,
    }


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


@app.get("/nutrition/{dish}")
async def get_nutrition(dish: str):
    query = dish.replace("_", " ")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://world.openfoodfacts.org/cgi/search.pl",
            params={
                "search_terms": query,
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": 1,
            },
        )

    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Nutrition data not found")

    data = response.json()
    if not data["products"]:
        raise HTTPException(status_code=404, detail="Nutrition data not found")

    product = data["products"][0]
    nutrients = product.get("nutriments", {})

    return {
        "dish": dish,
        "calories": nutrients.get("energy-kcal_100g"),
        "carbs_g": nutrients.get("carbohydrates_100g"),
        "protein_g": nutrients.get("proteins_100g"),
        "fat_g": nutrients.get("fat_100g"),
        "sugar_g": nutrients.get("sugars_100g"),
        "fiber_g": nutrients.get("fiber_100g"),
        "per": "100g",
    }

import io
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).parent.parent / "model"))
from dataset import FoodDataset
from fastapi import FastAPI, File, UploadFile
from model import FoodCNN
from PIL import Image

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

        dish = classes[int(predicted.item())]

    return {"dish": dish, "confidence": confidence.item()}


@app.get("/recipe/{dish}")
async def get_recipe(dish: str):

    return {"dish": dish, "recipe": "coming soon!"}

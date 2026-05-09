import io
import json
import logging
import random
import sqlite3
import sys
import time
import uuid
from difflib import get_close_matches
from pathlib import Path

import httpx  # type: ignore
import torch
import torchvision.transforms as transforms
from auth import get_db, require_admin, require_user  # type: ignore

sys.path.append(str(Path(__file__).parent.parent / "model"))
from fastapi import (  # type: ignore
    APIRouter,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from model import FoodCNN  # type: ignore
from PIL import Image
from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore
from slowapi.errors import RateLimitExceeded  # type: ignore
from slowapi.util import get_remote_address  # type: ignore

_FUN_FACTS_PATH = Path(__file__).parent / "fun_facts.json"
FUN_FACTS: dict[str, list[str]] = (
    json.loads(_FUN_FACTS_PATH.read_text()) if _FUN_FACTS_PATH.exists() else {}
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("keki")

app = FastAPI(title="Keki API")

v1 = APIRouter(prefix="/v1")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s %d %.1fms %s",
        request.method,
        request.url.path,
        response.status_code,
        duration,
        request.client.host if request.client else "-",
    )
    return response


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

PORTION_SIZES = {
    "pizza": 200,
    "hamburger": 250,
    "sushi": 150,
    "tacos": 170,
    "ramen": 500,
    "pad_thai": 300,
    "paella": 350,
    "steak": 250,
    "caesar_salad": 200,
    "french_fries": 150,
    "pancakes": 200,
    "waffles": 200,
    "ice_cream": 100,
    "cheesecake": 120,
    "chocolate_cake": 100,
    "tiramisu": 150,
    "macarons": 50,
}

DEFAULT_PORTION = 250


@v1.post("/classify")
@limiter.limit("10/minute")
async def classify(request: Request, file: UploadFile = File(...)):
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


# Use /recipe/{dish}?exclude=ingredient1,ingredient2 to exclude certain ingredients from results
@v1.get("/recipe/mock")
@limiter.limit("60/minute")
async def mock_recipe(request: Request):
    conn = sqlite3.connect("../db/recipes.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, name, ingredients, instructions, source, servings FROM recipes WHERE dish = 'pizza'",
    ).fetchall()
    conn.close()
    return {
        "dish": "pizza",
        "recipes": [
            {
                "id": row["id"],
                "name": row["name"],
                "ingredients": json.loads(row["ingredients"]),
                "instructions": row["instructions"],
                "source": row["source"],
                "servings": row["servings"],
            }
            for row in rows
        ],
    }


@v1.get("/search")
@limiter.limit("30/minute")
async def search(request: Request, q: str = Query(..., min_length=1, max_length=100)):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    term = f"%{q.strip().lower()}%"

    conn = sqlite3.connect("../db/recipes.db")
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            """
            SELECT DISTINCT dish, name, ingredients, servings
            FROM recipes
            WHERE LOWER(dish) LIKE ?
               OR LOWER(name) LIKE ?
               OR LOWER(ingredients) LIKE ?
            ORDER BY
                CASE WHEN LOWER(dish) LIKE ? THEN 0
                     WHEN LOWER(name) LIKE ? THEN 1
                     ELSE 2 END,
                dish
            LIMIT 20
            """,
            (term, term, term, term, term),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    results = [
        {
            "dish": row["dish"],
            "name": row["name"],
            "ingredients": json.loads(row["ingredients"]),
            "servings": row["servings"],
        }
        for row in rows
    ]
    return {"query": q, "results": results, "count": len(results)}


@v1.get("/recipe/{dish}")
@limiter.limit("60/minute")
async def get_recipe(request: Request, dish: str, exclude: str = None):  # type: ignore
    conn = sqlite3.connect("../db/recipes.db")
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT id, name, ingredients, instructions, source, servings FROM recipes WHERE dish = ?",
        (dish,),
    ).fetchall()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="Recipe not found")

    exclude_list = [e.strip().lower() for e in exclude.split(",")] if exclude else []
    recipes = []
    for row in rows:
        ingredients = json.loads(row["ingredients"])

        if exclude_list:
            ingredient_text = " ".join(ingredients).lower()
            if any(e in ingredient_text for e in exclude_list):
                continue

        recipes.append(
            {
                "id": row["id"],
                "name": row["name"],
                "ingredients": ingredients,
                "instructions": row["instructions"],
                "source": row["source"],
                "servings": row["servings"],
            }
        )

    if not recipes:
        raise HTTPException(
            status_code=404, detail="No recipes found without excluded ingredients"
        )
    return {"dish": dish, "recipes": recipes}


@v1.get("/nutrition/{dish}")
@limiter.limit("60/minute")
async def get_nutrition(request: Request, dish: str):
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


@v1.get("/history")
@limiter.limit("100/minute")
async def add_history(
    request: Request,
    user_id: str,
    dish: str,
    confidence: float = None,  # type: ignore
):  # type: ignore
    conn = sqlite3.connect("../db/recipes.db")
    conn.execute(
        "INSERT INTO history (user_id, dish, confidence) VALUES (?, ?, ?",
        (user_id, dish, confidence),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@v1.get("/history/{user_id}")
@limiter.limit("60/minute")
async def get_history(request: Request, user_id: str):
    conn = sqlite3.connect("../db/recipes.db")
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT dish, confidence, timestamp FROM history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50",
        (user_id,),
    ).fetchall()
    conn.close()

    return {"user_id": user_id, "history": [dict(row) for row in rows]}


@v1.get("/similar/{dish}")
@limiter.limit("60/minute")
async def get_similar(request: Request, dish: str):
    conn = sqlite3.connect("../db/recipes.db")
    conn.row_factory = sqlite3.Row

    row = conn.execute(
        "SELECT ingredients FROM recipes WHERE dish = ? LIMIT 1",
        (dish,),
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="No similar dishes found")

    target_ingredients = set(json.loads(row["ingredients"]))

    rows = conn.execute(
        "SELECT dish, ingredients FROM recipes WHERE dish != ? GROUP BY dish",
        (dish,),
    ).fetchall()
    conn.close()

    scores = []

    for r in rows:
        ingredients = set(json.loads(r["ingredients"]))
        overlap = len(target_ingredients & ingredients)
        if overlap > 0:
            scores.append({"dish": r["dish"], "shared_ingredients": overlap})

    scores.sort(key=lambda x: x["shared_ingredients"], reverse=True)

    return {"dish": dish, "similar": scores[:5]}


@v1.get("/portion/{dish}")
@limiter.limit("60/minute")
async def get_portion(request: Request, dish: str):
    portion_g = PORTION_SIZES.get(dish, DEFAULT_PORTION)
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
        raise HTTPException(status_code=404, detail="Nutrition info not found")

    data = response.json()
    if not data["products"]:
        raise HTTPException(status_code=404, detail="Nutrition info not found")

    nutrients = data["products"][0].get("nutriments", {})
    per100 = nutrients.get("energy-kcal_100g", 0)

    factor = portion_g / 100

    return {
        "dish": dish,
        "portion_g": portion_g,
        "calories": round((per100 or 0) * factor),
        "protein_g": round((nutrients.get("proteins_100g") or 0) * factor, 1),
        "carbs_g": round((nutrients.get("carbohydrates_100g") or 0) * factor, 1),
        "fat_g": round((nutrients.get("fat_100g") or 0) * factor, 1),
        "fiber_g": round((nutrients.get("fiber_100g") or 0) * factor, 1),
        "sugar_g": round((nutrients.get("sugars_100g") or 0) * factor, 1),
    }


@v1.get("/health")
@limiter.limit("60/minute")
async def health(request: Request):
    try:
        conn = sqlite3.connect("../db/recipes.db")
        conn.execute("SELECT COUNT(*) FROM recipes")
        recipe_count = conn.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]
        conn.close()
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {str(e)}"
        recipe_count = 0

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "db": db_status,
        "recipe_count": recipe_count,
    }


@v1.post("/classify_mock")
async def classify_mock():
    dish = random.choice(classes) if classes else "pizza"
    confidence = round(random.uniform(0.5, 0.99), 4)

    return {
        "uncertain": False,
        "dish": dish,
        "confidence": confidence,
        "top5": [
            {"dish": dish, "confidence": confidence},
        ],
    }


@v1.post("/auth/register")
async def register(request: Request, body: dict):
    username = (body.get("username") or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username required")

    token = str(uuid.uuid4())

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, token) VALUES (?, ?)", (username, token)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Username already exists")
    finally:
        conn.close()

    return {"username": username, "token": token}


app.include_router(v1)

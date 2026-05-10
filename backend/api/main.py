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
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
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
from fastapi.openapi.utils import get_openapi  # type: ignore
from fastapi.security import HTTPBearer  # type: ignore
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

security = HTTPBearer()

logger = logging.getLogger("keki")

app = FastAPI(title="Keki API", swagger_ui_parameters={"persistAuthorization": True})

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


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="Keki API",
        version="1.0.0",
        routes=app.routes,
    )
    if "components" not in schema:
        schema["components"] = {}
    schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
        }
    }
    # Apply security globally to all routes
    for path in schema["paths"].values():
        for method in path.values():
            method["security"] = [{"BearerAuth": []}]
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000

    method = request.method
    path = request.url.path
    status = response.status_code
    ip = request.client.host if request.client else "-"

    # Log to stdout (visible in journalctl)
    logger.info("%s %s %d %.1fms %s", method, path, status, duration, ip)

    # Log to DB in a try/except so a DB hiccup never kills the response
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO logs (method, path, status_code, duration_ms, client_ip) VALUES (?,?,?,?,?)",
            (method, path, status, round(duration, 2), ip),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("Failed to write to logs table: %s", e)

    return response


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore

model = FoodCNN(num_classes=101).to(device)
MODEL_PATH = Path(__file__).parent.parent / "model" / "onnx" / "foodcnn.onnx"
session = ort.InferenceSession(str(MODEL_PATH))
INPUT_NAME = session.get_inputs()[0].name

with open(
    Path(__file__).parent.parent
    / "model"
    / "trainingset"
    / "food-101"
    / "meta"
    / "train.json"
) as f:
    CLASSES = sorted(json.load(f).keys())

TTA_TRANSFORMS = [
    T.Compose(
        [
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    T.Compose(
        [
            T.Resize((128, 128)),
            T.CenterCrop(128),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    T.Compose(
        [
            T.Resize((144, 144)),
            T.CenterCrop(128),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    T.Compose(
        [
            T.Resize((144, 144)),
            T.CenterCrop(128),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    T.Compose(
        [
            T.Resize((128, 128)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
]

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
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    all_probs = []
    for transform in TTA_TRANSFORMS:
        tensor = transform(img).unsqueeze(0).numpy()  # type: ignore
        outputs = session.run(None, {INPUT_NAME: tensor})[0]
        all_probs.append(softmax(outputs[0]))  # type: ignore

    avg_probs = np.mean(all_probs, axis=0)
    top5_idx = avg_probs.argsort()[::-1][:5]

    predictions = [
        {"dish": CLASSES[i], "confidence": round(float(avg_probs[i]), 4)}
        for i in top5_idx
    ]

    return {
        "predictions": predictions,
        "uncertain": predictions[0]["confidence"] < 0.35,
    }


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


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
    q_clean = q.strip().lower()

    fuzzy_classes = get_close_matches(
        q_clean, [c.lower() for c in CLASSES], n=5, cutoff=0.6
    )

    term = f"%{q_clean}%"
    conn = get_db()
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT dish, name, ingredients, servings, tags, allergens
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
        ).fetchall()
    finally:
        conn.close()

    results = [
        {
            "dish": row["dish"],
            "name": row["name"],
            "ingredients": json.loads(row["ingredients"]) if row["ingredients"] else [],
            "servings": row["servings"],
            "tags": row["tags"].split(",") if row["tags"] else [],
            "allergens": row["allergens"].split(",") if row["allergens"] else [],
            "match_type": "db",
        }
        for row in rows
    ]

    existing_dishes = {r["dish"] for r in results}
    for cls in fuzzy_classes:
        if cls not in existing_dishes:
            results.append({"dish": cls, "match_type": "fuzzy_class"})

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
    dish = random.choice(CLASSES) if CLASSES else "pizza"
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


@v1.get("/recipe/{dish}/random")
@limiter.limit("30/minute")
async def random_recipe(request: Request, dish: str):
    conn = get_db()
    try:
        row = conn.execute(
            """
            SELECT dish, name, ingredients, instruction, servings, tags, allergens
            FROM recipes
            WHERE dish = ?
            ORDER BY RANDOM()
            LIMIT 1
            """,
            (dish,),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"No recipes foud for '{dish}'")

    return {
        "dish": row["dish"],
        "name": row["name"],
        "ingredients": json.loads(row["ingredients"]) if row["ingredients"] else [],
        "instructions": row["instructions"],
        "servings": row["servings"],
        "tags": row["tags"].split(",") if row["tags"] else [],
        "allergens": row["allergens"].split(",") if row["allergens"] else [],
    }


@v1.post("/favorites/{dish}")
@limiter.limit("30/minute")
async def add_favorite(request: Request, dish: str, user=Depends(require_user)):
    conn = get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO favorites (user_id, dish_name) VALUES (?, ?)",
            (user["id"], dish),
        )
        conn.commit()
    finally:
        conn.close()
    return {"dish": dish, "favorited": True}


@v1.delete("/favorites/{dish}")
@limiter.limit("30/minute")
async def remove_favorite(request: Request, dish: str, user=Depends(require_user)):
    conn = get_db()
    try:
        conn.execute(
            "DELETE FROM favorites WHERE user_id = ? AND dish_name = ?",
            (user["id"], dish),
        )
        conn.commit()
    finally:
        conn.close()
    return {"dish": dish, "favorited": False}


@v1.get("/favorites")
@limiter.limit("30/minute")
async def get_favorites(request: Request, user=Depends(require_user)):
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT dish_name, created_at FROM favorites WHERE user_id = ? ORDER BY created_at DESC",
            (user["id"],),
        ).fetchall()
    finally:
        conn.close()
    return {
        "favorites": [
            {"dish": r["dish_name"], "saved_at": r["created_at"]} for r in rows
        ]
    }


@v1.get("/dish/{dish}/fun-fact")
@limiter.limit("30/minute")
async def fun_fact(request: Request, dish: str):
    facts = FUN_FACTS.get(dish.lower().replace("-", "_"))
    if not facts:
        return {"dish": dish, "fact": None, "message": "No fun fact yet for this dish"}

    return {"dish": dish, "fact": random.choice(facts)}


@v1.get("/metrics")
async def metrics(request: Request, _=Depends(require_admin)):
    conn = get_db()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM logs WHERE timestamp >= datetime('now', '-1 day'"
        ).fetchone()[0]

        avg_latency = conn.execute(
            "SELECT AVG(duration_ms) FROM logs WHERE timestamp >= datetime('now', '-1 day'"
        ).fetchone()[0]

        by_path = conn.execute(
            """
            SELECT path, COUNT(*) as count, AVG(duration_ms) as avg_ms,
                SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END) as errors
            FROM logs
            WHERE timestamp >= datetime('now', '-1 day')
            GROUP BY path
            ORDER BY count DESC
            """
        ).fetchall()

        top_ips = conn.execute(
            """
            SELECT client_ip, COUNT(*) as count
            FROM logs
            WHERE timestamp >= datetime('now', '-1 day')
            GROUP BY client_ip
            ORDER BY count DESC
            LIMIT 10
            """
        ).fetchall()

    finally:
        conn.close()

    return {
        "period": "last_24h",
        "total_requests": total,
        "avg_latency_ms": round(avg_latency, 2) if avg_latency else 0,
        "by_endpoint": [
            {
                "path": r["path"],
                "requests": r["count"],
                "avg_ms": round(r["avg_ms"], 2),
                "errors": r["errors"],
            }
            for r in by_path
        ],
        "top_ips": [{"ip": r["client_ip"], "requests": r["count"]} for r in top_ips],
    }


app.include_router(v1)

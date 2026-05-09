import json
import sqlite3
import time
from pathlib import Path

import requests

DB_PATH = Path(__file__).parent / "recipes.db"
SPOONACULAR_KEY = "a46212bc5c854a9ca6788849ff561d84"
DAILY_LIMIT = 140

FOOD_101_CLASSES = [
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad",
    "beignets",
    "bibimbap",
    "bread_pudding",
    "breakfast_burrito",
    "bruschetta",
    "caesar_salad",
    "cannoli",
    "caprese_salad",
    "carrot_cake",
    "ceviche",
    "cheese_plate",
    "cheesecake",
    "chicken_curry",
    "chicken_quesadilla",
    "chicken_wings",
    "chocolate_cake",
    "chocolate_mousse",
    "churros",
    "clam_chowder",
    "club_sandwich",
    "crab_cakes",
    "creme_brulee",
    "croque_madame",
    "cup_cakes",
    "deviled_eggs",
    "donuts",
    "dumplings",
    "edamame",
    "eggs_benedict",
    "escargots",
    "falafel",
    "filet_mignon",
    "fish_and_chips",
    "foie_gras",
    "french_fries",
    "french_onion_soup",
    "french_toast",
    "fried_calamari",
    "fried_rice",
    "frozen_yogurt",
    "garlic_bread",
    "gnocchi",
    "greek_salad",
    "grilled_cheese_sandwich",
    "grilled_salmon",
    "guacamole",
    "gyoza",
    "hamburger",
    "hot_and_sour_soup",
    "hot_dog",
    "huevos_rancheros",
    "hummus",
    "ice_cream",
    "lasagna",
    "lobster_bisque",
    "lobster_roll_sandwich",
    "macaroni_and_cheese",
    "macarons",
    "miso_soup",
    "mussels",
    "nachos",
    "omelette",
    "onion_rings",
    "oysters",
    "pad_thai",
    "paella",
    "pancakes",
    "panna_cotta",
    "peking_duck",
    "pho",
    "pizza",
    "pork_chop",
    "poutine",
    "prime_rib",
    "pulled_pork_sandwich",
    "ramen",
    "ravioli",
    "red_velvet_cake",
    "risotto",
    "samosa",
    "sashimi",
    "scallops",
    "seaweed_salad",
    "shrimp_and_grits",
    "spaghetti_bolognese",
    "spaghetti_carbonara",
    "spring_rolls",
    "steak",
    "strawberry_shortcake",
    "sushi",
    "tacos",
    "takoyaki",
    "tiramisu",
    "tuna_tartare",
    "waffles",
]

ALIASES = {
    "beef_carpaccio": "carpaccio",
    "beef_tartare": "steak tartare",
    "bibimbap": "bibim bap",
    "foie_gras": "foie gras pate",
    "frozen_yogurt": "froyo",
    "grilled_cheese_sandwich": "grilled cheese",
    "huevos_rancheros": "huevos",
    "lobster_roll_sandwich": "lobster roll",
    "macaroni_and_cheese": "mac and cheese",
    "pulled_pork_sandwich": "pulled pork",
    "seaweed_salad": "wakame salad",
    "shrimp_and_grits": "shrimp grits",
    "spaghetti_carbonara": "carbonara",
    "tuna_tartare": "tuna tartar",
}

MEALDB_ALIASES = {
    "beef_carpaccio": "carpaccio",
    "beef_tartare": "tartare",
    "bibimbap": "bibimbap",
    "fish_and_chips": "chips",
    "french_fries": "chips",
    "french_toast": "french toast",
    "garlic_bread": "garlic bread",
    "gnocchi": "gnocchi",
    "grilled_cheese_sandwich": "grilled cheese",
    "grilled_salmon": "salmon",
    "guacamole": "guacamole",
    "gyoza": "gyoza",
    "hamburger": "burger",
    "hot_dog": "hot dog",
    "macaroni_and_cheese": "mac and cheese",
    "miso_soup": "miso",
    "nachos": "nachos",
    "onion_rings": "onion rings",
    "pancakes": "pancakes",
    "ravioli": "ravioli",
    "samosa": "samosa",
    "sashimi": "sashimi",
    "spring_rolls": "spring rolls",
    "tiramisu": "tiramisu",
    "waffles": "waffles",
}


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dish TEXT NOT NULL,
            name TEXT,
            ingredients TEXT,
            instructions TEXT,
            source TEXT,
            servings INTEGER
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dish ON recipes (dish)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dish_predicted TEXT NOT NULL,
            dish_correct TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            dish TEXT NOT NULL,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
    conn.commit()


def dish_already_seeded(conn, dish):
    row = conn.execute(
        "SELECT COUNT(*) FROM recipes WHERE dish = ?", (dish,)
    ).fetchone()
    return row[0] > 0


def fetch_from_mealdb(dish_name):
    query = dish_name.replace("_", " ")
    url = f"https://www.themealdb.com/api/json/v1/1/search.php?s={query}"
    response = requests.get(url)
    data = response.json()

    if not data["meals"] and dish_name in MEALDB_ALIASES:
        url = f"https://www.themealdb.com/api/json/v1/1/search.php?s={MEALDB_ALIASES[dish_name]}"
        response = requests.get(url)
        data = response.json()

    if not data["meals"]:
        return []

    results = []
    for meal in data["meals"][:5]:
        ingredients = []
        for i in range(1, 21):
            ingredient = (meal.get(f"strIngredient{i}") or "").strip()
            measure = (meal.get(f"strMeasure{i}") or "").strip()
            if ingredient:
                ingredients.append(f"{measure} {ingredient}".strip())

        results.append(
            {
                "name": meal["strMeal"],
                "ingredients": json.dumps(ingredients),
                "instructions": meal["strInstructions"],
                "source": "themealdb",
                "servings": 4,
            }
        )

    return results


def fetch_from_spoonacular(dish_name, requests_used):
    if requests_used >= DAILY_LIMIT:
        return [], requests_used

    query = dish_name.replace("_", " ")

    search_url = "https://api.spoonacular.com/recipes/complexSearch"
    search_resp = requests.get(
        search_url,
        params={
            "query": query,
            "number": 5,
            "apiKey": SPOONACULAR_KEY,
        },
    )
    requests_used += 1

    search_data = search_resp.json()
    if not search_data.get("results") and dish_name in ALIASES:
        search_resp = requests.get(
            search_url,
            params={
                "query": ALIASES[dish_name],
                "number": 5,
                "apiKey": SPOONACULAR_KEY,
            },
        )
        search_data = search_resp.json()
        requests_used += 1
    if not search_data.get("results"):
        return [], requests_used
    results = []
    for result in search_data["results"]:
        if requests_used >= DAILY_LIMIT:
            break

        detail_url = f"https://api.spoonacular.com/recipes/{result['id']}/information"
        detail_resp = requests.get(detail_url, params={"apiKey": SPOONACULAR_KEY})
        requests_used += 1

        detail = detail_resp.json()

        ingredients = [ing["original"] for ing in detail.get("extendedIngredients", [])]

        instructions = detail.get("instructions") or ""

        if instructions:
            results.append(
                {
                    "name": detail["title"],
                    "ingredients": json.dumps(ingredients),
                    "instructions": instructions,
                    "source": "spoonacular",
                    "servings": detail.get("servings", 4),
                }
            )

        time.sleep(0.3)

    return results, requests_used


def seed():
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    spoonacular_requests = 0
    found = 0
    missing = []
    skipped = 0

    for dish in FOOD_101_CLASSES:
        if dish_already_seeded(conn, dish):
            print(f"Skipping {dish} (already seeded)")
            skipped += 1
            continue

        print(f"Fetching {dish}...", end=" ")

        recipes = fetch_from_mealdb(dish)

        if not recipes:
            if spoonacular_requests >= DAILY_LIMIT:
                print("✗ (daily limit reached, will retry tomorrow)")
                missing.append(dish)
                continue

            recipes, spoonacular_requests = fetch_from_spoonacular(
                dish, spoonacular_requests
            )

        if recipes:
            for recipe in recipes:
                conn.execute(
                    """
                    INSERT INTO recipes (dish, name, ingredients, instructions, source, servings)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        dish,
                        recipe["name"],
                        recipe["ingredients"],
                        recipe["instructions"],
                        recipe["source"],
                        recipe.get("servings", 4),
                    ),
                )
            conn.commit()
            found += 1
            print(
                f"✓ ({len(recipes)} recipes, {spoonacular_requests} spoonacular requests used)"
            )
        else:
            missing.append(dish)
            print("✗ not found")

        time.sleep(0.5)

    conn.close()

    print(f"\nDone! {found} new dishes seeded, {skipped} skipped.")
    if missing:
        print(f"Still missing: {', '.join(missing)}")


if __name__ == "__main__":
    seed()

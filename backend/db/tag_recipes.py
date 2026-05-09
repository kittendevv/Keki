import json
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "recipes.db")

# --- Ingredient keyword lists ---
# These are checked against each ingredient string (lowercased).
# Add more as needed — the more specific the better.

# If ANY of these appear in the ingredients, the recipe is NOT vegetarian.
MEAT_KEYWORDS = [
    "chicken",
    "beef",
    "pork",
    "lamb",
    "turkey",
    "bacon",
    "sausage",
    "ham",
    "duck",
    "veal",
    "venison",
    "anchovy",
    "anchovies",
    "tuna",
    "salmon",
    "shrimp",
    "prawn",
    "crab",
    "lobster",
    "clam",
    "mussel",
    "squid",
    "fish",
    "meat",
    "gelatin",
    "lard",
    "pancetta",
    "chorizo",
]

# If ANY of these appear, the recipe is NOT vegan (but may still be vegetarian).
ANIMAL_PRODUCT_KEYWORDS = [
    "milk",
    "cream",
    "butter",
    "cheese",
    "yogurt",
    "egg",
    "eggs",
    "honey",
    "ghee",
    "whey",
    "casein",
    "lactose",
]

# Gluten sources — if any present, tag "contains-gluten", else "gluten-free"
GLUTEN_KEYWORDS = [
    "flour",
    "wheat",
    "bread",
    "pasta",
    "noodle",
    "barley",
    "rye",
    "semolina",
    "couscous",
    "breadcrumb",
    "soy sauce",  # most soy sauce has wheat
    "beer",
    "malt",
]

# Dairy sources
DAIRY_KEYWORDS = [
    "milk",
    "cream",
    "butter",
    "cheese",
    "yogurt",
    "ghee",
    "whey",
    "casein",
    "lactose",
    "mozzarella",
    "parmesan",
    "cheddar",
    "brie",
]

# Nut sources
NUT_KEYWORDS = [
    "almond",
    "cashew",
    "walnut",
    "pecan",
    "pistachio",
    "hazelnut",
    "macadamia",
    "pine nut",
    "peanut",
    "nut",
]

# Egg sources
EGG_KEYWORDS = ["egg", "eggs", "mayonnaise", "mayo"]

# Shellfish
SHELLFISH_KEYWORDS = [
    "shrimp",
    "prawn",
    "crab",
    "lobster",
    "clam",
    "mussel",
    "oyster",
    "scallop",
]

# Soy
SOY_KEYWORDS = ["soy", "tofu", "tempeh", "edamame", "miso", "soy sauce"]


def ingredients_contain(ingredients: list[str], keyword: list[str]) -> bool:
    combined = " ".join(ingredients).lower()
    return any(kw in combined for kw in keyword)


def tag_recipe(ingredients: list[str]) -> tuple[list[str], list[str]]:
    has_meat = ingredients_contain(ingredients, MEAT_KEYWORDS)
    has_animal = ingredients_contain(ingredients, ANIMAL_PRODUCT_KEYWORDS)
    has_gluten = ingredients_contain(ingredients, GLUTEN_KEYWORDS)
    has_dairy = ingredients_contain(ingredients, DAIRY_KEYWORDS)
    has_nuts = ingredients_contain(ingredients, NUT_KEYWORDS)
    has_eggs = ingredients_contain(ingredients, EGG_KEYWORDS)
    has_shellfish = ingredients_contain(ingredients, SHELLFISH_KEYWORDS)
    has_soy = ingredients_contain(ingredients, SOY_KEYWORDS)

    tags = []
    if not has_meat:
        tags.append("vegetarian")
    if not has_meat and not has_animal:
        tags.append("vegan")
    if not has_gluten:
        tags.append("gluten-free")
    if not has_dairy:
        tags.append("dairy-free")

    allergens = []
    if has_gluten:
        allergens.append("gluten")
    if has_dairy:
        allergens.append("dairy")
    if has_nuts:
        allergens.append("nuts")
    if has_eggs:
        allergens.append("eggs")
    if has_shellfish:
        allergens.append("shellfish")
    if has_soy:
        allergens.append("soy")

    return tags, allergens


conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

rows = cur.execute("SELECT id, ingredients FROM recipes").fetchall()
updated = 0

for row in rows:
    try:
        ingredients = json.loads(row["ingredients"]) if row["ingredients"] else []
    except json.JSONDecodeError:
        ingredients = []

    tags, allergens = tag_recipe(ingredients)

    cur.execute(
        "UPDATE recipes SET tags = ?, allergens = ? WHERE id = ?",
        (",".join(tags), ",".join(allergens), row["id"]),
    )
    updated += 1

conn.commit()
conn.close()
print(f"Tagged {updated} recipes with dietary tags and allergens.")

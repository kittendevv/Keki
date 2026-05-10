import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "recipes.db")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS favorites (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        dish_name  TEXT NOT NULL,
        created_at TEXT,
        UNIQUE(user_id, dish_name)
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        username   TEXT NOT NULL UNIQUE,
        token      TEXT NOT NULL UNIQUE,
        created_at TEXT
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        method      TEXT,
        path        TEXT,
        status_code INTEGER,
        duration_ms REAL,
        client_ip   TEXT,
        timestamp   TEXT
    )
""")

for column, defenition in [
    ("tags", "TEXT DEFAULT ''"),
    ("allergens", "TEXT DEFAULT ''"),
]:
    try:
        cur.execute(f"ALTER TABLE recipes ADD COLUMN {column} {defenition}")
        print(f"Added column {column}")
    except sqlite3.OperationalError:
        print(f"Column {column} already exists, skipping")

cur.execute("CREATE INDEX IF NOT EXISTS idx_logs_path ON logs(path)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_favorites_user ON favorites(user_id)")

conn.commit()
conn.close()
print("Database migration completed successfully.")

import os
import sqlite3
import uuid

from fastapi import Header, HTTPException, status  # type: ignore

DB_PATH = os.path.join(os.path.dirname(__file__), "../../db/recipes.db")

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "dev")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def require_user(authorization: str = Header(...)) -> dict:

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth header"
        )

    token = authorization.removeprefix("Bearer ").strip()

    conn = get_db()
    try:
        user = conn.execute(
            "SELECT id, username, token FROM users WHERE token = ?", (token,)
        ).fetchone()
    finally:
        conn.close()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    return dict(user)


def require_admin(authorization: str = Header(...)) -> dict:

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth header"
        )

    token = authorization.removeprefix("Bearer ").strip()

    if token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin only"
        )

    return True  # type: ignore

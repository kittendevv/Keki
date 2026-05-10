import os
import sqlite3
import uuid

import bcrypt
from fastapi import Depends, Header, HTTPException, status  # type: ignore
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # type: ignore

DB_PATH = os.path.join(os.path.dirname(__file__), "../db/recipes.db")

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "dev")

bearer_scheme = HTTPBearer()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed(encode))


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def require_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    token = credentials.credentials  # already stripped of "Bearer "

    conn = get_db()
    try:
        user = conn.execute(
            "SELECT id, username, token FROM users WHERE token = ?", (token,)
        ).fetchone()
    finally:
        conn.close()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    return dict(user)


def require_admin(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> bool:
    token = credentials.credentials

    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Admin only")

    return True

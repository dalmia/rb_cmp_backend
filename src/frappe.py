import requests
import json
from settings import settings
from db import get_action_for_user
from fastapi import HTTPException
from settings import settings
import asyncpg


async def get_db_connection():
    conn = await asyncpg.connect(settings.database_url)
    return conn


async def get_event_types():
    conn: asyncpg.Connection = await get_db_connection()

    query = 'SELECT * from "tabEvent Type"'
    rows = await conn.fetch(query)

    return [row["name"] for row in rows]


async def get_event_categories():
    conn: asyncpg.Connection = await get_db_connection()

    query = 'SELECT * from "tabEvent Category"'
    rows = await conn.fetch(query)

    return [row["name"] for row in rows]


async def get_event_subcategories():
    conn: asyncpg.Connection = await get_db_connection()

    query = 'SELECT * from "tabEvent Sub Category"'
    rows = await conn.fetch(query)

    return [row["name"] for row in rows]


def login_user(email: str, password: str):
    url = f"{settings.frappe_backend_base_url}/method/login"

    payload = json.dumps({"usr": email, "pwd": password})
    headers = {
        "Content-Type": "application/json",
    }

    return requests.request("POST", url, headers=headers, data=payload)


def get_user_profile(email: str):
    url = f"{settings.frappe_backend_base_url}/method/solve_ninja.api.profile.get_user_profile"

    payload = json.dumps({"username": email})

    headers = {
        "Authorization": f"token {settings.frappe_backend_client_id}:{settings.frappe_backend_client_secret}",
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="User not found")

    return response.json()["message"]


async def create_action_on_frappe(action_id: int):
    url = f"{settings.frappe_backend_base_url}/method/solve_ninja.api.events.create_events"

    action_details = await get_action_for_user(action_id)

    payload = json.dumps(
        {
            "title": action_details["title"],
            "type": action_details["type"],
            "category": action_details["category"],
            "user": action_details["user"]["email"],
            "description": action_details["description"],
            "skills": {
                skill["label"]: skill["summary"] for skill in action_details["skills"]
            },
        }
    )
    headers = {
        "Authorization": f"token {settings.frappe_backend_client_id}:{settings.frappe_backend_client_secret}",
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

import requests
from typing import List, Dict, Any, Optional
import json
import os
from dotenv import load_dotenv

# Load environment variables only once
load_dotenv()
DHIS2_BASE_URL = os.getenv("DHIS2_BASE_URL")
DHIS2_USERNAME = os.getenv("DHIS2_USERNAME")
DHIS2_PASSWORD = os.getenv("DHIS2_PASSWORD")


def get_all(
    endpoint: str,
    key: str,
    fields: str = "id,name",
    filters: Optional[Dict[str, str]] = None,
    page_size: int = 1000
) -> List[Dict[str, Any]]:
    page = 1
    all_items = []

    while True:
        params = {
            "page": page,
            "pageSize": page_size,
            "fields": fields
        }
        if filters:
            for field, condition in filters.items():
                params.setdefault("filter", []).append(f"{field}:{condition}")
        url = f"{DHIS2_BASE_URL}/api/{endpoint}"
        response = requests.get(url, auth=(DHIS2_USERNAME, DHIS2_PASSWORD), params=params)
        response.raise_for_status()
        data = response.json()
        items = data.get(key, [])
        if not items:
            break
        all_items.extend(items)

        pager = data.get("pager", {})
        if pager.get("page", 0) >= pager.get("pageCount", 0):
            break

        page += 1

    return all_items

def generate_dhis2_ids(count: int = 1) -> str:
    """Generate one or more unique DHIS2 IDs using the system's identifier endpoint."""
    if not all([DHIS2_BASE_URL, DHIS2_USERNAME, DHIS2_PASSWORD]):
        return "❌ Missing DHIS2 credentials"
    try:
        response = requests.get(
            f"{DHIS2_BASE_URL}/api/system/id",
            auth=(DHIS2_USERNAME, DHIS2_PASSWORD),
            params={"limit": count}
        )
        response.raise_for_status()
        ids = response.json().get("codes", [])
        return "\n".join(ids) if ids else "⚠️ No IDs returned."
    except Exception as e:
        return f"❌ Failed to generate ID(s): {str(e)}"
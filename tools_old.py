from typing import Dict, Any
from langchain.tools import tool
from dotenv import load_dotenv
import os
import requests
import json
import re
from typing import Optional, Dict, List

# Load .env credentials
load_dotenv()
base_url = os.getenv("DHIS2_BASE_URL")
username = os.getenv("DHIS2_USERNAME")
password = os.getenv("DHIS2_PASSWORD")


@tool
def get_dhis2_metadata(metadata_type: str, params: Optional[dict] = None) -> str:
    """
    Fetch metadata from DHIS2 dynamically with optional query parameters like page, pageSize, level, etc.
    """
    if not all([base_url, username, password]):
        return "❌ Missing DHIS2 credentials"

    url = f"{base_url}/api/{metadata_type}.json"
    try:
        response = requests.get(url, auth=(username, password), params=params)
        response.raise_for_status()
        data = response.json()

        items = data.get(metadata_type, [])
        pager = data.get("pager", {})
        current_page = pager.get("page", 1)
        total_pages = pager.get("pageCount", 1)

        names = "\n".join(
            f"{item.get('displayName', item.get('name', 'N/A'))} - ID: {item.get('id')}"
            for item in items
        )
        return f"{names}\n\nThis is page {current_page} of {total_pages}."
    except requests.exceptions.RequestException as e:
        return f"❌ Request error: {str(e)}"


@tool
def create_metadata(input_str: str) -> str:
    """
    Create DHIS2 metadata using JSON input: {"endpoint": "...", "payload": {...}}
    """
    return dhis2_tool_metadata_wrapper(input_str)

def dhis2_tool_metadata_wrapper(input_str: str) -> str:
    try:
        data = json.loads(input_str)
        endpoint = data["endpoint"]
        payload = data["payload"]
        schema_name = endpoint.rstrip("s")

        # Normalize to list
        if not isinstance(payload, list):
            payload = [payload]

        required_fields = get_required_fields_for_schema(schema_name)

        for item in payload:
            if schema_name == "dataElement":
                item.setdefault("aggregationType", "SUM")
                item.setdefault("categoryCombo", {"id": "bjDvmb4bfuf"})
                item.setdefault("zeroIsSignificant", False)

            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                fields_list = ", ".join(missing_fields)
                schema_url = f"/api/schemas/{schema_name}.json"
                return (
                    f"ℹ️ To create a new `{schema_name}`, please provide the following required field(s): {fields_list}.\n"
                    f"Refer to the full schema here: `{schema_url}`"
                )

        # All valid → submit once
        wrapped_payload = {endpoint: payload}
        return post_dhis2_api(endpoint, wrapped_payload)

    except Exception as e:
        return f"❌ Invalid input or schema validation failed: {str(e)}"


def validate_and_post_dhis2_item(endpoint: str, schema_name: str, payload: dict) -> str:
    try:
        # Inject defaults for dataElement
        if schema_name == "dataElement":
            payload.setdefault("aggregationType", "SUM")
            payload.setdefault("categoryCombo", {"id": "bjDvmb4bfuf"})
            payload.setdefault("zeroIsSignificant", False)

        required_fields = get_required_fields_for_schema(schema_name)
        print("===========================")
        print(payload)
        print("===========================")
        missing_fields = [field for field in required_fields if field not in payload]
        if missing_fields:
            return f"❌ Payload missing required fields for `{schema_name}`: {missing_fields}"

        wrapped_payload = {endpoint: [payload]}
        return post_dhis2_api(endpoint, wrapped_payload)
    except Exception as e:
        return f"❌ Error validating or posting `{schema_name}`: {str(e)}"


def get_required_fields_for_schema(schema_name: str) -> list:
    try:
        url = f"{base_url}/api/schemas/{schema_name}.json"
        print(f"🔍 Fetching schema: {url}")
        response = requests.get(url, auth=(username, password))
        response.raise_for_status()
        schema = response.json()
        return [prop["name"] for prop in schema.get("properties", []) if prop.get("required")]
    except Exception as e:
        raise RuntimeError(f"Failed to fetch schema for `{schema_name}`: {str(e)}")

def post_dhis2_api(endpoint: str, payload: dict) -> str:
    try:
        # url = f"{DHIS2_URL}/{endpoint.strip().strip('\"\'').lstrip('/')}"
        url = f"{base_url}/api/metadata"
        print(f"📤 Posting to: {url}\n")
        headers = {"Content-Type": "application/json"}
        print("===================")
        print(json.dumps(payload))
        print("===================")
        params = {'importStrategy': 'CREATE_UPDATE'}
        response = requests.post(url,
                             auth=(username, password),
                             headers=headers,
                             data=json.dumps(payload),
                             params=params)
        response.raise_for_status()
        result = response.json()
        return f"✅ POST successful to `{endpoint}`.\nResponse: {json.dumps(result, indent=2)}"
    except Exception as e:
        return f"❌ POST failed for endpoint `{endpoint}`: {str(e)}"




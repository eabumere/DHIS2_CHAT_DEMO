from typing import Dict, Any, Optional, List, Union
from langchain.tools import tool
from dotenv import load_dotenv
import os
import requests
import json
import time
from pathlib import Path
from re_order import reorder_dhis2_metadata_fixed
from .schema import get_required_fields_for_schema
import random
import string

# Load .env credentials
load_dotenv()
base_url = os.getenv("DHIS2_BASE_URL")
username = os.getenv("DHIS2_USERNAME")
password = os.getenv("DHIS2_PASSWORD")

SCHEMA_DIR = Path(__file__).parent.parent / "schema"

def generate_suffix(length=4):
    return '_' + ''.join(random.choices(string.ascii_letters + string.digits, k=length))

@tool
def get_dhis2_metadata(metadata_type: str, params: Optional[dict] = None) -> str:
    """Retrieve metadata from a DHIS2 instance using specified filters or identifiers."""
    if not all([base_url, username, password]):
        return "❌ Missing DHIS2 credentials"
    url = f"{base_url}/api/{metadata_type}.json?fields=*"
    try:
        response = requests.get(url, auth=(username, password), params=params)
        response.raise_for_status()
        data = response.json()
        items = data.get(metadata_type, [])
        print(items)
        pager = data.get("pager", {})
        current_page = pager.get("page", 1)
        total_pages = pager.get("pageCount", 1)
        names = "\n".join(
            f"{item.get('displayName', item.get('name', 'N/A'))} - ID: {item.get('id')}"
            for item in items
        )
        print('name -', names)
        # return f"{names}\n\nThis is page {current_page} of {total_pages}."
        return items
    except requests.exceptions.RequestException as e:
        return f"❌ Request error: {str(e)}"

@tool
def generate_dhis2_ids(count: int = 1) -> str:
    """Generate one or more unique DHIS2 IDs using the system's identifier endpoint."""
    if not all([base_url, username, password]):
        return "❌ Missing DHIS2 credentials"
    try:
        response = requests.get(
            f"{base_url}/api/system/id",
            auth=(username, password),
            params={"limit": count}
        )
        response.raise_for_status()
        ids = response.json().get("codes", [])
        return "\n".join(ids) if ids else "⚠️ No IDs returned."
    except Exception as e:
        return f"❌ Failed to generate ID(s): {str(e)}"

@tool
def get_dhis2_version() -> str:
    """Fetch the current version of the DHIS2 instance from the system info endpoint."""
    if not all([base_url, username, password]):
        return "❌ Missing DHIS2 credentials"
    try:
        response = requests.get(f"{base_url}/api/system/info", auth=(username, password))
        response.raise_for_status()
        return response.json().get("version", "unknown")
    except Exception as e:
        return f"❌ Failed to fetch DHIS2 version: {str(e)}"

def flatten_keys(obj: Union[Dict, List], parent_key: str = "") -> set:
    """Recursively flatten dictionary keys for comparison."""
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{parent_key}.{k}" if parent_key else k
            keys.add(full_key)
            keys.update(flatten_keys(v, full_key))
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        keys.update(flatten_keys(obj[0], parent_key))
    return keys


@tool
def validate_metadata_against_schema(metadata: dict) -> dict:
    """Validate and fix metadata payload structure against local schemas."""
    from jsonschema import validate, ValidationError

    # Ensure metadata is nested under 'metadata' key
    if "metadata" not in metadata:
        metadata = {"metadata": metadata}

    fixed_metadata = metadata.copy()
    issues = []

    for schema_file in SCHEMA_DIR.glob("*.json"):
        with open(schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)

        try:
            validate(instance=metadata["metadata"], schema=schema)
            return {
                "status": "valid",
                "schema": schema_file.stem,
                "metadata": metadata
            }

        except ValidationError as e:
            # Attempt to extract and patch missing fields if possible
            path = list(e.path)
            missing_field = str(e.message).split("'")[1] if "'" in str(e.message) else "unknown"
            issues.append({
                "schema": schema_file.stem,
                "error": str(e),
                "path": path,
                "missing_field": missing_field
            })

    return {
        "status": "invalid",
        "issues": issues,
        "suggestion": "Use `create_metadata` or repair tool to auto-complete required fields."
    }


def get_schema_with_retry(schema_name: str, max_retries: int = 3, backoff_factor: float = 1.5) -> dict:
    url = f"{base_url}/api/schemas/{schema_name}.json"
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, auth=(username, password))
            response.raise_for_status()
            data = response.json()
            if "name" in data:
                return data
            raise ValueError("Schema structure invalid.")
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Failed to fetch schema `{schema_name}` after {max_retries} attempts: {str(e)}")
            time.sleep(backoff_factor ** attempt)


def post_dhis2_api(endpoint: str, payload: dict, params: Optional[dict] = None) -> dict:
    url = f"{base_url}/api/metadata"
    try:
        response = requests.post(url, auth=(username, password), json=payload, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def read_response(response_json):
    print(response_json)
    type_reports = response_json.get("response", {}).get("typeReports", [])
    created_items = []
    updated_items = []

    for report in type_reports:
        klass = report.get("klass", "Unknown")
        # Strip "org.hisp.dhis." prefix
        friendly_name = klass.split('.')[-1]
        stats = report.get("stats", {})
        created = stats.get("created", 0)
        updated = stats.get("updated", 0)

        if created:
            created_items.append(f"{created} {friendly_name}")
        if updated:
            updated_items.append(f"{updated} {friendly_name}")

    message_parts = []
    if created_items:
        message_parts.append("Created: " + ", ".join(created_items))
    if updated_items:
        message_parts.append("Updated: " + ", ".join(updated_items))

    message = "; ".join(message_parts) if message_parts else "No changes made."

    return json.dumps({
        "status": "success",
        "createdItems": created_items,
        "updatedItems": updated_items,
        "message": message
    }, indent=2)


def interpret_post_response(response: dict) -> str:
    if not response:
        return "❌ No response from server."
    if response.get("status") == "OK":
        import_count = response.get("importCount", {})
        summary = ", ".join(f"{k}: {v}" for k, v in import_count.items() if v > 0)
        return f"✅ Metadata imported. {summary}"
    if "status" in response:
        return f"❌ Import failed. Status: {response['status']} - {response.get('message', '')}"
    return "⚠️ Unknown response format."

def generate_suffix(length=5) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def validate_payload_against_schema(payload: dict, schema: dict) -> None:
    from jsonschema import validate, ValidationError
    try:
        validate(instance=payload, schema=schema)
    except ValidationError as e:
        raise RuntimeError(f"Schema validation error: {str(e)}")

def apply_defaults(item: dict, schema: dict):
    """Recursively apply defaults and DHIS2-specific fallback defaults."""
    required = schema.get("required", [])
    print(required)
    properties = schema.get("properties", {})
    for field in required:
        if field not in item:
            default = properties.get(field, {}).get("default")
            if default is not None:
                item[field] = default
            else:
                # DHIS2-specific defaults
                if field == "aggregationType":
                    item[field] = "SUM"
                elif field == "categoryCombo":
                    item[field] = {"id": "bjDvmb4bfuf"}
                elif field == "zeroIsSignificant":
                    item[field] = False
    for field, subschema in properties.items():
        if field in item:
            if subschema.get("type") == "object" and isinstance(item[field], dict):
                apply_defaults(item[field], subschema)
            elif subschema.get("type") == "array" and "items" in subschema and isinstance(item[field], list):
                for subitem in item[field]:
                    if isinstance(subitem, dict):
                        apply_defaults(subitem, subschema["items"])

@tool
def create_metadata(input_str: str) -> str:
    """Create metadata on the DHIS2 server. Supports single and composite metadata payloads."""
    try:
        data = json.loads(input_str)
        # Determine payload structure
        if "endpoint" in data and "payload" in data:
            payload_dict = {data["endpoint"]: data["payload"]}
        else:
            payload_dict = data  # Composite payload
        if len(payload_dict.items()) > 1:
            reordered_payload = reorder_dhis2_metadata_fixed(payload_dict)
            # print(reordered_payload)
            response_json = post_dhis2_api("metadata", reordered_payload,
                                           params={"importStrategy": "CREATE_UPDATE"})  # PASS DICT
            interpreted = read_response(response_json)
            return  interpreted
        else:
            response_json = post_dhis2_api("metadata", payload_dict, params={"importStrategy": "CREATE_UPDATE"})
            interpreted = read_response(response_json)
            return  interpreted



        # for endpoint, payload in payload_dict.items():
        #     print(endpoint)
        #     print(payload)
        #
        #     # schema_name = endpoint.rstrip("s")
        #     # if not isinstance(payload, list):
        #     #     payload = [payload]
        #     # schema = get_schema_with_retry(schema_name)
        #     # for item in payload:
        #     # #     apply_defaults(item, schema)
        #     # #     ensure_required_fields(item, schema)
        #     # #     print(item)
        #     # #
        #     # #     validate_payload_against_schema(item, schema)
        #     #     created_items.append(item.get("name") or item.get("shortName") or "Unnamed item")
        #     #
        #     # final_payload[endpoint] = payload


    except RuntimeError as e:
        return json.dumps({"status": "error", "reason": str(e)}, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "reason": f"Invalid input or schema validation failed: {str(e)}"
        }, indent=2)


from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

@tool
def delete_metadata(endpoint: str, names: List[str]) -> ToolMessage:
    """Delete metadata objects by name. Resolves names to UIDs before deletion."""
    try:
        if not names:
            return ToolMessage(tool_call_id="", content="❌ No names provided for deletion.")

        # Step 1: Fetch existing metadata
        url = f"{base_url}/api/{endpoint}.json"
        params = {"fields": "id,name", "paging": "false"}
        response = requests.get(url, auth=(username, password), params=params)
        response.raise_for_status()
        items = response.json().get(endpoint, [])

        # Step 2: Match names to UIDs
        name_to_id = {item["name"]: item["id"] for item in items}
        missing_names = [name for name in names if name not in name_to_id]
        matched_ids = [name_to_id[name] for name in names if name in name_to_id]

        if not matched_ids:
            return ToolMessage(
                tool_call_id="",
                content=f"❌ No matching items found for deletion. Unknown names: {missing_names}"
            )

        # Step 3: Delete by UID
        payload = [{"id": uid} for uid in matched_ids]
        wrapped_payload = {endpoint: payload}
        response_json = post_dhis2_api(endpoint, wrapped_payload, params={"importStrategy": "DELETE"})

        msg = read_response(response_json)  # Use your formatting here
        if missing_names:
            msg += f"\n⚠️ Not found: {', '.join(missing_names)}"

        return ToolMessage(tool_call_id="", content=msg)

    except Exception as e:
        return ToolMessage(tool_call_id="", content=f"❌ Error during deletion: {str(e)}")


@tool
def update_metadata(input_str: str) -> str:
    """Update metadata on the DHIS2 server. Supports both single and composite payloads."""
    try:
        data = json.loads(input_str)

        # Check format: {"endpoint": ..., "payload": ...} or full {"dataElements": [...], "dataSets": [...], ...}
        if "endpoint" in data and "payload" in data:
            payload_dict = {data["endpoint"]: data["payload"]}
        else:
            payload_dict = data  # Composite format

        # Handle composite payloads
        if len(payload_dict.items()) > 1:
            reordered_payload = reorder_dhis2_metadata_fixed(payload_dict)
            response_json = post_dhis2_api(
                "metadata", reordered_payload, params={"importStrategy": "UPDATE"}
            )
            return read_response(response_json)

        # Single metadata type
        endpoint = list(payload_dict.keys())[0]
        schema_name = endpoint.rstrip("s")
        items = payload_dict[endpoint]
        if not isinstance(items, list):
            items = [items]

        updated_objects = []

        for item in items:
            if "id" not in item:
                return json.dumps({
                    "status": "error",
                    "reason": f"Missing 'id' in update payload for {schema_name}.",
                    "object": item
                }, indent=2)

            # Fetch full object from DHIS2 with all fields
            existing = requests.get(
                f"{base_url}/api/{endpoint}/{item['id']}.json?fields=*",
                auth=(username, password)
            )
            existing.raise_for_status()
            full_object = existing.json()

            # Merge with incoming updates
            full_object.update(item)
            updated_objects.append(full_object)

        # Final payload and POST to metadata endpoint
        final_payload = {endpoint: updated_objects}
        response_json = post_dhis2_api(
            "metadata", final_payload, params={"importStrategy": "UPDATE"}
        )
        return read_response(response_json)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "reason": f"Update failed: {str(e)}"
        }, indent=2)



    # try:
    #     if isinstance(input_str, str):
    #         data = json.loads(input_str)
    #     elif isinstance(input_str, dict):
    #         data = input_str
    #     else:
    #         return "❌ Invalid input type. Must be a JSON string or dict."
    #     print(json.dumps(data))
    #     endpoint = data.get("endpoint")
    #     identify_by = data.get("identify_by", {})
    #     print(identify_by)
    #     update_fields = data.get("update_fields", {})
    #     print(update_fields)
    #     if not endpoint or not identify_by or not update_fields:
    #         return "❌ Missing required fields: 'endpoint', 'identify_by', or 'update_fields'."
    #     schema_name = endpoint.rstrip("s")
    #     search_url = f"{base_url}/api/{endpoint}.json"
    #     response = requests.get(
    #         search_url,
    #         auth=(username, password),
    #         params={**identify_by, "fields": "id,name", "paging": "false"}
    #     )
    #     response.raise_for_status()
    #     results = response.json().get(endpoint, [])
    #     exact_matches = [item for item in results if all(item.get(k) == v for k, v in identify_by.items())]
    #     if not exact_matches:
    #         return f"❌ No {schema_name} found matching {identify_by}"
    #     if len(exact_matches) > 1:
    #         return f"⚠️ Multiple {schema_name}s found with {identify_by}. Please use a unique 'id'."
    #     target_id = exact_matches[0]["id"]
    #     schema = get_schema_with_retry(schema_name)
    #     apply_defaults(update_fields, schema)
    #     validate_payload_against_schema(update_fields, schema)
    #     update_url = f"{base_url}/api/{endpoint}/{target_id}"
    #     response = requests.put(update_url, auth=(username, password), json=update_fields)
    #     response.raise_for_status()
    #     return f"✅ Updated {schema_name} with ID {target_id} successfully."
    # except Exception as e:
    #     return f"❌ Update failed: {str(e)}"

#
# @tool
# def update_metadata(input_str: str) -> str:
#     """Update an existing metadata object on the DHIS2 server with new values."""
#     try:
#         if isinstance(input_str, str):
#             data = json.loads(input_str)
#         elif isinstance(input_str, dict):
#             data = input_str
#         else:
#             return "❌ Invalid input type. Must be a JSON string or dict."
#         print(json.dumps(data))
#         endpoint = data.get("endpoint")
#         identify_by = data.get("identify_by", {})
#         print(identify_by)
#         update_fields = data.get("update_fields", {})
#         print(update_fields)
#         if not endpoint or not identify_by or not update_fields:
#             return "❌ Missing required fields: 'endpoint', 'identify_by', or 'update_fields'."
#         schema_name = endpoint.rstrip("s")
#         search_url = f"{base_url}/api/{endpoint}.json"
#         response = requests.get(
#             search_url,
#             auth=(username, password),
#             params={**identify_by, "fields": "id,name", "paging": "false"}
#         )
#         response.raise_for_status()
#         results = response.json().get(endpoint, [])
#         exact_matches = [item for item in results if all(item.get(k) == v for k, v in identify_by.items())]
#         if not exact_matches:
#             return f"❌ No {schema_name} found matching {identify_by}"
#         if len(exact_matches) > 1:
#             return f"⚠️ Multiple {schema_name}s found with {identify_by}. Please use a unique 'id'."
#         target_id = exact_matches[0]["id"]
#         schema = get_schema_with_retry(schema_name)
#         apply_defaults(update_fields, schema)
#         validate_payload_against_schema(update_fields, schema)
#         update_url = f"{base_url}/api/{endpoint}/{target_id}"
#         response = requests.put(update_url, auth=(username, password), json=update_fields)
#         response.raise_for_status()
#         return f"✅ Updated {schema_name} with ID {target_id} successfully."
#     except Exception as e:
#         return f"❌ Update failed: {str(e)}"

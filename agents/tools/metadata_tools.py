from typing import Dict, Any
from langchain.tools import tool
from dotenv import load_dotenv
import os
import requests
import json
import re
from typing import Optional, Dict, List
import time


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

def dhis2_tool_metadata_wrapper(input_str: str) -> dict:
    try:
        data = json.loads(input_str)
        endpoint = data["endpoint"]
        payload = data["payload"]
        schema_name = endpoint.rstrip("s")

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
                return {
                    "status": "error",
                    "reason": f"Missing required fields: {fields_list}",
                    "missingFields": missing_fields,
                    "schema": schema_url
                }

        wrapped_payload = {endpoint: payload}
        response_json = post_dhis2_api(endpoint, wrapped_payload)

        message = interpret_post_response(response_json)
        created_names = [item.get("name") or item.get("shortName") or "Unnamed item" for item in payload]

        return {
            "status": "success",
            "endpoint": endpoint,
            "payload": payload,
            "createdItems": created_names,
            "message": message
        }

    except RuntimeError as e:
        return {"status": "error", "reason": str(e)}
    except Exception as e:
        return {"status": "error", "reason": f"Invalid input or schema validation failed: {str(e)}"}

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
        schema = get_schema_with_retry(schema_name)
        return [prop["name"] for prop in schema.get("properties", []) if prop.get("required")]
    except Exception as e:
        raise RuntimeError(f"❌ Failed to get required fields for `{schema_name}`: {str(e)}")


def get_schema_with_retry(schema_name: str, max_retries: int = 3, backoff_factor: float = 1.5) -> dict:
    """
    Fetch a DHIS2 schema with retry logic.

    Args:
        schema_name (str): The name of the schema (e.g., 'dataElement').
        max_retries (int): Maximum number of retry attempts.
        backoff_factor (float): Multiplier for wait time between retries.

    Returns:
        dict: Parsed schema JSON.

    Raises:
        RuntimeError: If the schema cannot be retrieved after retries.
    """
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
            sleep_time = backoff_factor ** attempt
            time.sleep(sleep_time)


@tool
def delete_metadata(endpoint: str, uids: List[str]) -> str:
    """
    Delete DHIS2 metadata by specifying the endpoint and list of UIDs.
    Example: endpoint="dataElements", uids=["abc123", "def456"]
    """
    try:
        if not uids:
            return "❌ No UIDs provided for deletion."

        payload = [{ "id": uid } for uid in uids]
        wrapped_payload = {endpoint: payload}

        response_json = post_dhis2_api(endpoint, wrapped_payload, params={"importStrategy": "DELETE"})
        return interpret_post_response(response_json)

    except Exception as e:
        return f"❌ Error during deletion: {str(e)}"


@tool
def update_metadata(input_str: str) -> str:
    """
    Update DHIS2 metadata using either JSON string or dict input.
    {
        "endpoint": "dataElements",
        "identify_by": {"name": "OldName"},
        "update_fields": {"name": "NewName"}
    }
    """
    try:
        if isinstance(input_str, str):
            data = json.loads(input_str)
        elif isinstance(input_str, dict):
            data = input_str
        else:
            return "❌ Invalid input type. Must be a JSON string or dict."

        endpoint = data.get("endpoint")
        identify_by = data.get("identify_by", {})
        update_fields = data.get("update_fields", {})

        if not endpoint or not identify_by or not update_fields:
            return "❌ Missing required fields: 'endpoint', 'identify_by', or 'update_fields'."

        schema_name = endpoint.rstrip("s")

        # Step 1: Search for the target item
        search_url = f"{base_url}/api/{endpoint}.json"
        response = requests.get(
            search_url,
            auth=(username, password),
            params={**identify_by, "fields": "id,name", "paging": "false"}
        )
        response.raise_for_status()
        results = response.json().get(endpoint, [])

        print("==== Printing Response from Search ====")
        print(results)
        print("==== Printing Response from Search ====")

        exact_matches = [
            item for item in results if all(item.get(k) == v for k, v in identify_by.items())
        ]

        if not exact_matches:
            return f"❌ No {schema_name} found matching {identify_by}"
        if len(exact_matches) > 1:
            return f"⚠️ Multiple {schema_name}s found with {identify_by}. Please use a unique 'id'."

        target_id = exact_matches[0]["id"]

        # # Step 2: Check for name conflict
        # if "name" in update_fields:
        #     conflict_check = requests.get(
        #         search_url,
        #         auth=(username, password),
        #         params={"name": update_fields["name"], "fields": "id", "paging": "false"}
        #     )
        #     conflict_check.raise_for_status()
        #     existing = conflict_check.json().get(endpoint, [])
        #     print(" ===== Returned item Start ===== ")
        #     print(target_id)
        #     print(" ===== Returned item End ===== ")
        #     if any(item["id"] != target_id for item in existing):
        #         return f"❌ A {schema_name} with the name '{update_fields['name']}' already exists."

        # Step 3: Fetch full object before updating
        get_url = f"{base_url}/api/{endpoint}/{target_id}.json"
        full_response = requests.get(get_url, auth=(username, password))
        full_response.raise_for_status()
        full_object = full_response.json()

        # Step 4: Apply updates to full object
        full_object.update(update_fields)
        # Include id for update in metadata import payload
        full_object["id"] = target_id

        # Step 5: Submit update using post_dhis2_api with importStrategy UPDATE
        payload = {
            endpoint: [full_object]
        }
        params = {"importStrategy": "UPDATE"}

        update_result = post_dhis2_api(endpoint=endpoint, payload=payload, params=params)

        # You can inspect update_result for success or error messages
        if update_result.get("status") == "SUCCESS":
            return f"✅ {schema_name.capitalize()} updated successfully: ID = {target_id}"
        else:
            # Could extract and return detailed import summary errors here if desired
            return f"❌ Update failed: {update_result}"

    except Exception as e:
        return f"❌ Error updating metadata: {str(e)}"


def find_metadata_by_fields(endpoint: str, query_fields: Dict[str, str]) -> Optional[dict]:
    try:
        url = f"{base_url}/api/{endpoint}.json"
        response = requests.get(url, auth=(username, password), params={"paging": "false"})
        response.raise_for_status()
        items = response.json().get(endpoint, [])

        for item in items:
            if all(item.get(k) == v for k, v in query_fields.items()):
                return item
        return None
    except Exception as e:
        raise RuntimeError(f"Failed to search {endpoint}: {str(e)}")


def post_dhis2_api(endpoint: str, payload: dict, params: dict = None) -> dict:
    try:
        url = f"{base_url}/api/metadata"
        headers = {"Content-Type": "application/json"}
        request_params = params if params else {'importStrategy': 'CREATE'}

        response = requests.post(
            url,
            auth=(username, password),
            headers=headers,
            data=json.dumps(payload),
            params=request_params
        )
        response.raise_for_status()
        return response.json()  # Return parsed JSON response
    except Exception as e:
        raise RuntimeError(f"POST failed for endpoint `{endpoint}`: {str(e)}")

def interpret_post_response(response_json: dict) -> str:
    status = response_json.get("status")
    response_type = response_json.get("responseType")
    conflicts = response_json.get("conflicts", [])
    import_count = response_json.get("importCount", {})
    created = import_count.get("imported", 0)
    updated = import_count.get("updated", 0)
    ignored = import_count.get("ignored", 0)

    if status != "OK" or response_type == "ERROR":
        if conflicts:
            conflict_details = []
            for conflict in conflicts:
                obj = conflict.get("object", "Unknown object")
                msg = conflict.get("value", "No details")
                conflict_details.append(f"❌ {obj}: {msg}")
            return "⚠️ Conflict(s) occurred:\n" + "\n".join(conflict_details)
        return "❌ Operation failed without detailed conflict info."

    summary = []
    if created:
        summary.append(f"✅ {created} item(s) created")
    if updated:
        summary.append(f"🔄 {updated} item(s) updated")
    if ignored:
        summary.append(f"⚠️ {ignored} item(s) ignored")

    return "\n".join(summary) if summary else "✅ Operation completed, but no changes reported."







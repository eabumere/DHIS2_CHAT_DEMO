# analytics_tools.py
import os
import requests
from dotenv import load_dotenv
# from faiss_search.search import fiass_query
from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain.tools import tool

try:
    from faiss_search.search import vectorstore
except:
    from agents.tools.faiss_search.search import vectorstore
# Load DHIS2 credentials from env
load_dotenv()

# Load environment variables

DHIS2_BASE_URL = os.getenv("DHIS2_BASE_URL")
DHIS2_USERNAME = os.getenv("DHIS2_USERNAME")
DHIS2_PASSWORD = os.getenv("DHIS2_PASSWORD")
# Define confidence threshold and convert to float
# FAISS_THRESHOLD = float(os.getenv("FAISS_THRESHOLD", "0.12"))
FAISS_THRESHOLD = 0.25  # TEMPORARY for testing

@tool
def query_analytics(
    indicators: list[str],
    periods: list[str],
    org_units: list[str],
    skip_meta: bool = False,
    display_property: str = "NAME",
    include_num_den: bool = True,
    skip_data: bool = False,
    output_id_scheme: str = "NAME"
) -> Dict[str, Any]:
    """Query analytics data from DHIS2."""
    # Args:
    #     indicators: List of indicator/data element UIDs (e.g., ["nFICjJluo74"])
    #     periods: List of period strings (e.g., ["202401", "LAST_12_MONTHS"])
    #     org_units: List of org unit UIDs (e.g., ["ImspTQPwCqd"])
    #     skip_meta: Skip metadata (default: False)
    #     display_property: Display name type (e.g., "NAME")
    #     include_num_den: Include numerator/denominator (default: True)
    #     skip_data: Skip data section (default: False)
    #     output_id_scheme: Output ID scheme (e.g., "NAME")


    indicator_string = ";".join(indicators)
    period_string = ";".join(periods)
    org_unit_string = ";".join(org_units)

    params = {
        "dimension": [
            f"dx:{indicator_string}",
            f"pe:{period_string}",
            f"ou:{org_unit_string}"
        ],
        "displayProperty": display_property,
        "includeNumDen": str(include_num_den).lower(),
        "skipMeta": str(skip_meta).lower(),
        "skipData": str(skip_data).lower(),
        "outputIdScheme": output_id_scheme
    }

    try:
        response = requests.get(
            f"{DHIS2_BASE_URL}/api/analytics",
            params=params,
            auth=(DHIS2_USERNAME, DHIS2_PASSWORD)
        )
        response.raise_for_status()
        return {
            "url": response.url,
            "data": response.json()
        }
    except Exception as e:
        return {"error": str(e)}



@tool
def search_metadata(query: str) -> Dict[str, Any]:
    """
    Searches metadata using vector similarity based on the query string.
    Returns either a single match or a list of high-confidence options for user selection.
    """
    try:
        docs_and_scores: List[Document] = vectorstore.similarity_search_with_score(query, k=5)
        filtered_matches = []

        for doc, score in docs_and_scores:
            if score <= FAISS_THRESHOLD:
                metadata = doc.metadata
                filtered_matches.append({
                    "name": metadata.get("name", ""),
                    "id": metadata.get("id", ""),
                    "doc_type": metadata.get("type", "Unknown"),
                    "score": float(score)
                })

        if not filtered_matches:
            return {
                "status": "no_match",
                "message": f"No indicator match found with score ≤ {FAISS_THRESHOLD}.",
                "suggestions": []
            }

        if len(filtered_matches) == 1:
            return {
                "status": "auto_selected",
                "selected": filtered_matches[0]
            }

        return {
            "status": "multiple_matches",
            "suggestions": filtered_matches
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_all(
    endpoint: str,
    key: str,
    fields: str = "*",
    filters: Optional[Dict[str, str]] = None,
    page_size: int = 600
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
#
#
# @tool
# def get_data_elements() -> List[Dict[str, Any]]:
#     """Fetch all DHIS2 data elements."""
#     return get_all("dataElements.json", "dataElements")
#
#
# @tool
# def get_indicators() -> List[Dict[str, Any]]:
#     """Fetch all DHIS2 indicators."""
#     return get_all("indicators.json", "indicators")
#
#
# @tool
# def get_program_indicators() -> List[Dict[str, Any]]:
#     """Fetch all DHIS2 program indicators."""
#     return get_all("programIndicators.json", "programIndicators")
#
#


@tool
def compute_total(values: List[Union[str, float, int]]) -> Union[float, Any]:
    """
    Compute the total sum of a list of values. Handles string numbers too.
    """
    try:
        return sum(float(v) for v in values if v is not None and str(v).strip() != "")
    except Exception as e:
        return f"Error computing total: {e}"

@tool
def compute_average(values: List[Union[str, float, int]]) -> Union[float, Any]:
    """
    Compute the average (mean) of a list of values.
    Ignores null/blank values and handles strings.
    """
    try:
        numeric_values = [float(v) for v in values if v is not None and str(v).strip() != ""]
        if not numeric_values:
            return 0.0
        return sum(numeric_values) / len(numeric_values)
    except Exception as e:
        return f"Error computing average: {e}"

@tool
def compute_max(values: List[Union[str, float, int]]) -> Union[float, Any]:
    """
    Compute the maximum value in a list.
    Handles string numbers and ignores blanks/nulls.
    """
    try:
        numeric_values = [float(v) for v in values if v is not None and str(v).strip() != ""]
        return max(numeric_values) if numeric_values else 0.0
    except Exception as e:
        return f"Error computing max: {e}"

@tool
def compute_min(values: List[Union[str, float, int]]) -> Union[float, Any]:
    """
    Compute the minimum value in a list.
    Handles string numbers and ignores blanks/nulls.
    """
    try:
        numeric_values = [float(v) for v in values if v is not None and str(v).strip() != ""]
        return min(numeric_values) if numeric_values else 0.0
    except Exception as e:
        return f"Error computing min: {e}"

@tool
def get_organisation_units(filters: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    Fetch DHIS2 organisation units with optional filtering.

    Args:
        filters: Optional dict of DHIS2 filter expressions.
                 Example: {"level": "eq:2", "name": "ilike:Sierra"}
    """
    return get_all(
        endpoint="organisationUnits.json",
        key="organisationUnits",
        fields="id,name,level",
        # fields="id,code,name,parent,children,level,path,ancestors",
        filters=filters
    )

# Suggest metadata
# results = search_metadata.invoke({"query": "maternal mortality", "doc_type": "indicator"})
# results = search_metadata.invoke({"query": "maternal mortality"})

# print(results)

# Get level 2 org units
# units = get_organisation_units.invoke({"level": "eq:2"})
# print(units)

# # Get org units whose name includes "District"
# units = get_organisation_units({"name": "ilike:District"})
# print(units)
#
# # Get all children of a specific parent
# units = get_organisation_units({"parent.id": "ImspTQPwCqd"})
# print(units)


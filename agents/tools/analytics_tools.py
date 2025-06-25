# analytics_tools.py
from typing import Dict, Any
import requests
import os
from langchain.tools import tool
DHIS2_BASE_URL = os.getenv("DHIS2_BASE_URL")
DHIS2_USERNAME = os.getenv("DHIS2_USERNAME")
DHIS2_PASSWORD = os.getenv("DHIS2_PASSWORD")

@tool
def query_analytics(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Query DHIS2 analytics endpoint with the given payload.
    """
    url = f"{DHIS2_BASE_URL}/api/analytics"
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers, auth=(DHIS2_USERNAME, DHIS2_PASSWORD))
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@tool
def get_analytics_metadata() -> Dict[str, Any]:
    """
    Get metadata relevant for analytics (e.g., dimensions, indicators).
    """
    url = f"{DHIS2_BASE_URL}/api/metadata?assumeTrue=false"
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers, auth=(DHIS2_USERNAME, DHIS2_PASSWORD))
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

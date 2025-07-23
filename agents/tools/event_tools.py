# agents/tools/event_tools.py

from langchain.tools import tool
import requests
import os
from dotenv import load_dotenv

load_dotenv()

DHIS2_BASE_URL = os.getenv("DHIS2_BASE_URL")
DHIS2_USERNAME = os.getenv("DHIS2_USERNAME")
DHIS2_PASSWORD = os.getenv("DHIS2_PASSWORD")

auth = (DHIS2_USERNAME, DHIS2_PASSWORD)


@tool
def record_event_data(payload: dict) -> dict:
    """
    Records data for a single event-based program.
    Payload must include: program, orgUnit, eventDate, dataValues, etc.
    """
    url = f"{DHIS2_BASE_URL}/api/events"
    response = requests.post(url, json=payload, auth=auth)
    return response.json()

# agents/tools/tracker_tools.py

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
def register_tracked_entity(payload: dict) -> dict:
    """
    Registers a new tracked entity in DHIS2.
    Payload must include: orgUnit, trackedEntityType, attributes, enrollments, etc.
    """
    url = f"{DHIS2_BASE_URL}/api/trackedEntityInstances"
    response = requests.post(url, json=payload, auth=auth)
    return response.json()


@tool
def record_followup_visit(payload: dict) -> dict:
    """
    Records a follow-up event for an existing tracked entity.
    Payload must include event, trackedEntityInstance, programStage, orgUnit, eventDate, dataValues, etc.
    """
    url = f"{DHIS2_BASE_URL}/api/events"
    response = requests.post(url, json=payload, auth=auth)
    return response.json()

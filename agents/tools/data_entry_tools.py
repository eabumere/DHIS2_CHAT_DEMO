# agents/tools/data_entry_tools.py

from langchain.tools import tool
from pydantic import BaseModel, Field

import requests
import os
from dotenv import load_dotenv

load_dotenv()

DHIS2_BASE_URL = os.getenv("DHIS2_BASE_URL")
DHIS2_USERNAME = os.getenv("DHIS2_USERNAME")
DHIS2_PASSWORD = os.getenv("DHIS2_PASSWORD")

auth = (DHIS2_USERNAME, DHIS2_PASSWORD)


class SubmitPayload(BaseModel):
    dataValues: list = Field(..., description="List of data values with dataElement, period, orgUnit, categoryOptionCombo, attributeOptionCombo, value")

@tool(args_schema=SubmitPayload)
def submit_aggregate_data(dataValues: list) -> dict:
    """
    Submits aggregate data values to DHIS2.
    """
    payload = {"dataValues": dataValues}
    url = f"{DHIS2_BASE_URL}/api/dataValueSets"
    response = requests.post(url, json=payload, auth=auth)
    return response.json()

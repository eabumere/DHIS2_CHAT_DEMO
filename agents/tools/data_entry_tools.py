# agents/tools/data_entry_tools.py

import os
from dotenv import load_dotenv
from langchain.tools import tool
import requests
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd
import streamlit as st


load_dotenv()
DHIS2_BASE_URL = os.getenv("DHIS2_BASE_URL")
DHIS2_USERNAME = os.getenv("DHIS2_USERNAME")
DHIS2_PASSWORD = os.getenv("DHIS2_PASSWORD")

auth = (DHIS2_USERNAME, DHIS2_PASSWORD)


class DataValue(BaseModel):
    dataElement: str
    period: str
    orgUnit: str
    categoryOptionCombo: str
    attributeOptionCombo: str
    value: str


class SubmitPayload(BaseModel):
    data_values: Optional[List[DataValue]] = Field(
        None, description="List of DHIS2 data value objects."
    )
    preview_only: Optional[bool] = Field(
        False, description="If true, only previews the payload without submitting."
    )
    column_mapping: Optional[Dict[str, str]] = Field(
        None,
        description="Mapping of uploaded column names to DHIS2 required field names. Keys are uploaded names, values are DHIS2 fields."
    )
    params: str = Field(
        ...,  # <-- means "required field"
        description='Operation strategy: must be "CREATE_AND_UPDATE" or "DELETE".'
    )


@tool(args_schema=SubmitPayload)
def submit_aggregate_data(
        data_values: Optional[List[DataValue]] = None,
        preview_only: bool = False,
        column_mapping: Optional[Dict[str, str]] = None,
        params: str = None,  # now required
) -> dict:
    """
    Submits or deletes DHIS2 aggregate data values.

    - If preview_only is True, returns the payload without submitting.
    - If data_values is not provided, constructs the payload from the session dataframe and column_mapping.
    - The `params` argument determines the action: CREATE_AND_UPDATE (default) or DELETE.
    """
    required_cols = [
        "dataElement",
        "period",
        "orgUnit",
        "categoryOptionCombo",
        "attributeOptionCombo",
        "value"
    ]

    # Build payload from dataframe + column_mapping if no explicit data_values
    if not data_values and not column_mapping:
        return {
            "error": "No data_values or column_mapping provided for submission. Please provide either data_values directly, or column_mapping to extract from uploaded dataframe."
        }

    if column_mapping:
        df = st.session_state.get("raw_data_df_uploaded")

        if df is None or not isinstance(df, pd.DataFrame):
            return {"error": "No uploaded dataframe found in session."}

        # Normalize and rename columns
        df.columns = [col.strip().lower() for col in df.columns]
        renaming_map = {k.strip().lower(): v for k, v in column_mapping.items()}
        df.rename(columns=renaming_map, inplace=True)

        if not all(col in df.columns for col in required_cols):
            return {
                "error": "Missing one or more required columns after applying mapping.",
                "columns_found": list(df.columns),
                "required_columns": required_cols,
                "column_mapping_used": renaming_map,
            }

        payload = {"dataValues": df[required_cols].to_dict(orient="records")}

    elif data_values:
        payload = {"dataValues": [dv.model_dump() for dv in data_values]}

    else:
        return {"error": "No data_values or column_mapping provided for submission."}

    if params.upper() != "DELETE":
        if preview_only:
            return {
                "preview_payload": payload,
                "operation": params
            }

    # Determine endpoint and method

    if not preview_only:
        try:
            print(f"Received params: {params}")
            print(isinstance(params, str))

            if params.upper() == "DELETE":
                import_strategy = {"importStrategy": "DELETE"}
            else:
                import_strategy = {"importStrategy": "CREATE_AND_UPDATE"}

            url = f"{DHIS2_BASE_URL}/api/dataValueSets"
            response = requests.post(url, json=payload, auth=auth, params=import_strategy)
            response.raise_for_status()
            return {"data": response.json()}

        except requests.RequestException as e:
            return {
                "error": "DHIS2 operation failed.",
                "operation": params,
                "status_code": response.status_code if 'response' in locals() else None,
                "details": str(e),
                "response_text": response.text if 'response' in locals() else None,
            }


class ColumnMappingInput(BaseModel):
    columns: List[str]


@tool(args_schema=ColumnMappingInput)
def suggest_column_mapping(columns: List[str]) -> Dict[str, str]:
    """
    Suggests a mapping from uploaded column names to DHIS2 required fields.
    Uses intelligent matching (case, substring) to handle near-matches.
    """

    required_cols = list(DataValue.__annotations__.keys())
    suggestions = {}

    for required_col in required_cols:
        best_match = max(
            columns,
            key=lambda c: int(required_col.lower() == c.lower()) * 2 + int(required_col.lower() in c.lower()),
            default=None
        )
        if best_match:
            suggestions[best_match] = required_col

    return suggestions

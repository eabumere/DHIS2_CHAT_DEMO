# agents/tools/data_entry_tools.py

import os
from dotenv import load_dotenv
from langchain.tools import tool
import requests
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd
import streamlit as st
from fuzzysearch import find_near_matches


load_dotenv()
DHIS2_BASE_URL = os.getenv("DHIS2_BASE_URL")
DHIS2_USERNAME = os.getenv("DHIS2_USERNAME")
DHIS2_PASSWORD = os.getenv("DHIS2_PASSWORD")

auth = (DHIS2_USERNAME, DHIS2_PASSWORD)
if "suggested_columns_data_entry" not in st.session_state:
    st.session_state.suggested_columns_data_entry = {}

class DataValue(BaseModel):
    dataElement: str
    period: str
    orgUnit: str
    categoryOptionCombo: str
    attributeOptionCombo: str
    value: str


class SubmitPayload(BaseModel):
    preview_only: Optional[bool] = Field(
        False, description="If true, only previews the payload without submitting."
    )
    column_mapping: Dict[str, str] = Field(
        None,  # Required now
        description="Mapping of uploaded column names to DHIS2 required field names. Keys are uploaded names, values are DHIS2 fields."
    )
    params: str = Field(
        ..., description="DHIS2 operation type: 'CREATE_AND_UPDATE' or 'DELETE'."
    )

def manipulate_data(df: pd.DataFrame, operation: str, match_criteria: dict, update_values: dict = None) -> pd.DataFrame:
    if operation == "UPDATE":
        mask = pd.Series(True, index=df.index)
        for key, value in match_criteria.items():
            mask &= df[key] == value
        df.loc[mask, list(update_values.keys())] = pd.Series(update_values)
        return df

    elif operation == "DELETE":
        mask = pd.Series(True, index=df.index)
        for key, value in match_criteria.items():
            mask &= df[key] == value
        return df[~mask]

    elif operation == "CREATE":
        new_row = {**match_criteria, **(update_values or {})}
        return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    else:
        raise ValueError(f"Unsupported operation: {operation}")


@tool(args_schema=SubmitPayload)
def submit_aggregate_data(
    preview_only: bool = False,
    column_mapping: Dict[str, str] = None,
    params: str = None,
) -> dict:
    """
    Submits or deletes DHIS2 aggregate data values.

    - Builds the payload from the uploaded dataframe in session using column_mapping.
    - `params` defines the operation: 'CREATE_AND_UPDATE' or 'DELETE'.
    - If preview_only is True, returns the payload without submitting.
    """
    df = st.session_state.get("raw_data_df_uploaded")
    if df is None or not isinstance(df, pd.DataFrame):
        return {"error": "No uploaded dataframe found in session."}

    # Use fallback if column_mapping not provided
    if not column_mapping:
        column_mapping = st.session_state.get("suggested_columns_data_entry")
        if not column_mapping:
            return {"error": "No column_mapping provided and no mapping found in session_state.suggested_columns_data_entry."}

    # Normalize and rename
    df.columns = [col.strip().lower() for col in df.columns]
    renaming_map = {k.strip().lower(): v for k, v in column_mapping.items()}
    df.rename(columns=renaming_map, inplace=True)

    required_cols = ["dataElement", "period", "orgUnit", "categoryOptionCombo", "attributeOptionCombo", "value"]
    if not all(col in df.columns for col in required_cols):
        return {
            "error": "Missing one or more required columns after applying mapping.",
            "columns_found": list(df.columns),
            "required_columns": required_cols,
            "column_mapping_used": renaming_map,
        }

    payload = {"dataValues": df[required_cols].to_dict(orient="records")}

    if preview_only:
        return {"preview_payload": payload, "operation": params}

    try:
        if params.upper() == "DELETE":
            import_strategy = {"importStrategy": "DELETE"}
        else:
            import_strategy = {"importStrategy": "CREATE_AND_UPDATE"}

        url = f"{DHIS2_BASE_URL}/api/dataValueSets"
        response = requests.post(url, json=payload, auth=auth, params=import_strategy)
        response.raise_for_status()

        return {"data": response.json(), "operation": params}

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


from fuzzysearch import find_near_matches

@tool(args_schema=ColumnMappingInput)
def suggest_column_mapping(columns: List[str]) -> Dict[str, str]:
    """
    Suggests a mapping from uploaded column names to DHIS2 required fields.
    Uses fuzzy matching to handle typos and approximate matches.
    """

    required_cols = list(DataValue.__annotations__.keys())
    suggestions = {}
    print("+++++ columns: List[str] +++++")
    print(columns)
    for required_col in required_cols:
        best_match = None
        lowest_distance = float('inf')

        for col in columns:
            # Try to find near matches of required_col inside each column name
            matches = find_near_matches(required_col.lower(), col.lower(), max_l_dist=2)

            if matches:
                match = matches[0]  # Take the first match
                distance = match.dist

                if distance < lowest_distance:
                    best_match = col
                    lowest_distance = distance

        if best_match:
            suggestions[best_match] = required_col

    # Store for review or later use
    print("+++++ suggestions +++++")
    print(suggestions)
    st.session_state.pdf_text = suggestions
    return suggestions


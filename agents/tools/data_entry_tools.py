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
# from .faiss_search.search import hybrid_search
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from io import StringIO
import re
from typing import Dict, Any
import asyncio
from utils.llm import get_llm
FAISS_THRESHOLD = 0.2
# FAISS_THRESHOLD = float(os.getenv("FAISS_THRESHOLD", 0.5))

llm = get_llm()
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


def ensure_dataframe(obj):
    # Already a DataFrame
    if isinstance(obj, pd.DataFrame):
        return obj

    # List of dicts
    if isinstance(obj, list) and all(isinstance(row, dict) for row in obj):
        return pd.DataFrame(obj)

    # String output
    if isinstance(obj, str):
        text = obj.strip()

        # 1️⃣ Try JSON first
        if text.startswith("{") or text.startswith("["):
            try:
                return pd.read_json(StringIO(text))
            except Exception:
                pass

        # 2️⃣ Remove leading sentences before any table-like pattern
        patterns = [
            r"(\n\s*\|.*\|)",        # Markdown header
            r"(\n\s*\d+\s*\|.*\|)",  # Numbered row start
            r"(^dataElement\s*\|)",  # Column header with |
        ]
        start_index = None
        for pat in patterns:
            m = re.search(pat, text, flags=re.MULTILINE)
            if m:
                start_index = m.start()
                break
        if start_index is not None:
            text = text[start_index:]

        # 3️⃣ Read as pipe-delimited table
        try:
            df = pd.read_csv(StringIO(text), sep="|", engine="python")
            df = df.dropna(axis=1, how="all")  # drop empty cols

            # Strip spaces from column names
            df.columns = [c.strip() for c in df.columns]

            # Drop unnamed index columns safely
            unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
            df = df.drop(columns=unnamed_cols)

            # Drop rows that are actually markdown separator lines
            df = df[~df.iloc[:, 0].astype(str).str.contains("---", na=False)]

            df = df.reset_index(drop=True)
            df = df.reset_index(drop=True)

            # 🚿 Clean all whitespace
            df.columns = [c.strip() for c in df.columns]
            df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

            if df.empty:
                raise ValueError("Parsed DataFrame is empty")

            return df

        except Exception as e:
            raise ValueError(f"Failed to parse table from string: {e}")

    raise TypeError(f"Unsupported type for DataFrame conversion: {type(obj)}")


def apply_structured_instruction(df: pd.DataFrame, instruction: Dict, params: str) -> pd.DataFrame:
    match_criteria = instruction.get("match_criteria", {})
    update_values = instruction.get("update_values", {})

    print("+++ match_criteria +++")
    print(match_criteria)

    print("+++ update_values +++")
    print(update_values)

    print("++++ instruction ++++")
    print(instruction)

    print("++++ params ++++")
    print(params)


    # If no match criteria, do nothing
    if not match_criteria:
        return df

    # Create a mask to filter matching rows
    mask = pd.Series([True] * len(df))
    for col, val in match_criteria.items():
        if col not in df.columns:
            raise ValueError(f"Match column '{col}' not found in DataFrame")
        mask &= df[col] == val

    # Handle DELETE operation: return only rows to be deleted
    if params == "DELETE":
        return df[mask].copy()

    # Handle CREATE_AND_UPDATE: apply updates and return full DataFrame
    if params == "CREATE_AND_UPDATE":
        if not update_values:
            return df  # Nothing to update
        for update_col, update_val in update_values.items():
            if update_col not in df.columns:
                raise ValueError(f"Update column '{update_col}' not found in DataFrame")
            df.loc[mask, update_col] = update_val
        return df

    # Fallback: return unchanged if unknown params
    return df


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
    structured_instruction: Dict[str, Any] = Field(
        None,
        description=(
            "Structured JSON instructions to match and update data rows. "
            "Example: {"
            "  'operation': 'UPDATE', "
            "  'match_criteria': {...}, "
            "  'update_values': {...}, "
            "  'specific_rows': [0, 2, 5] "
            "}"
        )
    )
    instruction_prompt: str = Field(
        ...,
        description=(
            "Natural language prompt describing the intended dataframe operation. "
            "Used by the LLM to interpret user intent and validate or complement the structured_instruction."
        )
    )




@tool
def submit_aggregate_data_from_text(
    parsed_payload: Dict[str, Any],
    preview_only: bool = False,
    operation: str = "SUBMIT"
) -> Dict[str, Any]:
    """
    Submit, update, or delete aggregate data in DHIS2 from a parsed natural language payload.

    Args:
        parsed_payload (dict): Must include resolved IDs for:
            {{
              "dataElement": "<RESOLVED ID>",
              "orgUnit": "<RESOLVED ID>",
              "period": "YYYYMM",
              "categoryOptionCombos": "<RESOLVED ID>",
              "attributeOptionCombos": "<RESOLVED ID>",
              "value": number
            }}
        preview_only (bool): If True, preview the parsed data without committing.
        params: Operation type: 'CREATE_AND_UPDATE' or 'DELETE'.
        operation (str): One of ["SUBMIT", "UPDATE", "DELETE"].

    Returns:
        dict: Preview or final confirmation with resolved metadata.
    """
    if operation == "DELETE":
        import_strategy = {"importStrategy": "DELETE"}
    else:
        import_strategy = {"importStrategy": "CREATE_AND_UPDATE"}
    print(f"submit_aggregate_data_from_text received => {parsed_payload}")
    print(f"params received => {import_strategy}")

    try:
        # --- Validate required fields ---
        required_fields = ["dataElement", "orgUnit", "period", "categoryOptionCombos", "attributeOptionCombos", "value"]
        for field in required_fields:
            if field not in parsed_payload or not parsed_payload[field]:
                return {"status": "error", "message": f"Missing required field: {field}"}

        if parsed_payload["categoryOptionCombos"] == parsed_payload["attributeOptionCombos"]:
            return {
                "status": "error",
                "message": "CategoryOptionCombos and AttributeOptionCombos resolved to the same UID. Please review your input to ensure they are distinct."
            }
        # --- Build response ---
        final_payload = {
            "operation": operation,
            "preview_only": preview_only,
            "parsed_payload": parsed_payload
        }



        if preview_only:
            return {
                "status": "PREVIEW",
                "message": "This is a preview of the data to be submitted.",
                "result": final_payload
            }
        # Prepare data payload
        # Extract relevant records
        # ✅ Build DHIS2 dataValues payload
        print(f"final_payload => {final_payload}")
        payload = {
            "dataValues": [
                {
                    "dataElement": parsed_payload["dataElement"],
                    "orgUnit": parsed_payload["orgUnit"],
                    "period": parsed_payload["period"],
                    "categoryOptionCombo": parsed_payload["categoryOptionCombos"],
                    "attributeOptionCombo": parsed_payload["attributeOptionCombos"],
                    "value": parsed_payload["value"],
                }
            ]
        }


        print("+++ payload +++")
        print(payload)

        # Send request to DHIS2 API
        try:
            posted_data = post_data(import_strategy, payload)
            print(f"posted_data => {posted_data}")
            return posted_data

        except Exception as e:
            return {
                "error": "DHIS2 operation failed.",
                "operation": operation,
                "details": str(e),
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}




@tool(args_schema=SubmitPayload)
def submit_aggregate_data(
    preview_only: bool = False,
    column_mapping: Optional[Dict[str, str]] = None,
    params: Optional[str] = None,
    structured_instruction: Optional[Dict[str, Any]] = None,
    instruction_prompt: Optional[str] = None,
) -> dict:
    """
    Submits or deletes DHIS2 aggregate data values.

    - Builds the payload from the uploaded dataframe in session using column_mapping.
    - If structured_instruction is provided, applies filtering/updating before submission.
    - `params` defines the operation: 'CREATE_AND_UPDATE' or 'DELETE'.
    - If preview_only is True, returns the payload without submitting.

    Args:
        preview_only: If True, only return payload without submission.
        column_mapping: Mapping from uploaded column names to DHIS2 expected columns.
        params: Operation type: 'CREATE_AND_UPDATE' or 'DELETE'.
        structured_instruction: Predefined structured instructions for filtering/updating.
        instruction_prompt: Freeform natural language instruction to be executed via agent.
    """
    required_cols = ["dataElement", "period", "orgUnit", "categoryOptionCombo", "attributeOptionCombo", "value"]
    master_df = st.session_state.get("raw_data_df_uploaded")
    if master_df is None or not isinstance(master_df, pd.DataFrame):
        return {"error": "No master uploaded dataframe found in session."}

    df = master_df.copy()

    # Use fallback if column_mapping not provided
    if not column_mapping:
        column_mapping = st.session_state.get("suggested_columns_data_entry")
        if not column_mapping:
            return {"error": "No column_mapping provided and no mapping found in session_state.suggested_columns_data_entry."}

    # Normalize columns and rename per mapping
    df.columns = [col.strip().lower() for col in df.columns]
    renaming_map = {k.strip().lower(): v for k, v in column_mapping.items()}
    df.rename(columns=renaming_map, inplace=True)
    original_df = df.copy()
    operation = params.upper()  # e.g., "DELETE", "CREATE_AND_UPDATE"
    to_delete_rows = pd.DataFrame()
    dataframe_emptied = False
    if instruction_prompt:
        try:
            agent = create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True)
            print(instruction_prompt)
            result = agent.invoke({"input": f"{instruction_prompt} return the updated dataframe"})
            # Use invoke() instead of run() — invoke returns a dict with outputs, including python execution results
            # Check if it's nested
            if isinstance(result.get("output"), dict) and "python_repl" in result["output"]:
                processed_df = result["output"]["python_repl"]
            else:
                processed_df = result.get("output")

            print("Updated DataFrame:", processed_df)
            print("Type:", type(processed_df))

            # If it's a string that looks like JSON or table
            if isinstance(processed_df, str):
                stripped = processed_df.strip()
                # Special case: "dataframe is empty" message
                if "dataframe is now empty" in stripped.lower() or "dataframe is empty" in stripped.lower():
                    dataframe_emptied = True  # Return empty
                else:
                    df = ensure_dataframe(processed_df)
            if not dataframe_emptied:
                if operation == "DELETE":
                    to_delete_rows = original_df.copy()
            else:
                df = df[required_cols]
                original_df = original_df[required_cols]
                df['period'] = df['period'].astype(str).str.strip()
                original_df['period'] = original_df['period'].astype(str).str.strip()
                df['value'] = df['value'].astype(str).str.strip()
                original_df['value'] = original_df['value'].astype(str).str.strip()
                if operation == "DELETE":

                    to_delete_rows = original_df.merge(df, indicator=True, how="outer") \
                        .query('_merge == "left_only"') \
                        .drop('_merge', axis=1)
                    print("target_rows:", to_delete_rows)

        except Exception as e:
            if "out-of-bounds" in str(e):
                # Extract relevant records
                if operation == "DELETE" and 'all' in instruction_prompt.lower():
                    to_delete_rows = original_df.copy()
            else:
                return {"error": "Failed to apply instruction via LangChain agent.", "details": str(e)}
    elif structured_instruction:
        print("++++ Cleaned after Structured Instruction ++++")
        df = apply_structured_instruction(df, structured_instruction, params)
        print(df)

        # For DELETE operation with structured_instruction,
        # we only send the matched rows to delete
        if structured_instruction.get("operation", "").upper() == "DELETE":
            # For delete, params should be deleted explicitly
            params = "DELETE"



    if not all(col in df.columns for col in required_cols):
        return {
            "error": "Missing one or more required columns after applying mapping and instructions.",
            "columns_found": list(df.columns),
            "required_columns": required_cols,
            "column_mapping_used": renaming_map,
        }

    # Prepare data payload
    # Extract relevant records
    if operation == "DELETE":
        payload = {"dataValues": to_delete_rows[required_cols].to_dict(orient="records")}
    else:
        payload = {"dataValues": df[required_cols].to_dict(orient="records")}

    print("+++ payload +++")
    print(payload)

    if preview_only:
        return {"preview_payload": payload, "operation": params}
    # Send request to DHIS2 API
    try:
        import_strategy = {"importStrategy": "CREATE_AND_UPDATE"}
        if params and params.upper() == "DELETE":
            import_strategy = {"importStrategy": "DELETE"}
        posted_data = post_data(import_strategy, payload)
        print(f"posted_data => {posted_data}")
        return posted_data
    except Exception as e:
        return {
            "error": "DHIS2 operation failed.",
            "params": params,
            "details": str(e)
        }

def post_data(import_strategy, payload):
    response = {}
    try:
        url = f"{DHIS2_BASE_URL}/api/dataValueSets"
        response = requests.post(url, json=payload, auth=auth, params=import_strategy)
        response.raise_for_status()
        print({"data": response.json(), "operation": import_strategy["importStrategy"]})
        return {"data": response.json(), "operation": import_strategy["importStrategy"]}
    except requests.RequestException as e:
        return {
            "error": "DHIS2 operation failed.",
            "status_code": response.status_code if 'response' in locals() else None,
            "details": str(e),
            "response_text": response.text if 'response' in locals() else None
        }

class ColumnMappingInput(BaseModel):
    columns: List[str]




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


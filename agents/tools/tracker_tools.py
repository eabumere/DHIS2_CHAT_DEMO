# agents/tools/tracker_tools.py
from azure.core.credentials import AzureKeyCredential
from PyPDF2 import PdfReader, PdfWriter
from collections import defaultdict
import json
from typing import Dict, Any
from datetime import datetime, timedelta, timezone
from langchain.tools import tool
import requests
from azure.storage.blob import BlobClient, BlobServiceClient, BlobSasPermissions, generate_blob_sas
import os
import re
import streamlit as st
from dotenv import load_dotenv
import tempfile
from azure.ai.documentintelligence import DocumentIntelligenceClient


load_dotenv()

DHIS2_BASE_URL = os.getenv("DHIS2_BASE_URL")
DHIS2_USERNAME = os.getenv("DHIS2_USERNAME")
DHIS2_PASSWORD = os.getenv("DHIS2_PASSWORD")

auth = (DHIS2_USERNAME, DHIS2_PASSWORD)

# Azure configs
# Azure configs
endpoint = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
key = os.getenv("DOC_INTELLIGENCE_KEY")
model_id = os.getenv("MODEL_ID")

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")
blob_name = "Sample v2 combined.pdf"   # 👈 your blob filename

AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
# Parse connection string to get account key
match = re.search(r"AccountKey=([^;]+)", AZURE_STORAGE_CONNECTION_STRING)
AZURE_STORAGE_ACCOUNT_KEY = match.group(1)



# Blob client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
# Init clients
client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
blob = BlobClient.from_connection_string(
    conn_str=AZURE_STORAGE_CONNECTION_STRING,
    container_name=AZURE_STORAGE_CONTAINER,
    blob_name=blob_name,
)

# --- Dynamic conversion like your original code ---
def process_row(row):
    obj = row['valueObject']
    row_data = {}
    for this_key, value in obj.items():
        row_data[this_key] = {
            "value": value.get('valueString', ""),
            "confidence": value.get('confidence', None)
        }
    return row_data

@tool
def process_scanned_register(pdf_file: str) -> str:
    """
    Processes a local scanned PDF register (given as a file path).
    Splits into pages, extracts table data, merges patient records.
    Returns dict of patient records keyed by ART No Patient ID.
    """

    temp_path = None
    print(pdf_file)

    if "pdf_file" in st.session_state:

        print(st.session_state.pdf_file)
        temp_path = st.session_state.pdf_file
    else:
        st.warning("⚠️ No PDF stored in session yet.")

    clean_data = []
    reader = PdfReader(temp_path)

    if temp_path is not None:
        for i, page in enumerate(reader.pages, start=1):
            writer = PdfWriter()
            writer.add_page(page)
            single_page_path = f"page_{i}.pdf"
            with open(single_page_path, "wb") as f_out:
                writer.write(f_out)
            try:
                with open(single_page_path, "rb") as fd:
                    poller = client.begin_analyze_document(
                        model_id=model_id,
                        body=fd,
                        content_type="application/pdf"
                    )
                    result = poller.result()  # <-- this can take time
                    print(f"--- Page {i} --- Analysis complete")
            except Exception as e:
                print(f"❌ Error analyzing page {i}: {e}")
            finally:
                # Delete the temp single-page PDF
                if os.path.exists(single_page_path):
                    os.unlink(single_page_path)
                    print(f"Deleted temporary file: {single_page_path}")

            for idx, doc in enumerate(result.documents):
                for name, field in doc.fields.items():
                    if "TableData" in name:
                        processing_block = []
                        # Make sure field is not None and has a valueArray
                        if field and field.get("valueArray"):
                            for row in field["valueArray"]:
                                processing_block.append(process_row(row))
                        else:
                            print(f"⚠️ No valueArray found in {name}")
                        if len(processing_block)>0:
                            clean_data.append([processing_block])

        # --- Merge logic ---
        merged = defaultdict(dict)
        # data = json.dumps(clean_data, indent=2)
        for page in clean_data:              # page = [[record1, record2, ...]]
            for group in page:               # group = [record1, record2, record3]
                for record in group:         # record = dict ✅
                    patient_id = record.get("ART No Patient ID:", {}).get("value")
                    if not patient_id:
                        continue

                    for k, v in record.items():
                        if k != "ART No Patient ID:":
                            merged[patient_id][k] = v

                    # Always include ART No Patient ID
                    merged[patient_id]["ART No Patient ID:"] = record["ART No Patient ID:"]

        final_rows = list(merged.values())
        response_obj = (json.dumps(final_rows, indent=2))
        return response_obj
    else:
        return "Document not processed"


def get_file_bytes(file):
    """
    Ensure file is in bytes.
    If already bytes -> return as is.
    """
    if isinstance(file, bytes):
        return file
    else:
        raise ValueError("Expected raw bytes. Pass file content using open(..., 'rb').read().")


# -----------------------------
# Upload document + return loader docs
# -----------------------------
@tool
def upload_to_blob_storage(file: bytes, filename: str = "uploaded.bin") -> Dict[str, Any]:
    """
    Upload a document (image or file) to Azure Blob Storage,
    then return a short-lived SAS URL for secure access.

    Args:
        file (bytes): Raw file bytes
        filename (str): Original filename (used to infer extension)
    """
    try:
        file_bytes = get_file_bytes(file)

        # Create a unique blob name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = os.path.basename(filename).replace(" ", "_")
        blob_name = f"facility_register_{timestamp}_{safe_filename}"

        # Upload to Azure Blob Storage
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_STORAGE_CONTAINER,
            blob=blob_name
        )
        blob_client.upload_blob(file_bytes, overwrite=True)

        # Generate short-lived SAS URL (15 min expiry)
        expiry_time = datetime.now(timezone.utc) + timedelta(minutes=15)
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=AZURE_STORAGE_CONTAINER,
            blob_name=blob_name,
            permission=BlobSasPermissions(read=True),
            expiry=expiry_time,
            account_key=AZURE_STORAGE_ACCOUNT_KEY
        )

        blob_url = f"{blob_client.url}?{sas_token}"

        return {
            "success": True,
            "blob_url": blob_url,
            "blob_name": blob_name
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

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

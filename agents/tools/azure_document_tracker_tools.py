#azure_document_tools.py
from langchain.tools import tool
import os
import io
import requests as rq
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from typing import Dict, Any
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat, AnalyzeResult
from datetime import datetime, timedelta, timezone
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
# from azure.ai.resources.client import AIClient
import mimetypes
import re
import os
from langchain_community.document_loaders import (
    AzureBlobStorageFileLoader,
    AzureBlobStorageContainerLoader,
    AzureAIDataLoader,
    AzureAIDocumentIntelligenceLoader,
)

# -----------------------------
# Azure Config
# -----------------------------
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "dhis2-documents")
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
# Parse connection string to get account key
match = re.search(r"AccountKey=([^;]+)", AZURE_STORAGE_CONNECTION_STRING)
AZURE_STORAGE_ACCOUNT_KEY = match.group(1)

# Blob client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

# Azure AI Client (auth with DefaultAzureCredential)
# ai_client = AIClient.from_connection_string(
#     credential=DefaultAzureCredential(),
#     endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
# )


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

# -----------------------------
# Analyze with Layout Model
# -----------------------------
@tool
def analyze_with_layout_model(blob_url: str, limit: int, model: str) -> Dict[str, Any]:
    """
    Analyze a PDF/Document from Azure Blob URL using prebuilt layout model
    and return Markdown content.
    """
    try:
        # Initialize client
        client = DocumentIntelligenceClient(
            endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
            credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_KEY)
        )

        # Download file from blob URL
        resp = rq.get(blob_url)
        resp.raise_for_status()
        file_stream = resp.content  # bytes

        # Analyze using prebuilt-layout model with Markdown output
        poller = client.begin_analyze_document(
            model_id=model,
            body=file_stream,
            content_type="application/pdf",  # or image/jpeg, image/png
            output_content_format=DocumentContentFormat.MARKDOWN
        )
        result: AnalyzeResult = poller.result()
        return {
                "success": True,
                "content": result.content[:limit]
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
# -----------------------------
# Analyze with Custom Model
# -----------------------------
@tool
def analyze_with_custom_model(blob_url: str, model_id: str) -> Dict[str, Any]:
    """Analyze document using a custom model."""
    try:
        loader = AzureAIDocumentIntelligenceLoader(
            api_endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
            api_key=AZURE_DOCUMENT_INTELLIGENCE_KEY,
            file_path=blob_url,
            api_model=model_id,
        )
        docs = loader.load()

        extracted = []
        for doc in docs:
            extracted.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        return {"success": True, "model_id": model_id, "results": extracted}
    except Exception as e:
        return {"success": False, "error": str(e)}

# -----------------------------
# Map to DHIS2 metadata
# -----------------------------
@tool
def map_to_dhis2_metadata(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Map extracted fields to DHIS2 metadata."""
    try:
        mapping_config = {
            "dataElement": ["anc_visits", "malaria_cases", "immunization", "data_element"],
            "period": ["month", "year", "date", "period", "time"],
            "orgUnit": ["facility", "clinic", "hospital", "org_unit", "location"],
            "categoryOptionCombo": ["category", "disaggregation", "breakdown"],
            "attributeOptionCombo": ["attribute", "classification"],
            "value": ["count", "number", "total", "value", "amount"]
        }

        mapping_result = {}
        for item in extraction_result.get("results", []):
            text = item["content"].lower()
            for dhis2_field, synonyms in mapping_config.items():
                if any(s in text for s in synonyms):
                    mapping_result[dhis2_field] = {
                        "value": text,
                        "confidence": item["metadata"].get("confidence", 1.0)
                    }
                    break

        return {"success": True, "mapping": mapping_result}
    except Exception as e:
        return {"success": False, "error": str(e)}

# -----------------------------
# Validate Payload
# -----------------------------
@tool
def validate_dhis2_payload(mapping_result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate DHIS2 payload for required fields."""
    try:
        required = {"dataElement", "period", "orgUnit", "value"}
        mapped_fields = set(mapping_result.get("mapping", {}).keys())
        missing = required - mapped_fields

        return {"success": True, "missing_fields": list(missing)}
    except Exception as e:
        return {"success": False, "error": str(e)}

# -----------------------------
# Train Custom Model
# -----------------------------
# @tool
# def train_custom_model(model_id: str, container_sas_url: str) -> Dict[str, Any]:
#     """Train a custom model with Azure Document Intelligence."""
#     try:
#         poller = AIClient.document_intelligence.begin_build_document_model(
#             build_mode="template",
#             model_id=model_id,
#             blob_container_url=container_sas_url
#         )
#         model = poller.result()
#         return {
#             "success": True,
#             "model_id": model.model_id,
#             "status": model.status,
#             "created_at": model.created_date_time.isoformat() if model.created_date_time else None,
#         }
#     except Exception as e:
#         return {"success": False, "error": str(e)}
#
# # -----------------------------
# # Flow: Scan → Storage → Analysis → Mapping → Validation → Train
# # -----------------------------
# def pipeline_flow(blob_url: str, container_sas_url: str, model_id: str) -> Dict[str, Any]:
#     """End-to-end pipeline flow."""
#     results = {}
#
#     # 1. Analyze
#     layout = analyze_with_layout_model(blob_url)
#     results["layout"] = layout
#
#     # 2. Map to DHIS2
#     if layout.get("success"):
#         mapping = map_to_dhis2_metadata(layout)
#         results["mapping"] = mapping
#
#         # 3. Validate
#         validation = validate_dhis2_payload(mapping)
#         results["validation"] = validation
#
#         # 4. Train model (if layout approved)
#         if not validation.get("missing_fields"):
#             training = train_custom_model(model_id, container_sas_url)
#             results["training"] = training
#
#     return results

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from PyPDF2 import PdfReader, PdfWriter
from azure.storage.blob import BlobClient
import os
import json
import tempfile
from dotenv import load_dotenv
from collections import defaultdict
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

import json

load_dotenv()
DHIS2_BASE_URL = os.getenv("DHIS2_BASE_URL")
DHIS2_USERNAME = os.getenv("DHIS2_USERNAME")
DHIS2_PASSWORD = os.getenv("DHIS2_PASSWORD")
# Azure configs
endpoint = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
key = os.getenv("DOC_INTELLIGENCE_KEY")
model_id = os.getenv("MODEL_ID")

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")
blob_name = "Sample v2 combined OrgUnit.pdf"   # 👈 your blob filename

# Init clients
client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
blob = BlobClient.from_connection_string(
    conn_str=AZURE_STORAGE_CONNECTION_STRING,
    container_name=AZURE_STORAGE_CONTAINER,
    blob_name=blob_name,
)


def get_all(
    endpoint: str,
    key: str,
    fields: str = "id,name",
    filters: Optional[Dict[str, str]] = None,
    page_size: int = 1000
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

def generate_dhis2_ids(count: int = 1) -> List[str]:
    """Generate one or more unique DHIS2 IDs using the system's identifier endpoint."""
    if not all([DHIS2_BASE_URL, DHIS2_USERNAME, DHIS2_PASSWORD]):
        return []
    try:
        response = requests.get(
            f"{DHIS2_BASE_URL}/api/system/id",
            auth=(DHIS2_USERNAME, DHIS2_PASSWORD),
            params={"limit": count}
        )
        response.raise_for_status()
        return response.json().get("codes", [])
    except Exception as e:
        print(f"❌ Failed to generate ID(s): {str(e)}")
        return []



def map_to_dhis2_tracker(patients: list, org_unit: str, te_type: str, program: str) -> dict:
    """
    Map extracted patient records to DHIS2 tracker create payload.

    Args:
        patients (list): Extracted patient rows (with {value, confidence}).
        org_unit (str): DHIS2 orgUnit UID.
        te_type (str): DHIS2 trackedEntityType UID.
        program (str): DHIS2 program UID.

    Returns:
        dict: Tracker payload for DHIS2 v40.
    """
    tracked_entities = []

    for row in patients:
        tei = {
            "trackedEntityType": te_type,
            "trackedEntity": None,
            "orgUnit": org_unit,
            "attributes": [],
            "enrollments": []

        }
        attributes = []
        # Example mappings — you will need to adjust UIDs
        mappings = {
            "Patient ID: National ID": "AuPLng5hLbE",
            "Transfer: (in) From Date": "HwDGCdte3Ck",
            "Surname and Given name": "TfdH5KvFmMy",
            "DoB": "gHGyrwKPzej",
            "Sex (m/f)": "CklPZdOd6H1",
            "≥ 15 yrs": "NHviewDKFN6",
            "Transfer: (Out) From Date": "bLwiNONGPFF",
            "<1 yr": "AgeLess001",
            "1- 4 yrs": "yB8zzdlea4H",
            "5 - 14 yrs": "gBFJy81Zeyi",
            "ART No Patient ID:": "CWVHZ3hPwKs",
            "Physical Address": "VqEFza8wbwA",
            "Patient's Phone No": "P2cwLGskgxn",
            "Rx Supporter's No": "a5KkX8OWppp",
            "ART Start Date": "saTeJuuVyBd",
            "Weight (kg)": "OvY4VVhSDeJ",
            "Height (cm)": "lw1SqmMlnfh",
            "BMI": "Jgvl6hDE2y8",
            "wt/ht %": "W0ngIXKv9Iq",
            "Malnourished (y/n)": "grQdOvGre90",
            "CD4 Count": "gRbTBXfQpTf",
            "WHO Stage (1,2,3,4)": "DbMJe0tX5Kn",
            "TB Screen (n,p)": "aVU5Gi66OnU",
            "Functional Status (a,w,b)": "Npj3tBUOsmY",
            "CTX Prophylaxis (y,n)": "x7QnR5JE0I3",
            "Regimen Initial ART": "emJlQvBQ5OG",
            "TB RX ID": "GgQWHp6Buak",
            "TB Rx Site": "Bxt6B4D7YBj",
            "MUAC (cm)": "k1x8IyapRjf",
            "Count (%)": "sAHvR3qSmuG",
            "Pregnant (y,n)": "mDMkUmPySTZ",
            "FP method used": "I6BRJNfETao",
            "LMP": "w0QyOj6SFIw",
            "INH (IPT) Prophylaxis": "c6wxhKuSPTt"
        }

        for field, attr_uid in mappings.items():
            value = row.get(field, {}).get("value")
            if value:
                tei["attributes"].append({
                    "attribute": attr_uid,
                    "value": value
                })




        start_date = datetime.today().strftime("%Y-%m-%dT00:00:00.000") or "2023-01-01"


        tei["enrollments"].append({
            "program": program,
            "orgUnit": org_unit,
            "enrolledAt": start_date,
            "occurredAt": start_date,
            "status": "ACTIVE"
        })

        tracked_entities.append(tei)

    return {
            "trackedEntities": tracked_entities
    }
def register_tracked_entity(payload_: dict, mode: str) -> str:
    """
    Send a tracker mutate request to DHIS2 (v40).
    Returns:
        str: The trackedEntityInstance ID if successful, otherwise ''.
    """
    if mode == "Create":
        codes = generate_dhis2_ids(len(payload_["trackedEntities"]))
        # Update each dict with the matching code
        for i, code in enumerate(codes):
            payload_["trackedEntities"][i]["trackedEntity"] = code

    try:
        response = requests.post(
            f"{DHIS2_BASE_URL}/api/tracker",
            params={"async": "false", "importStrategy": "CREATE_AND_UPDATE"},
            json=payload_,
            auth=(DHIS2_USERNAME, DHIS2_PASSWORD)
        )
        response.raise_for_status()
        data = response.json()

        print(f"Payload submitted: {payload_}")
        print(f"DHIS2 response: {data}")

        # Try to extract TEI UID (depends on DHIS2 response structure)
        reports = (
            data.get("bundleReport", {})
            .get("typeReportMap", {})
            .get("TRACKED_ENTITY", {})
            .get("objectReports", [])
        )

        if reports:
            return reports[0].get("uid", "")

        return ""

    except Exception as e:
        print(e)
        return ""
# --- Dynamic conversion like your original code ---
def process_row(row):
    obj = row['valueObject']
    row_data = {}
    for key, value in obj.items():
        row_data[key] = {
            "value": value.get('valueString', ""),
            "confidence": value.get('confidence', None)
        }
    return row_data
# Download blob to local temp file
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
    f.write(blob.download_blob().readall())
    temp_path = f.name

clean_data = []
orgUnit = ""
reader = PdfReader(temp_path)

for i, page in enumerate(reader.pages, start=1):
    writer = PdfWriter()
    writer.add_page(page)
    single_page_path = f"page_{i}.pdf"
    with open(single_page_path, "wb") as f_out:
        writer.write(f_out)

    # Now run analysis on single_page_path like before
    with open(single_page_path, "rb") as fd:
        poller = client.begin_analyze_document(
            model_id=model_id,
            body=fd,
            content_type="application/pdf"
        )
        result = poller.result()
        print(f"--- Page {i} ---")
        for idx, doc in enumerate(result.documents):
            for name, field in doc.fields.items():
                if "OrgUnit" in name:
                    orgUnit_ = field.get("valueString", None)
                    if orgUnit_ is not None:
                        orgUnit = field.get("valueString")
                        print(f"⚠️ valueString found in {name}: {orgUnit}")
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
org_unit = None
org_unit_look_up = None
if orgUnit != "":
    org_unit_look_up = get_all(
        endpoint="organisationUnits.json",
        key="organisationUnits",
        fields="id,name,level",
        # fields="id,code,name,parent,children,level,path,ancestors",
        filters={"name": f"ilike:{orgUnit}"}
    )

if org_unit_look_up is not None:
    org_unit = org_unit_look_up[0]["id"]
    print(f"Lookup Org Unit : {org_unit}")
if org_unit is None:
    org_unit = "cYSowRjnmHE"
mapped_dhis2_tracker = map_to_dhis2_tracker(final_rows, org_unit, "o3jXXatOefs", "tsfnrMrX5bE")
register_tracked_entity(mapped_dhis2_tracker, "Create")
payload = (json.dumps(mapped_dhis2_tracker, indent=2))
print(payload)


# Method I
# clean_data = []
# with open(temp_path, "rb") as fd:
#     for page_number in range(1, 4):  # assuming 3 pages
#         fd.seek(0)  # rewind file for each request
#         poller = client.begin_analyze_document(
#             model_id=model_id,
#             body=fd,
#             content_type="application/pdf"
#         )
#         result = poller.result()
#
#         print(f"--- Page {page_number} ---")
#
#         for idx, doc in enumerate(result.documents):
#             for name, field in doc.fields.items():
#                 if "TableData" in name:
#                     if field and field.get("valueArray"):
#                         for row in field["valueArray"]:
#                             clean_data.append(process_row(row))
#                     else:
#                         print(f"⚠️ No valueArray found in {name}")

# Method 2
# Open file and analyze with Azure Document Intelligence
# with open(temp_path, "rb") as fd:
#     poller = client.begin_analyze_document(
#         model_id=model_id,
#         body=fd,  # 👈 pass file stream instead of URL
#         content_type="application/pdf"
#     )
#     result = poller.result()
#
# print(result)
#
#
#
# clean_data = []
#
#
# for idx, doc in enumerate(result.documents):
#     for name, field in doc.fields.items():
#         if "TableData" in name:
#             # Make sure field is not None and has a valueArray
#             if field and field.get("valueArray"):
#                 for row in field["valueArray"]:
#                     clean_data.append(process_row(row))
#             else:
#                 print(f"⚠️ No valueArray found in {name}")


# print(json.dumps(clean_data, indent=2))

#tester.py
import requests
from azure_document_tracker_tools import  upload_to_blob_storage, analyze_with_layout_model


# files = ["ART Register_1.pdf"]
# for file_path in files:
#
#     # Open the PDF as bytes
#     with open(file_path, "rb") as f:
#         file_bytes = f.read()
#     result = upload_to_blob_storage.invoke({"file": file_bytes, "filename":file_path})
#
#     blob_url = result["blob_url"]
#     print(blob_url)
#     analyzed_table = analyze_with_layout_model.invoke({"blob_url": blob_url, "limit": 3000, "model": "prebuilt-layout"})
#     print(analyzed_table)



DHIS2_BASE_URL = "https://dhis-upgrade.fhi360.org"
DHIS2_USERNAME = "aejakhegbe"
DHIS2_PASSWORD = "%Wekgc7345dgfgfq#"


auth = (DHIS2_USERNAME, DHIS2_PASSWORD)
payload = {
  "trackedEntities": [
    {
      "orgUnit": "cYSowRjnmHE",
      "trackedEntityType": "o3jXXatOefs",
      "attributes": [
        {
          "attribute": "AuPLng5hLbE",
          "value": "910377"
        },
        {
          "attribute": "HwDGCdte3Ck",
          "value": "2019-02-20"
        },
        {
          "attribute": "TfdH5KvFmMy",
          "value": "PAUL DAVIS"
        },
        {
          "attribute": "gHGyrwKPzej",
          "value": "2009-11-22"
        },
        {
          "attribute": "CklPZdOd6H1",
          "value": "M"
        },
        {
          "attribute": "NHviewDKFN6",
          "value": "1"
        },
        {
          "attribute": "CWVHZ3hPwKs",
          "value": "ART 1521"
        },
        {
          "attribute": "VqEFza8wbwA",
          "value": "7599, REDWOOD DRIVE, LAG"
        },
        {
          "attribute": "P2cwLGskgxn",
          "value": "356-746-4060"
        },
        {
          "attribute": "a5KkX8OWppp",
          "value": "451-208-6332"
        },
        {
          "attribute": "ArtStart01",
          "value": "2019-12 -05"
        },
        {
          "attribute": "OvY4VVhSDeJ",
          "value": "36.1"
        },
        {
          "attribute": "lw1SqmMlnfh",
          "value": "134"
        },
        {
          "attribute": "Jgvl6hDE2y8",
          "value": "20.1"
        },
        {
          "attribute": "W0ngIXKv9Iq",
          "value": "26.94"
        },
        {
          "attribute": "grQdOvGre90",
          "value": "N"
        },
        {
          "attribute": "gRbTBXfQpTf",
          "value": "166"
        },
        {
          "attribute": "DbMJe0tX5Kn",
          "value": "4"
        },
        {
          "attribute": "aVU5Gi66OnU",
          "value": "P\nA"
        },
        {
          "attribute": "x7QnR5JE0I3",
          "value": "Y"
        },
        {
          "attribute": "emJlQvBQ5OG",
          "value": "TDF + 3TC+DTG"
        },
        {
          "attribute": "GgQWHp6Buak",
          "value": "TB2025 -178"
        },
        {
          "attribute": "Bxt6B4D7YBj",
          "value": "PULMONARY"
        },
        {
          "attribute": "k1x8IyapRjf",
          "value": "27.6"
        }
      ],
      "enrollments": [
        {
          "program": "tsfnrMrX5bE",
          "orgUnit": "cYSowRjnmHE",
          "enrolledAt": "2025-09-11T00:00:00.000",
          "occurredAt": "2025-09-11T00:00:00.000",
          "status": "ACTIVE"
        }
      ]
    }
  ]
}


def register_tracked_entity() -> str:
    """
    Send a tracker import request to DHIS2 (v42).
    Returns:
        str: The trackedEntity UID if successful, otherwise ''.
    """
    try:
        response = requests.post(
            f"{DHIS2_BASE_URL}/api/tracker",
            params={"async": "false", "importStrategy": "CREATE_AND_UPDATE"},
            json=payload,
            auth=auth
        )
        response.raise_for_status()
        data = response.json()

        print(f"Payload submitted: {payload}")
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


register_tracked_entity()
# from metadata_tools import create_metadata
#
# with open("test.json") as f:
#     payload = f.read()
#
# print(create_metadata(payload))
import requests

payload = {"categories": [{"id": "HivKJ5xcA6w"}]}
params= {"importStrategy": "DELETE"}
DHIS2_BASE_URL = "https://play.im.dhis2.org/stable-2-42-1"
DHIS2_USERNAME = "admin"
DHIS2_PASSWORD = "district"

try:
    print(params)
    url = f"{DHIS2_BASE_URL}/api/metadata"
    response = requests.post(url, auth=(DHIS2_USERNAME, DHIS2_PASSWORD), json=payload, params=params)
    response.raise_for_status()
    print(response.json())

except requests.exceptions.HTTPError as http_err:
    print("HTTP error:", response.status_code)
    print("Response content:", response.text)
except Exception as e:
    print({"status": "error", "message": str(e)})

# embed_dhis2_weaviate_metadata.py
# Weaviate Vector Store Integration

import os
import requests
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import AzureOpenAIEmbeddings  # Ensure installed: pip install -U langchain-openai
from langchain.docstore.document import Document
from dotenv import load_dotenv
import weaviate

load_dotenv()

# Load Azure OpenAI Embedding config
embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),  # e.g., "text-embedding-ada-002"
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# Load DHIS2 API config
DHIS2_BASE_URL = os.getenv("DHIS2_BASE_URL")
DHIS2_USERNAME = os.getenv("DHIS2_USERNAME")
DHIS2_PASSWORD = os.getenv("DHIS2_PASSWORD")

# Load Weaviate config
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")  # Optional, for cloud/WCS

CLASS_NAME = "Document"  # Adjust if your class name is different


def fetch_metadata():
    def get_all(endpoint, key, page_size=600):
        page = 1
        all_items = []

        while True:
            if endpoint == 'organisationUnits.json':
                url = f"{DHIS2_BASE_URL}/api/{endpoint}?page={page}&pageSize={page_size}&fields=id,code,name,parent,children,level,path,ancestors"
            else:
                url = f"{DHIS2_BASE_URL}/api/{endpoint}?page={page}&pageSize={page_size}&fields=*"
            response = requests.get(url, auth=(DHIS2_USERNAME, DHIS2_PASSWORD))
            response.raise_for_status()
            data = response.json()
            items = data.get(key, [])

            if not items:
                break

            all_items.extend(items)

            # Check if it's the last page
            pager = data.get("pager", {})
            if pager.get("page", 0) >= pager.get("pageCount", 0):
                break

            page += 1

        return all_items

    data_elements_fetched = get_all("dataElements.json", "dataElements")
    indicators_fetched = get_all("indicators.json", "indicators")
    program_indicators_fetched = get_all("programIndicators.json", "programIndicators")
    org_units_fetched = get_all("organisationUnits.json", "organisationUnits")
    print(org_units_fetched)

    return data_elements_fetched, indicators_fetched, program_indicators_fetched, org_units_fetched


def build_documents(data_elements_, indicators_, program_indicators_, org_unit_):
    docs_ = []
    print(f"Count of Data Element  => {len(data_elements_)}")

    for de in data_elements_:
        content = f"{de['displayName']} - {de.get('description', '')}"
        docs_.append(Document(
            page_content=content,
            metadata={"uid": de["id"], "name": de["displayName"], "type": "dataElement"}
        ))

    print(f"Count of Indicators  => {len(indicators_)}")
    for ind in indicators_:
        content = f"{ind['displayName']} - {ind.get('description', '')}"
        docs_.append(Document(
            page_content=content,
            metadata={"uid": ind["id"], "name": ind["displayName"], "type": "indicator"}
        ))

    print(f"Count of Program Indicators  => {len(program_indicators_)}")
    for ind in program_indicators_:
        content = f"{ind['displayName']} - {ind.get('description', '')}"
        docs_.append(Document(
            page_content=content,
            metadata={
                "uid": ind["id"],
                "name": ind["displayName"],
                "type": "programIndicator"
            }
        ))

    print(f"Count of Organization Unit  => {len(org_unit_)}")
    for orgUnit in org_unit_:
        # Replace 'id' with 'uid' in parent, children, ancestors
        parent = {}
        if orgUnit.get('parent'):
            parent_id = orgUnit['parent'].get('id')
            if parent_id:
                parent = {"uid": parent_id}

        children = []
        if orgUnit.get('children'):
            children = [child.get('id') for child in orgUnit.get('children', []) if 'id' in child]
        ancestors = []

        if orgUnit.get('ancestors'):
            ancestors = [ancestor.get('id') for ancestor in orgUnit.get('ancestors', []) if 'id' in ancestor]

        content = f"OrgUnit Name: {orgUnit['name']}\nCode: {orgUnit.get('code', '')}\nPath: {orgUnit.get('path', '')}"

        docs_.append(Document(
            page_content=content,
            metadata={
                "uid": orgUnit["id"],  # Top-level ID is renamed here
                "code": orgUnit.get('code', ''),
                "name": orgUnit["name"],
                "parent": parent,
                "children": children,
                "level": orgUnit["level"],
                "path": orgUnit.get('path', ''),
                "ancestors": ancestors,
                "organisationUnitGroups": orgUnit.get('organisationUnitGroups', ''),
                "type": "organisationUnit"
            }
        ))

    return docs_


def delete_all_objects(client, class_name):
    print(f"🗑️ Deleting all objects in class '{class_name}' from Weaviate...")
    # Query all IDs of objects in the class
    query = f"""
    {{
      Get {{
        {class_name} {{
          _additional {{
            id
          }}
        }}
      }}
    }}
    """
    result = client.query.raw(query)
    ids = [obj['_additional']['id'] for obj in result['data']['Get'][class_name]]

    # Delete objects in batches (Weaviate usually supports batch sizes around 100)
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        with client.batch as batch:
            for obj_id in batch_ids:
                batch.delete_object(object_id=obj_id, class_name=class_name)
    print(f"✅ Deleted {len(ids)} objects.")


def embed_and_save_weaviate(docs):
    print(f"📦 Embedding {len(docs)} documents to Weaviate...")

    # Connect to Weaviate
    if WEAVIATE_API_KEY:
        weaviate_client = weaviate.connect_to_wcs(
            cluster_url=WEAVIATE_URL,
            auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
        )
    else:
        weaviate_client = weaviate.connect_to_local()

    # Insert new documents
    db = WeaviateVectorStore.from_documents(docs, embedding_model, client=weaviate_client)

    print(f"✅ Documents embedded and stored in Weaviate at '{WEAVIATE_URL}'")


if __name__ == "__main__":
    data_elements, indicators, program_indicators, organisationUnits = fetch_metadata()
    docs = build_documents(data_elements, indicators, program_indicators, organisationUnits)
    embed_and_save_weaviate(docs)

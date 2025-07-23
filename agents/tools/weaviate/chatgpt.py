# embed_dhis2_metadata_weaviate.py

import os
import requests
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain.docstore.document import Document
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure
from weaviate.classes.data import DataObject
from weaviate.classes.init import AdditionalConfig

load_dotenv()

# Load Azure OpenAI Embedding config
embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# Load DHIS2 API config
DHIS2_BASE_URL = os.getenv("DHIS2_BASE_URL")
DHIS2_USERNAME = os.getenv("DHIS2_USERNAME")
DHIS2_PASSWORD = os.getenv("DHIS2_PASSWORD")

# Weaviate config
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Connect to Weaviate (local or WCS)
if WEAVIATE_API_KEY:
    client = WeaviateClient(
        connection_params=ConnectionParams.from_url(
            url=WEAVIATE_URL,
            grpc_port=50051
        ),
        auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
        additional_config=AdditionalConfig(timeout=(10, 60))
    )
else:
    client = WeaviateClient(
        connection_params=ConnectionParams.from_url(
            url=WEAVIATE_URL,
            grpc_port=50051
        ),
        additional_config=AdditionalConfig(timeout=(10, 60))
    )

client.connect()  # Ensure the client is connected

# Define the Weaviate class if it doesn't exist
if not client.collections.exists("DHIS2Metadata"):
    client.collections.create(
        name="DHIS2Metadata",
        properties=[
            {"name": "content", "dataType": ["text"]},
            {"name": "type", "dataType": ["text"]},
            {"name": "name", "dataType": ["text"]},
            {"name": "item_id", "dataType": ["text"]},  # 'id' is reserved, so rename
        ],
        vectorizer_config=Configure.Vectorizer.none()
    )

collection = client.collections.get("DHIS2Metadata")


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
            pager = data.get("pager", {})
            if pager.get("page", 0) >= pager.get("pageCount", 0):
                break
            page += 1

        return all_items

    data_elements = get_all("dataElements.json", "dataElements")
    indicators = get_all("indicators.json", "indicators")
    program_indicators = get_all("programIndicators.json", "programIndicators")
    org_units = get_all("organisationUnits.json", "organisationUnits")

    return data_elements, indicators, program_indicators, org_units


def build_documents(data_elements_, indicators_, program_indicators_, org_units_):
    documents = []

    for de in data_elements_:
        content = f"{de['displayName']} - {de.get('description', '')}"
        documents.append(Document(
            page_content=content,
            metadata={"item_id": de["id"], "name": de["displayName"], "type": "dataElement"}
        ))

    for ind in indicators_:
        content = f"{ind['displayName']} - {ind.get('description', '')}"
        documents.append(Document(
            page_content=content,
            metadata={"item_id": ind["id"], "name": ind["displayName"], "type": "indicator"}
        ))

    for pi in program_indicators_:
        content = f"{pi['displayName']} - {pi.get('description', '')}"
        documents.append(Document(
            page_content=content,
            metadata={"item_id": pi["id"], "name": pi["displayName"], "type": "programIndicator"}
        ))

    for ou in org_units_:
        content = f"Name: {ou['name']} | Code: {ou.get('code', '')} | ID: {ou['id']} | Path: {ou.get('path', '')}"
        documents.append(Document(
            page_content=content,
            metadata={"item_id": ou["id"], "name": ou["name"], "type": "organisationUnit"}
        ))

    return documents


def embed_and_upload_to_weaviate(documents):
    print(f"📦 Embedding and uploading {len(documents)} documents to Weaviate...")

    texts = [doc.page_content for doc in documents]
    embeddings = embedding_model.embed_documents(texts)

    for doc, vector in zip(documents, embeddings):
        obj = DataObject(
            properties={
                "content": doc.page_content,
                "type": doc.metadata.get("type", ""),
                "name": doc.metadata.get("name", ""),
                "item_id": doc.metadata.get("item_id", ""),  # Updated here
            },
            vector=vector
        )
        collection.data.insert(obj)

    print("✅ Upload complete.")


if __name__ == "__main__":
    data_elements, indicators, program_indicators, org_units = fetch_metadata()
    docs = build_documents(data_elements, indicators, program_indicators, org_units)
    embed_and_upload_to_weaviate(docs)

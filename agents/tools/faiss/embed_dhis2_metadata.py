# embed_dhis2_metadata.py
# Facebook AI Similarity Search (Faiss)

import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings  # Install with: pip install -U langchain-openai

# from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv

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

def fetch_metadata():
    def get_all(endpoint, key, page_size=600):
        page = 1
        all_items = []

        while True:
            url = f"{DHIS2_BASE_URL}/api/{endpoint}?page={page}&pageSize={page_size}"
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

    data_elements = get_all("dataElements.json", "dataElements")
    indicators = get_all("indicators.json", "indicators")

    return data_elements, indicators


def build_documents(data_elements, indicators):
    docs = []
    print(f"Count of Data Element  => {len(data_elements)}")

    for de in data_elements:
        content = f"{de['displayName']} - {de.get('description', '')}"
        docs.append(Document(
            page_content=content,
            metadata={"id": de["id"], "name": de["displayName"], "type": "dataElement"}
        ))

    print(f"Count of Indicators  => {len(indicators)}")
    for ind in indicators:
        content = f"{ind['displayName']} - {ind.get('description', '')}"
        docs.append(Document(
            page_content=content,
            metadata={"id": ind["id"], "name": ind["displayName"], "type": "indicator"}
        ))

    return docs

def embed_and_save_index(docs, save_path="index/"):
    print(f"📦 Embedding {len(docs)} documents...")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(save_path)
    print(f"✅ FAISS index saved to '{save_path}'")

if __name__ == "__main__":
    data_elements, indicators = fetch_metadata()
    docs = build_documents(data_elements, indicators)
    embed_and_save_index(docs)

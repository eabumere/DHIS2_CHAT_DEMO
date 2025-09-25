# embed_dhis2_metadata.py
# Facebook AI Similarity Search (Faiss)

import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings  # Install with: pip install -U langchain-openai

# from langchain.vectorstores import FAISS
# from langchain.docstore.document import Document
from langchain_core.documents import Document
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
            if endpoint == 'organisationUnits.json':
                url = f"{DHIS2_BASE_URL}/api/{endpoint}?page={page}&pageSize={page_size}&fields=id,name" #code,name,parent,children,level,path,ancestors
            elif endpoint == 'dataElements.json':
                url = f"{DHIS2_BASE_URL}/api/{endpoint}?page={page}&pageSize={page_size}&fields=id,name,description,shortName,categoryCombo[id,name,categories[id,name,categoryOptions[id,name]]]"
            elif endpoint == 'categoryOptionCombos.json':
                url = f"{DHIS2_BASE_URL}/api/{endpoint}?page={page}&pageSize={page_size}&fields=id,name"
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
    program_indicators_fetched= get_all("programIndicators.json", "programIndicators")
    category_option_combo_fetched= get_all("categoryOptionCombos.json", "categoryOptionCombos")
    org_units_fetched = get_all("organisationUnits.json", "organisationUnits")
    # print(org_units_fetched)

    return data_elements_fetched, indicators_fetched, program_indicators_fetched, category_option_combo_fetched, org_units_fetched




def build_documents(
    data_elements_: list,
    indicators_: list,
    program_indicators_: list,
    org_unit_: list,
    category_option_combo_: list
):
    '''
    Build document representations of DHIS2 metadata for semantic search.
    Each document includes relevant content and metadata.

    :param data_elements_: List of data elements
    :param indicators_: List of indicators
    :param program_indicators_: List of program indicators
    :param org_unit_: List of organisation units
    :param category_option_combo_: List of category option combos
    :return: List of Document objects
    '''
    docs_ = []
    coc_docs_ = []

    print(f"Count of Data Elements  => {len(data_elements_)}")
    for de in data_elements_:
        content = f"{de.get('description', '')} - {de["name"]}"
        docs_.append(Document(
            page_content=content,
            metadata={
                "id": de["id"],
                "name": de["name"],
                "shortName": de["shortName"],
                "description": de.get('description', ''),
                "categoryCombo": de.get("categoryCombo", {}),
                "categories": de.get("categoryCombo", {}).get("categories", {}),
                "type": "dataElement"
            }
        ))

    print(f"Count of Indicators  => {len(indicators_)}")
    for ind in indicators_:
        content = f"{ind['displayName']} - {ind.get('description', '')}"
        docs_.append(Document(
            page_content=content,
            metadata={
                "id": ind["id"],
                "name": ind["displayName"],
                "numerator": ind.get("numerator", ""),
                "denominator": ind.get("denominator", ""),
                "type": "indicator"
            }
        ))

    print(f"Count of Program Indicators  => {len(program_indicators_)}")
    for ind in program_indicators_:
        content = f"{ind['displayName']} - {ind.get('description', '')}"
        docs_.append(Document(
            page_content=content,
            metadata={
                "id": ind["id"],
                "name": ind["displayName"],
                "type": "programIndicator"
            }
        ))

    print(f"Count of organisationUnits  => {len(org_unit_)}")
    for orgUnit in org_unit_:
        content = f"{orgUnit['name']}"
        docs_.append(Document(
            page_content=content,
            metadata={
                "id": orgUnit["id"],
                "name": orgUnit["name"],
                "type": "organisationUnits"
            }
        ))
    print(f"Count of CoCs  => {len(category_option_combo_)}")
    for CoC in category_option_combo_:
        content = f"{CoC['name']}"
        coc_docs_.append(Document(
            page_content=content,
            metadata={
                "id": CoC["id"],
                "name": CoC["name"],
                "type": "categoryOptionCombo"
            }
        ))
    return docs_, coc_docs_

def embed_and_save_index(docs, save_path):
    print(f"📦 Embedding {len(docs)} documents...")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(save_path)
    print(f"✅ FAISS index saved to '{save_path}'")


def run_embedding():
    # data_elements, indicators, program_indicators, organisationUnits = fetch_metadata()
    data_elements, indicators, program_indicators, category_option_combos, organisation_units = fetch_metadata()

    # docs = build_documents(data_elements, indicators, program_indicators, organisationUnits)
    docs, coc_docs_ = build_documents(data_elements, indicators, program_indicators, organisation_units, category_option_combos)

    # Step 1: split by //
    host_part = DHIS2_BASE_URL.split("//", 1)[1]

    # Step 2: get the subdomain
    subdomain = host_part.split(".")[0]

    embed_and_save_index(coc_docs_, save_path=f"index/{subdomain}/coc/")
    embed_and_save_index(docs, save_path=f"index/{subdomain}/")


if __name__ == "__main__":
    run_embedding()

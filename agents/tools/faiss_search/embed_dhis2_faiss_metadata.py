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
    program_indicators_fetched= get_all("programIndicators.json", "programIndicators")
    # org_units_fetched = get_all("organisationUnits.json", "organisationUnits")
    # print(org_units_fetched)

    return data_elements_fetched, indicators_fetched, program_indicators_fetched #org_units_fetched


def build_documents(data_elements_, indicators_, program_indicators_):
    '''
    content is what gets embedded and determines semantic similarity.
    :param data_elements_: list of data elements
    :param indicators_: list of indicators
    :param program_indicators_: list of program indicators
    :return: list of documents
    '''
    docs_ = []
    print(f"Count of Data Element  => {len(data_elements_)}")

    for de in data_elements_:
        content = f"{de['displayName']} - {de.get('description', '')}"
        docs_.append(Document(
            page_content=content,
            metadata={"id": de["id"], "name": de["displayName"], "type": "dataElement"}
        ))

    print(f"Count of Indicators  => {len(indicators_)}")
    for ind in indicators_:
        content = f"{ind['displayName']} - {ind.get('description', '')}"
        docs_.append(Document(
            page_content=content,
            metadata={"id": ind["id"],
                      "name": ind["displayName"],
                      "numerator": ind["numerator"],
                      "denominator": ind["denominator"],
                      "type": "indicator"}
        ))

    print(f"Count of Program Indicators  => {len(program_indicators_)}")
    for ind in program_indicators_:
        content = f"{ind['displayName']} - {ind.get('description', '')}"
        docs_.append(Document(
            page_content=content,
            metadata=
                    {
                        "id": ind["id"],
                        "name": ind["displayName"],
                        "type": "programIndicator"
                    }
        ))

    # print(f"Count of Organization Unit  => {len(org_unit_)}")
    # for orgUnit in org_unit_:
    #     content = f"Name: {orgUnit['name']} | Code: {orgUnit.get('code', '')} | ID: {orgUnit['id']} | Path: {orgUnit.get('path', '')}"
    #     docs_.append(Document(
    #         page_content=content,
    #         metadata=
    #                 {
    #                     "id": orgUnit["id"],
    #                     "code": orgUnit.get('code', ''),
    #                     "name": orgUnit["name"],
    #                     "parent": orgUnit.get('parent', ''),
    #                     "children": orgUnit.get('children', ''),
    #                     "level": orgUnit["level"],
    #                     "path": orgUnit.get('path', ''),
    #                     "ancestors": orgUnit.get('ancestors', ''),
    #                     "organisationUnitGroups": orgUnit.get('organisationUnitGroups', ''),
    #                     "type": "organisationUnit"
    #                 }
    #         ))
    #
    return docs_

def embed_and_save_index(docs, save_path="index/"):
    print(f"📦 Embedding {len(docs)} documents...")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(save_path)
    print(f"✅ FAISS index saved to '{save_path}'")

if __name__ == "__main__":
    # data_elements, indicators, program_indicators, organisationUnits = fetch_metadata()
    data_elements, indicators, program_indicators = fetch_metadata()

    # docs = build_documents(data_elements, indicators, program_indicators, organisationUnits)
    docs = build_documents(data_elements, indicators, program_indicators)

    embed_and_save_index(docs)

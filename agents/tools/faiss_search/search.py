# search.py

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings  # Ensure it's installed

# Load environment variables
load_dotenv()

# Set up the embedding model
embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# Load the FAISS index with fallback paths
faiss_paths = [
    "faiss_search/index/",
    "tools/faiss_search/index/",
    "agents/tools/faiss_search/index/",
    "index/"
]

for path in faiss_paths:
    try:
        vectorstore = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
        break
    except Exception as e:
        last_exception = e
else:
    raise RuntimeError(f"Failed to load FAISS index from all fallback paths. Last error: {last_exception}")


# Search the index
def fiass_query(query):
    results = vectorstore.similarity_search(query, k=10)
    # Print top results
    print(f"\n🔍 Top results for: '{query}'\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.page_content} - {r.metadata['type']} - {r.metadata['id']}\n")

def filter_documents_by_metadata(doc_type: str, filters: dict = None):
    """
    Filters documents from the FAISS vectorstore based on document type and metadata filters.

    Args:
        doc_type (str): The type of document (e.g., "organisationUnit", "dataElement").
        filters (dict): Optional metadata filters. Keys are metadata fields, values are exact match values.

    Returns:
        List of Document objects matching the criteria.
    """
    results = []

    for doc in vectorstore.docstore._dict.values():
        if doc.metadata.get("type") != doc_type:
            continue

        if filters:
            match = all(doc.metadata.get(k) == v for k, v in filters.items())
            if not match:
                continue

        results.append(doc)

    return results

# fiass_query("FUrCpcvMAmC")
# Get all org units with level == 2

#Exact Match
# level2_units = filter_documents_by_metadata("organisationUnit", {"level": 2})
# print(level2_units)
# # Get all data elements with a specific ID
# some_data_element = filter_documents_by_metadata("dataElement", {"id": "uF4KBYgV7zJ"})
# print(some_data_element)
#
# # Get all indicators (no additional filters)
# all_indicators = filter_documents_by_metadata("indicator")
# print(all_indicators)
# search.py

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings  # Ensure it's installed

# Load environment variables
load_dotenv()
FAISS_THRESHOLD = float(os.getenv("FAISS_THRESHOLD", 0.12))
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

last_exception = ""
for path in faiss_paths:
    try:
        vectorstore = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
        break
    except Exception as e:
        last_exception = e
else:
    raise RuntimeError(f"Failed to load FAISS index from all fallback paths. Last error: {last_exception}")

def hybrid_search(query: str):
    # Step 1: Try exact search
    matches = []
    query_lower = query.strip().lower()
    filtered_matches = []

    for doc in vectorstore.docstore._dict.values():
        if (
            query_lower == doc.page_content.strip().lower() or
            query_lower == doc.metadata.get("name", "").strip().lower() or
            query_lower == doc.metadata.get("shortName", "").strip().lower() or
            query_lower == doc.metadata.get("description", "").strip().lower()
        ):
            matches.append(doc)
    if matches:
        print("✅ Exact match found.")
        print(f"\n🔍 Top results for: '{query}'\n")
        for i, r in enumerate(matches, 1):
            print(f"{i}. {r.page_content} - {r.metadata['type']} - {r.metadata['id']}\n")
            filtered_matches.append({
                "name": r.metadata.get("name", ""),
                "id": r.metadata.get("id", ""),
                "doc_type": r.metadata.get("type", "Unknown")
            })
        return filtered_matches
        # return exact_matches[:top_k]

    # Step 2: Fall back to semantic search
    print("🔍 Falling back to semantic search...")
    return fiass_query(query)

def fiass_query(query):
    """
    Perform a FAISS similarity search and print results above a given score threshold.

    Args:
        query (str): The search query.
        threshold (float): The minimum similarity score to include a result (lower = more similar).
    """
    results = vectorstore.similarity_search_with_score(query, k=5)
    filtered_matches = []

    print(f"\n🔍 Top results for: '{query}' (threshold ≤ {FAISS_THRESHOLD})\n")
    count = 0
    for i, (doc, score) in enumerate(results, 1):
        if score <= FAISS_THRESHOLD:
            count += 1
            doc_type = doc.metadata.get("type", "unknown")
            doc_id = doc.metadata.get("id", "N/A")
            name = doc.metadata.get("name", "")
            print(f"{count}. {doc.page_content} - {doc_type} - {doc_id} (score: {score:.4f})\n")
            filtered_matches.append({
                "name": name,
                "id": doc_id,
                "doc_type": doc_type,
                "score": float(score)
            })
    return filtered_matches

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

# print(hybrid_search("received HIV Testing Services (HTS) and received their test results"))
# hybrid_search("HTS_TST")

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
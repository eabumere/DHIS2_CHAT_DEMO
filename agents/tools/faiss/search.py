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

# Load the FAISS index
vectorstore = FAISS.load_local("index/", embedding_model, allow_dangerous_deserialization=True)

# Search the index
query = "Birth"
results = vectorstore.similarity_search(query, k=3)

# Print top results
print(f"\n🔍 Top results for: '{query}'\n")
for i, r in enumerate(results, 1):
    print(f"{i}. {r.page_content}\n")

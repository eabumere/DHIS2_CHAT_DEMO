import os
from weaviate import WeaviateClient
from langchain_openai import AzureOpenAIEmbeddings

# Initialize Weaviate client
client = WeaviateClient(
    embedded_options=None,
    grpc_port=50051,  # adjust if needed
    http_port=8080,
    startup_timeout=10,
)

# 🧠 Embed the query using AzureOpenAI
query_text = "number of deliveries by facility"
embedder = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

query_vector = embedder.embed_query(query_text)

# 🔍 Choose the correct collection (replace with your actual one)
collection_name = "LangChain_96e945eb5f194bdabb4472fdca1269fa"

# 🧪 Search using near_vector
results = client.collections.get(collection_name).query.near_vector(
    near_vector=query_vector,
    limit=5
)

# 📦 Print Results
print(f"\n🔎 Top Results for: '{query_text}'\n")
for obj in results.objects:
    print("📄", obj.properties.get("text") or obj.properties)  # adjust key if needed
    print("---")

# ✅ Clean shutdown
client.close()

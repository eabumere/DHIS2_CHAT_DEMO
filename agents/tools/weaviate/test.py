
import weaviate

client = weaviate.WeaviateClient(
    url="http://localhost:8080",
    # auth_client_secret=weaviate.AuthApiKey(api_key="YOUR_API_KEY")  # if needed
)

# Get the schema
schema = client.schema.get()

# Print class names
classes = schema.get("classes", [])
print("Existing classes:")
for c in classes:
    print(f"- {c['class']}")

# WARNING: This deletes entire schema and all data!
# print("🗑️ Deleting entire schema from Weaviate (all classes and data)...")
# weaviate_client.schema.delete_all()
# print("✅ Schema deleted.")
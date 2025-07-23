from metadata_tools import create_metadata

with open("test.json") as f:
    payload = f.read()

print(create_metadata(payload))
#tester.py

from azure_document_tracker_tools import  upload_to_blob_storage, analyze_with_layout_model

files = ["ART Register_1.pdf"]
for file_path in files:

    # Open the PDF as bytes
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    result = upload_to_blob_storage.invoke({"file": file_bytes, "filename":file_path})

    blob_url = result["blob_url"]
    print(blob_url)
    analyzed_table = analyze_with_layout_model.invoke({"blob_url": blob_url, "limit": 3000, "model": "prebuilt-layout"})
    print(analyzed_table)



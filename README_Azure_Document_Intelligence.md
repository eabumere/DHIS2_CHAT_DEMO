# Azure Document Intelligence for DHIS2

## Overview

This implementation uses **Azure Document Intelligence** with **LangChain integration** to create a production-ready solution for processing facility registers and converting handwritten data to DHIS2 JSON payloads. This approach provides enterprise-grade accuracy, scalability, and reliability with seamless LangChain integration.

## 🏗️ Architecture

### **LangChain + Azure Document Intelligence Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   Azure Blob     │    │   LangChain     │
│   Upload        │───▶│   Storage        │───▶│   AzureAIDoc    │
│   (Scan/Photo)  │    │                  │    │   Intelligence  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   DHIS2 API     │◀───│   Azure Function │◀───│   Document      │
│   Response      │    │   (Processing)   │    │   Analysis      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Key Components**

1. **LangChain Azure Document Intelligence Integration**
   - **AzureAIDocumentIntelligenceLoader**: LangChain's official loader for Azure Document Intelligence
   - **Document Processing**: Seamless integration with LangChain's document processing pipeline
   - **Metadata Extraction**: Rich metadata extraction with confidence scoring
   - **Table Recognition**: Advanced table structure analysis

2. **Azure Document Intelligence**
   - **Custom Model Training**: Learn specific facility register layouts
   - **Layout Analysis**: Extract tables, text, and structure
   - **Handwriting Recognition**: High-accuracy OCR for handwritten text
   - **Model Versioning**: Deploy and manage different model versions

3. **Azure Blob Storage**
   - Document upload and storage
   - Training data management
   - Processed document archiving

4. **Azure Functions**
   - Serverless document processing
   - DHIS2 API integration
   - Error handling and retry logic

5. **Azure Key Vault**
   - Secure credential management
   - Configuration secrets
   - Access control

## 🚀 Features

### **1. LangChain Integration**
- **AzureAIDocumentIntelligenceLoader**: Official LangChain loader for Azure Document Intelligence
- **Document Processing Pipeline**: Seamless integration with LangChain's document processing
- **Metadata Extraction**: Rich metadata with confidence scoring
- **Table Recognition**: Advanced table structure analysis

### **2. Custom Model Training**
- **Layout Learning**: Train models on your specific facility register formats
- **Field Recognition**: Automatically identify DHIS2-relevant fields
- **Handwriting Support**: High-accuracy recognition of handwritten entries
- **Model Evolution**: Retrain and improve models over time

### **3. Intelligent Data Extraction**
- **Table Structure**: Understand complex table layouts
- **Key-Value Pairs**: Extract form fields and labels
- **Confidence Scoring**: Quality assessment for extracted data
- **Multi-format Support**: Handle various document types

### **4. DHIS2 Integration**
- **Automatic Mapping**: Map extracted fields to DHIS2 metadata
- **Data Validation**: Ensure DHIS2 compliance before submission
- **Batch Processing**: Handle multiple records efficiently
- **Error Handling**: Robust error recovery and reporting

### **5. Production Features**
- **Scalability**: Handle high-volume document processing
- **Monitoring**: Application Insights integration
- **Security**: Azure Key Vault for credential management
- **Cost Optimization**: Pay-per-use pricing model

## 📋 Prerequisites

### **Azure Resources Required**
1. **Azure Subscription** with billing enabled
2. **Azure Document Intelligence** resource (S0 tier recommended)
3. **Azure Storage Account** (Standard LRS)
4. **Azure Function App** (Consumption plan)
5. **Azure Key Vault** (for secrets management)

### **DHIS2 Configuration**
- DHIS2 instance URL
- API credentials (username/password)
- Required data elements and metadata

## 🛠️ Installation & Setup

### **1. Azure Resource Deployment**

```bash
# Install Azure CLI
az login

# Create resource group
az group create --name dhis2-document-intelligence --location eastus

# Deploy Azure resources using ARM template
az deployment group create \
  --resource-group dhis2-document-intelligence \
  --template-file azure_deployment_config.json
```

### **2. Environment Configuration**

```bash
# Set environment variables
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://your-document-intelligence.cognitiveservices.azure.com/"
export AZURE_DOCUMENT_INTELLIGENCE_KEY="your-document-intelligence-key"
export AZURE_STORAGE_CONNECTION_STRING="your-storage-connection-string"
export DHIS2_BASE_URL="https://your-dhis2-instance.com"
export DHIS2_USERNAME="your-dhis2-username"
export DHIS2_PASSWORD="your-dhis2-password"
```

### **3. Install Dependencies**

```bash
# Install Azure Document Intelligence requirements
pip install -r requirements_azure_document_intelligence.txt

# Install Azure Functions Core Tools
npm install -g azure-functions-core-tools@4
```

### **4. Deploy Azure Function**

```bash
# Navigate to function directory
cd azure_functions

# Deploy function
func azure functionapp publish dhis2-document-processor
```

## 🎯 Usage

### **1. Custom Model Training**

#### **Prepare Training Data**
```python
# Upload training samples to Azure Blob Storage
training_samples = [
    "empty_facility_register.pdf",
    "filled_facility_register.pdf",
    "handwritten_entries.pdf"
]

# Train custom model
from agents.tools.azure_document_tools import train_custom_model

training_result = train_custom_model(
    model_name="facility_register_v1",
    training_data={"blob_url": "https://storage.blob.core.windows.net/training-data/"}
)
```

#### **Model Training Process**
1. **Upload Samples**: 5-15 labeled samples of your facility register
2. **Label Fields**: Mark DHIS2-relevant fields in each sample
3. **Train Model**: Azure trains the custom model (2-3 hours)
4. **Deploy Model**: Make the trained model available for processing

### **2. Document Processing**

#### **Using LangChain AzureAIDocumentIntelligenceLoader**
```python
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

# Process document with LangChain loader
loader = AzureAIDocumentIntelligenceLoader(
    api_endpoint="https://your-document-intelligence.cognitiveservices.azure.com/",
    api_key="your-api-key",
    file_path="https://storage.blob.core.windows.net/documents/facility_register.pdf",
    api_model="prebuilt-layout"  # or custom model ID
)

# Load and analyze documents
documents = loader.load()

# Extract information
for doc in documents:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

#### **Using Azure Function**
```python
import requests
import json

# Process document through Azure Function
function_url = "https://dhis2-document-processor.azurewebsites.net/api/dhis2_document_processor"

payload = {
    "blob_url": "https://storage.blob.core.windows.net/documents/facility_register.pdf",
    "model_id": "facility_register_v1",  # Optional: use custom model
    "mapping_config": {
        "facility_name": "orgUnit",
        "reporting_period": "period",
        "anc_visits": "dataElement",
        "patient_count": "value"
    }
}

response = requests.post(function_url, json=payload)
result = response.json()
```

#### **Using Python Agent**
```python
from agents.azure_document_intelligence_agent import azure_document_intelligence_executor

# Process document with agent
result = azure_document_intelligence_executor.invoke({
    "messages": [{"role": "user", "content": "Process this facility register"}]
})
```

### **3. End-to-End Workflow**

#### **Step 1: Document Upload**
```python
# Upload document to Azure Blob Storage
from agents.tools.azure_document_tools import upload_to_blob_storage

upload_result = upload_to_blob_storage.invoke({"image": document_image})
blob_url = upload_result["blob_url"]
```

#### **Step 2: Document Analysis**
```python
# Analyze with Azure Document Intelligence using LangChain
from agents.tools.azure_document_tools import analyze_with_layout_model

analysis_result = analyze_with_layout_model.invoke({
    "blob_url": blob_url
})
```

#### **Step 3: DHIS2 Mapping**
```python
# Map extracted data to DHIS2
from agents.tools.azure_document_tools import map_to_dhis2_metadata

mapping_result = map_to_dhis2_metadata.invoke({
    "extraction_result": analysis_result,
    "model_type": "layout"
})
```

#### **Step 4: Submit to DHIS2**
```python
# The Azure Function automatically handles DHIS2 submission
# Or use the agent for manual submission
```

## 📊 Monitoring & Analytics

### **Application Insights**
- **Performance Monitoring**: Track processing times and throughput
- **Error Tracking**: Monitor and alert on processing failures
- **Custom Metrics**: Track DHIS2 submission success rates
- **Log Analytics**: Centralized logging and analysis

### **Key Metrics**
- Document processing success rate
- Average processing time
- DHIS2 API response times
- Custom model accuracy
- Cost per document processed

### **Alerts**
- Function execution failures
- DHIS2 API errors
- Model training failures
- Storage quota warnings

## 🔧 Configuration

### **DHIS2 Mapping Configuration**
```json
{
  "facility_register": {
    "anc_visits": "dataElement",
    "malaria_cases": "dataElement",
    "facility_name": "orgUnit",
    "reporting_period": "period",
    "patient_count": "value"
  }
}
```

### **Custom Model Configuration**
```json
{
  "facility_register_model": {
    "model_id": "facility-register-v1",
    "description": "Custom model for facility register processing",
    "training_data_requirements": {
      "minimum_samples": 5,
      "recommended_samples": 15
    }
  }
}
```

## 💰 Cost Optimization

### **Estimated Monthly Costs**
- **Document Intelligence (S0)**: $50-100/month
- **Storage (100GB)**: $5-20/month
- **Function App (Consumption)**: $10-30/month
- **Total**: $65-150/month

### **Cost Optimization Strategies**
1. **Batch Processing**: Process multiple documents together
2. **Model Reuse**: Use trained models for similar document types
3. **Storage Lifecycle**: Archive old documents to cheaper storage
4. **Function Optimization**: Optimize function execution time

## 🔒 Security

### **Data Protection**
- **Encryption at Rest**: All data encrypted in Azure Storage
- **Encryption in Transit**: HTTPS for all communications
- **Access Control**: Azure Key Vault for credential management
- **Audit Logging**: Comprehensive access and operation logs

### **Compliance**
- **GDPR Compliance**: Data residency and privacy controls
- **HIPAA Ready**: Healthcare data protection features
- **ISO 27001**: Information security management

## 🚀 Advanced Features

### **1. LangChain Document Processing**
```python
# Use LangChain's document processing pipeline
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load documents
loader = AzureAIDocumentIntelligenceLoader(
    api_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
    file_path=blob_url,
    api_model="prebuilt-layout"
)

documents = loader.load()

# Process documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)
```

### **2. Model Versioning**
```python
# Deploy specific model version
from agents.tools.azure_document_tools import deploy_model_version

deployment_result = deploy_model_version.invoke({
    "model_id": "facility_register_v1"
})
```

### **3. Batch Processing**
```python
# Process multiple documents
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
results = []

for doc in documents:
    result = process_document(doc)
    results.append(result)
```

### **4. Custom Field Extraction**
```python
# Define custom fields for extraction
custom_fields = {
    "patient_id": "text",
    "diagnosis": "text",
    "treatment_date": "date",
    "outcome": "choice"
}
```

## 🐛 Troubleshooting

### **Common Issues**

1. **Model Training Failures**
   - Ensure sufficient training samples (5-15)
   - Check sample quality and labeling
   - Verify Azure Document Intelligence quota

2. **Document Processing Errors**
   - Check document format (PDF, JPEG, PNG)
   - Verify blob storage access
   - Review function logs

3. **DHIS2 Integration Issues**
   - Validate DHIS2 API credentials
   - Check network connectivity
   - Review DHIS2 API response format

4. **LangChain Integration Issues**
   - Verify Azure Document Intelligence credentials
   - Check document loader configuration
   - Review document processing pipeline

### **Debugging Tools**
- **Azure Portal**: Monitor resources and logs
- **Application Insights**: Detailed performance analysis
- **Function Logs**: Real-time function execution logs
- **Storage Explorer**: Browse uploaded documents

## 📈 Performance Optimization

### **Processing Speed**
- **Parallel Processing**: Process multiple documents simultaneously
- **Model Caching**: Reuse trained models for similar documents
- **Function Scaling**: Automatic scaling based on demand

### **Accuracy Improvement**
- **Model Retraining**: Regular model updates with new samples
- **Field Validation**: Post-processing validation rules
- **Confidence Thresholds**: Adjust based on requirements

## 🔮 Future Enhancements

### **Planned Features**
- **Multi-language Support**: Process documents in different languages
- **Real-time Processing**: Live document processing capabilities
- **Advanced Analytics**: Machine learning insights on document patterns
- **Mobile Integration**: Direct mobile app integration

### **Research Areas**
- **Deep Learning**: Advanced neural networks for better accuracy
- **Computer Vision**: Enhanced document structure analysis
- **Natural Language Processing**: Better understanding of document content
- **Federated Learning**: Collaborative model training across organizations

## 📞 Support

### **Documentation**
- [Azure Document Intelligence Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/document-intelligence/)
- [LangChain Azure Document Intelligence](https://python.langchain.com/docs/integrations/document_loaders/azure_ai_document_intelligence)
- [Azure Functions Documentation](https://docs.microsoft.com/en-us/azure/azure-functions/)
- [DHIS2 API Documentation](https://docs.dhis2.org/)

### **Community**
- [Azure Community](https://techcommunity.microsoft.com/t5/azure/ct-p/Azure)
- [LangChain Community](https://discord.gg/langchain)
- [DHIS2 Community](https://community.dhis2.org/)
- [GitHub Issues](https://github.com/your-repo/issues)

### **Professional Support**
- Azure Support Plans
- DHIS2 Professional Services
- Custom Development Services

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**This Azure Document Intelligence implementation with LangChain integration provides a production-ready, scalable solution for processing facility registers and integrating with DHIS2. The combination of LangChain's document processing pipeline, custom model training, handwriting recognition, and automated DHIS2 integration makes it ideal for healthcare data management.** 
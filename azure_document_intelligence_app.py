# azure_document_intelligence_app.py
import streamlit as st
import pandas as pd
from PIL import Image
import os

from agents.azure_document_intelligence_agent import azure_document_intelligence_executor
from agents.tools.azure_document_tracker_tools import (
    upload_to_blob_storage,
    analyze_with_layout_model,
    analyze_with_custom_model,
    train_custom_model,
    map_to_dhis2_metadata,
    validate_dhis2_payload,
    deploy_model_version,
    get_model_training_status
)

st.set_page_config(
    page_title="Azure Document Intelligence for DHIS2",
    page_icon="☁️",
    layout="wide"
)

st.title("☁️ Azure Document Intelligence for DHIS2")
st.markdown("""
Enterprise-grade document processing using Azure Document Intelligence (Form Recognizer).
Train custom models, process facility registers, and integrate with DHIS2 seamlessly.
""")

# Sidebar for Azure configuration
st.sidebar.header("🔧 Azure Configuration")

# Check if Azure credentials are configured
azure_configured = all([
    os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT"),
    os.getenv("AZURE_FORM_RECOGNIZER_KEY"),
    os.getenv("AZURE_STORAGE_CONNECTION_STRING")
])

if not azure_configured:
    st.sidebar.warning("⚠️ Azure credentials not configured")
    st.sidebar.markdown("""
    Please set the following environment variables:
    - `AZURE_FORM_RECOGNIZER_ENDPOINT`
    - `AZURE_FORM_RECOGNIZER_KEY`
    - `AZURE_STORAGE_CONNECTION_STRING`
    """)
else:
    st.sidebar.success("✅ Azure credentials configured")

# Main content area
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Document Processing", 
    "🤖 Model Training", 
    "🗺️ DHIS2 Mapping", 
    "📊 Monitoring", 
    "💬 Chat Interface"
])

with tab1:
    st.header("📄 Document Processing")
    
    # Document upload
    uploaded_file = st.file_uploader(
        "Upload Document", 
        type=['png', 'jpg', 'jpeg', 'pdf', 'tiff', 'bmp'],
        help="Upload a facility register or document for processing"
    )
    
    if uploaded_file is not None:
        # Display the uploaded document
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            st.session_state["uploaded_document_image"] = image
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📷 Uploaded Document")
                st.image(image, caption="Uploaded Document", use_column_width=True)
            
            with col2:
                st.subheader("🔍 Processing Options")
                
                # Processing options
                processing_mode = st.selectbox(
                    "Processing Mode",
                    ["Layout Analysis", "Custom Model", "Both"],
                    help="Choose how to process the document"
                )
                
                if st.button("🚀 Process Document"):
                    if azure_configured:
                        with st.spinner("Processing document with Azure Document Intelligence..."):
                            try:
                                # Upload to Azure Blob Storage
                                upload_result = upload_to_blob_storage.invoke({"image": image})
                                
                                if upload_result.get("success"):
                                    blob_url = upload_result.get("blob_url")
                                    st.success(f"✅ Document uploaded to Azure: {blob_url}")
                                    
                                    # Process with selected mode
                                    if processing_mode in ["Layout Analysis", "Both"]:
                                        st.subheader("📋 Layout Analysis Results")
                                        layout_result = analyze_with_layout_model.invoke({"blob_url": blob_url})
                                        
                                        if layout_result.get("success"):
                                            st.success("✅ Layout analysis completed")
                                            
                                            # Display tables
                                            tables = layout_result.get("tables", [])
                                            if tables:
                                                st.write(f"**Tables Found:** {len(tables)}")
                                                for i, table in enumerate(tables):
                                                    with st.expander(f"Table {i+1} ({table['row_count']}x{table['column_count']})"):
                                                        # Convert table data to DataFrame
                                                        table_data = []
                                                        for cell in table.get("cells", []):
                                                            table_data.append({
                                                                "Row": cell.get("row_index"),
                                                                "Column": cell.get("column_index"),
                                                                "Content": cell.get("content"),
                                                                "Confidence": f"{cell.get('confidence', 0):.2f}"
                                                            })
                                                        
                                                        if table_data:
                                                            df = pd.DataFrame(table_data)
                                                            st.dataframe(df)
                                            
                                            # Display key-value pairs
                                            kv_pairs = layout_result.get("key_value_pairs", [])
                                            if kv_pairs:
                                                st.write(f"**Key-Value Pairs Found:** {len(kv_pairs)}")
                                                kv_data = []
                                                for kv in kv_pairs:
                                                    kv_data.append({
                                                        "Key": kv.get("key"),
                                                        "Value": kv.get("value"),
                                                        "Confidence": f"{kv.get('confidence', 0):.2f}"
                                                    })
                                                
                                                if kv_data:
                                                    kv_df = pd.DataFrame(kv_data)
                                                    st.dataframe(kv_df)
                                            
                                            # Store results
                                            st.session_state["layout_analysis_result"] = layout_result
                                        else:
                                            st.error(f"❌ Layout analysis failed: {layout_result.get('error')}")
                                    
                                    if processing_mode in ["Custom Model", "Both"]:
                                        st.subheader("🤖 Custom Model Analysis")
                                        custom_result = analyze_with_custom_model.invoke({"blob_url": blob_url})
                                        
                                        if custom_result.get("success"):
                                            st.success("✅ Custom model analysis completed")
                                            
                                            # Display extracted data
                                            extracted_data = custom_result.get("extracted_data", {})
                                            if extracted_data:
                                                st.write(f"**Fields Extracted:** {len(extracted_data)}")
                                                field_data = []
                                                for field_name, field_info in extracted_data.items():
                                                    field_data.append({
                                                        "Field": field_name,
                                                        "Content": field_info.get("content"),
                                                        "Confidence": f"{field_info.get('confidence', 0):.2f}"
                                                    })
                                                
                                                if field_data:
                                                    field_df = pd.DataFrame(field_data)
                                                    st.dataframe(field_df)
                                            
                                            # Store results
                                            st.session_state["custom_model_result"] = custom_result
                                        else:
                                            st.warning(f"⚠️ Custom model not available: {custom_result.get('error')}")
                                    
                                    # Store blob URL for later use
                                    st.session_state["blob_url"] = blob_url
                                    
                                else:
                                    st.error(f"❌ Upload failed: {upload_result.get('error')}")
                                    
                            except Exception as e:
                                st.error(f"❌ Processing error: {str(e)}")
                    else:
                        st.error("❌ Azure credentials not configured")

with tab2:
    st.header("🤖 Model Training")
    
    if azure_configured:
        # Model training section
        st.subheader("Train Custom Model")
        
        model_name = st.text_input(
            "Model Name",
            value="facility_register_model",
            help="Name for the custom model"
        )
        
        # Training data upload
        training_files = st.file_uploader(
            "Upload Training Data",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            accept_multiple_files=True,
            help="Upload 5-15 labeled samples for training"
        )
        
        if training_files and st.button("🎯 Start Training"):
            with st.spinner("Starting model training..."):
                try:
                    # Upload training files
                    uploaded_urls = []
                    for file in training_files:
                        if file.type.startswith('image'):
                            image = Image.open(file)
                            upload_result = upload_to_blob_storage.invoke({"image": image})
                            if upload_result.get("success"):
                                uploaded_urls.append(upload_result.get("blob_url"))
                    
                    if uploaded_urls:
                        # Start training
                        training_result = train_custom_model.invoke({
                            "model_name": model_name,
                            "training_data": {"blob_url": uploaded_urls[0]}  # Use first URL as reference
                        })
                        
                        if training_result.get("success"):
                            st.success("✅ Model training started successfully!")
                            
                            # Display training info
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Model Name", training_result.get("model_name"))
                            with col2:
                                st.metric("Training ID", training_result.get("training_id"))
                            with col3:
                                st.metric("Status", training_result.get("status"))
                            
                            st.info(f"⏱️ Estimated training time: {training_result.get('estimated_time')}")
                            
                            # Store training info
                            st.session_state["training_info"] = training_result
                        else:
                            st.error(f"❌ Training failed: {training_result.get('error')}")
                    else:
                        st.error("❌ No training files uploaded successfully")
                        
                except Exception as e:
                    st.error(f"❌ Training error: {str(e)}")
        
        # Check training status
        if "training_info" in st.session_state:
            st.subheader("📊 Training Status")
            
            if st.button("🔄 Check Status"):
                training_info = st.session_state["training_info"]
                model_id = training_info.get("training_id")
                
                if model_id:
                    status_result = get_model_training_status.invoke({"model_id": model_id})
                    
                    if status_result.get("success"):
                        st.success(f"✅ Model Status: {status_result.get('status')}")
                        
                        # Display status details
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Created:**", status_result.get("created_at", "Unknown"))
                        with col2:
                            st.write("**Last Updated:**", status_result.get("last_updated", "Unknown"))
                        
                        if status_result.get("description"):
                            st.write("**Description:**", status_result.get("description"))
                    else:
                        st.error(f"❌ Status check failed: {status_result.get('error')}")
        
        # Model deployment
        st.subheader("🚀 Deploy Model")
        
        model_to_deploy = st.text_input(
            "Model ID to Deploy",
            help="Enter the model ID to deploy"
        )
        
        if model_to_deploy and st.button("🚀 Deploy Model"):
            with st.spinner("Deploying model..."):
                try:
                    deploy_result = deploy_model_version.invoke({"model_id": model_to_deploy})
                    
                    if deploy_result.get("success"):
                        st.success("✅ Model deployed successfully!")
                        
                        deployment_info = deploy_result.get("deployment", {})
                        st.write("**Deployment Info:**")
                        st.json(deployment_info)
                    else:
                        st.error(f"❌ Deployment failed: {deploy_result.get('error')}")
                        
                except Exception as e:
                    st.error(f"❌ Deployment error: {str(e)}")
    else:
        st.warning("⚠️ Azure credentials required for model training")

with tab3:
    st.header("🗺️ DHIS2 Mapping")
    
    if "layout_analysis_result" in st.session_state or "custom_model_result" in st.session_state:
        st.subheader("Map to DHIS2 Metadata")
        
        # Choose which result to use for mapping
        if "custom_model_result" in st.session_state:
            extraction_result = st.session_state["custom_model_result"]
            model_type = "custom"
            st.success("✅ Using custom model results")
        elif "layout_analysis_result" in st.session_state:
            extraction_result = st.session_state["layout_analysis_result"]
            model_type = "layout"
            st.success("✅ Using layout analysis results")
        
        if st.button("🗺️ Map to DHIS2"):
            with st.spinner("Mapping data to DHIS2 metadata..."):
                try:
                    mapping_result = map_to_dhis2_metadata.invoke({
                        "extraction_result": extraction_result,
                        "model_type": model_type
                    })
                    
                    if mapping_result.get("success"):
                        st.success("✅ DHIS2 mapping completed!")
                        
                        # Display mapping results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Mapping Summary")
                            st.metric("Overall Confidence", f"{mapping_result.get('overall_confidence', 0):.1f}%")
                            st.metric("Mapped Fields", len(mapping_result.get("mapped_fields", [])))
                            st.metric("DHIS2 Fields Covered", len(mapping_result.get("dhis2_fields_covered", [])))
                        
                        with col2:
                            st.subheader("🎯 DHIS2 Fields Covered")
                            covered_fields = mapping_result.get("dhis2_fields_covered", [])
                            if covered_fields:
                                for field in covered_fields:
                                    st.write(f"• {field}")
                            else:
                                st.write("No DHIS2 fields mapped")
                        
                        # Display detailed mapping
                        st.subheader("📋 Detailed Mapping")
                        mapping_data = mapping_result.get("mapping", {})
                        if mapping_data:
                            mapping_list = []
                            for field_name, field_info in mapping_data.items():
                                if isinstance(field_info, dict):
                                    mapping_list.append({
                                        "Source Field": field_name,
                                        "DHIS2 Field": field_info.get("dhis2_field"),
                                        "Value": field_info.get("value"),
                                        "Confidence": f"{field_info.get('confidence', 0):.2f}"
                                    })
                            
                            if mapping_list:
                                mapping_df = pd.DataFrame(mapping_list)
                                st.dataframe(mapping_df)
                        
                        # Validate mapping
                        st.subheader("✅ Validation")
                        validation_result = validate_dhis2_payload.invoke({
                            "mapping_result": mapping_result
                        })
                        
                        if validation_result.get("success"):
                            status = validation_result.get("status")
                            confidence = validation_result.get("confidence", 0)
                            
                            if status == "valid":
                                st.success(f"✅ Mapping is valid (Confidence: {confidence:.1f}%)")
                            elif status == "partial":
                                st.warning(f"⚠️ Partial mapping (Confidence: {confidence:.1f}%)")
                            else:
                                st.error(f"❌ Invalid mapping (Confidence: {confidence:.1f}%)")
                            
                            # Display validation details
                            missing_fields = validation_result.get("missing_fields", [])
                            if missing_fields:
                                st.write("**Missing Required Fields:**")
                                for field in missing_fields:
                                    st.write(f"• {field}")
                            
                            validation_errors = validation_result.get("validation_errors", [])
                            if validation_errors:
                                st.write("**Validation Errors:**")
                                for error in validation_errors:
                                    st.write(f"• {error}")
                        
                        # Store results
                        st.session_state["dhis2_mapping_result"] = mapping_result
                        st.session_state["validation_result"] = validation_result
                        
                    else:
                        st.error(f"❌ Mapping failed: {mapping_result.get('error')}")
                        
                except Exception as e:
                    st.error(f"❌ Mapping error: {str(e)}")
    else:
        st.info("ℹ️ Upload and process a document first to enable DHIS2 mapping")

with tab4:
    st.header("📊 Monitoring")
    
    if azure_configured:
        # Azure resource monitoring
        st.subheader("Azure Resources")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Form Recognizer", "Active", delta="S0 Tier")
        
        with col2:
            st.metric("Storage Account", "Active", delta="Standard LRS")
        
        with col3:
            st.metric("Function App", "Active", delta="Consumption Plan")
        
        # Processing statistics
        st.subheader("Processing Statistics")
        
        if "layout_analysis_result" in st.session_state:
            layout_result = st.session_state["layout_analysis_result"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tables = layout_result.get("tables", [])
                st.metric("Tables Extracted", len(tables))
            
            with col2:
                kv_pairs = layout_result.get("key_value_pairs", [])
                st.metric("Key-Value Pairs", len(kv_pairs))
            
            with col3:
                confidence = layout_result.get("overall_confidence", 0)
                st.metric("Overall Confidence", f"{confidence:.1f}%")
        
        # Model training status
        if "training_info" in st.session_state:
            st.subheader("Model Training Status")
            
            training_info = st.session_state["training_info"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Name", training_info.get("model_name", "Unknown"))
            
            with col2:
                st.metric("Status", training_info.get("status", "Unknown"))
            
            with col3:
                st.metric("Training ID", training_info.get("training_id", "Unknown"))
    else:
        st.warning("⚠️ Azure credentials required for monitoring")

with tab5:
    st.header("💬 Chat Interface")
    st.markdown("""
    Chat with the Azure Document Intelligence agent. Try commands like:
    - "Process this document with Azure Document Intelligence"
    - "Train a custom model for facility registers"
    - "Map the extracted data to DHIS2"
    - "Deploy the latest model version"
    """)
    
    # Initialize chat history
    if "azure_chat_messages" not in st.session_state:
        st.session_state.azure_chat_messages = []
    
    # Display chat messages
    for message in st.session_state.azure_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about Azure Document Intelligence..."):
        # Add user message to chat history
        st.session_state.azure_chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from Azure Document Intelligence agent
        with st.chat_message("assistant"):
            with st.spinner("Processing with Azure Document Intelligence..."):
                try:
                    result = azure_document_intelligence_executor.invoke({
                        "messages": [{"role": "user", "content": prompt}]
                    })
                    
                    response = result.get("output", "No response generated")
                    if hasattr(response, 'content'):
                        response_content = response.content
                    else:
                        response_content = str(response)
                    
                    st.markdown(response_content)
                    
                    # Add assistant response to chat history
                    st.session_state.azure_chat_messages.append({"role": "assistant", "content": response_content})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.azure_chat_messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("""
**Azure Document Intelligence Features:**
- ☁️ **Azure Integration**: Enterprise-grade document processing
- 🤖 **Custom Models**: Train models for your specific document layouts
- ✍️ **Handwriting Recognition**: High-accuracy OCR for handwritten text
- 🗺️ **DHIS2 Mapping**: Automatic mapping to DHIS2 metadata
- 📊 **Production Ready**: Scalable, monitored, and secure
- 💰 **Cost Optimized**: Pay-per-use pricing model
""") 
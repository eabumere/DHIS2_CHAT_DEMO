import streamlit as st
from agent import openai_executor
from langchain_core.messages import HumanMessage
import json
import requests
import os
import io


def apply_custom_theme(theme_name):
    if theme_name == "Dark":
        st.markdown("""
            <style>
            body { background-color: #0e1117; color: #f0f0f0; }
            .stButton>button { background-color: #333 !important; color: #fff; }
            </style>
        """, unsafe_allow_html=True)
    elif theme_name == "Light":
        st.markdown("""
            <style>
            body { background-color: #ffffff; color: #000000; }
            </style>
        """, unsafe_allow_html=True)
    elif theme_name == "Ocean":
        st.markdown("""
            <style>
            body { background-color: #e0f7fa; color: #01579b; }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            body { background-color: inherit; color: inherit; }
            </style>
        """, unsafe_allow_html=True)


st.set_page_config(page_title="DHIS2 Chat", layout="wide")

# Initialize theme if not set
if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark"

# Sidebar theme selector
theme_choice = st.sidebar.radio("Choose Theme", ["Dark", "Light", "Ocean", "Default"])
st.session_state["theme"] = theme_choice
apply_custom_theme(st.session_state["theme"])

st.title("🗨️ DHIS2 Chat Assistant")
st.markdown("Use chat to activate a DHIS2 instance by creating metadata like datasets, programs, etc.")

# Session state setup
if "messages" not in st.session_state:
    st.session_state.messages = []
if "fetched_metadata" not in st.session_state:
    st.session_state.fetched_metadata = []
if "metadata_type" not in st.session_state:
    st.session_state.metadata_type = ""

# Sidebar info
st.sidebar.title("DHIS2 AI Solutions")
st.sidebar.info("Upload metadata or use chat to generate DHIS2 objects.")

# Display past messages with safe role fallback
for message in st.session_state.messages:
    role = message["role"]
    display_role = "assistant" if role in {"assistant", "ai", "tool"} else "user"
    with st.chat_message(display_role):
        st.markdown(message["content"])

# Handle chat input
if prompt := st.chat_input("Ask me to create a dataset, program, etc..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    result = openai_executor.invoke({
        "messages": [
            HumanMessage(content=m["content"]) if m["role"] == "user" else m["content"]
            for m in st.session_state.messages
        ]
    })

    assistant_reply = result.get("output", "⚠️ No output from agent.")

    # Try to parse metadata from reply
    try:
        metadata_start = assistant_reply.find("[")
        metadata_end = assistant_reply.rfind("]") + 1
        if metadata_start != -1 and metadata_end != -1:
            json_text = assistant_reply[metadata_start:metadata_end]
            fetched_metadata = json.loads(json_text)
            if isinstance(fetched_metadata, list):
                st.session_state.fetched_metadata = fetched_metadata
                st.session_state.metadata_type = "Unknown"
    except Exception:
        pass

    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

        # Toggle metadata view if JSON was parsed
        if st.session_state.fetched_metadata:
            st.markdown("Would you like to view the full metadata below?")
            st.session_state["show_metadata_section"] = st.toggle("Show structured metadata view", value=True)
        else:
            st.session_state["show_metadata_section"] = False

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

# Display structured metadata view
if st.session_state.get("show_metadata_section", False):
    st.markdown("---")
    st.header(f"Fetched Metadata ({st.session_state.metadata_type or 'Unknown type'})")

    metadata = st.session_state.fetched_metadata
    with st.expander("🔍 Raw metadata preview"):
        st.json(metadata)

    page_size = 20
    total = len(metadata)
    total_pages = (total + page_size - 1) // page_size

    show_all = st.checkbox("Show all metadata items", value=False)

    if show_all:
        st.write(f"Total items: {total}")
        for i, item in enumerate(metadata, start=1):
            display_name = item.get("displayName") or item.get("name") or f"Item {i}"
            st.write(f"{i}. {display_name} — ID: {item.get('id', 'N/A')}")
    else:
        page = st.number_input("Select page", min_value=1, max_value=total_pages, value=1)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total)
        st.write(f"Showing items {start_idx + 1} to {end_idx} of {total}")
        for i, item in enumerate(metadata[start_idx:end_idx], start=start_idx + 1):
            display_name = item.get("displayName") or item.get("name") or f"Item {i}"
            st.write(f"{i}. {display_name} — ID: {item.get('id', 'N/A')}")

# File upload and metadata import
file = st.file_uploader("Upload JSON metadata", type=["json"])
if file:
    try:
        metadata = json.load(file)
        st.success("Sending uploaded metadata to DHIS2...")

        response = requests.post(
            f"{os.getenv('DHIS2_BASE_URL')}/api/metadata",
            auth=(os.getenv("DHIS2_USERNAME"), os.getenv("DHIS2_PASSWORD")),
            headers={"Content-Type": "application/json"},
            json=metadata,
            params={"importStrategy": "CREATE_UPDATE"}
        )
        st.json(response.json())
    except Exception as e:
        st.error(f"Failed to upload metadata: {e}")

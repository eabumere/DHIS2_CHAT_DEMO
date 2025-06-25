import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from multi_agent import metadata_agent_executor, analytics_executor, routing_decision
import json

st.set_page_config(page_title="DHIS2 Assistant", layout="wide")
# Display your company logo
# Use columns to align logo and title side by side
col1, col2 = st.columns([1, 5])  # Adjust ratio as needed

with col1:
    st.image("fhi360.png")  # Adjust width to fit nicely

with col2:
    st.title("🗨️ DHIS2 Chat Assistant")
# st.markdown("Ask me to create metadata or run analytics in DHIS2. I will remember our chat.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input
if prompt := st.chat_input("Ask DHIS2 Assistant..."):
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(prompt)

    # 🔄 Use full chat history for routing decision and execution
    state = {"messages": st.session_state.messages}

    route = routing_decision(state)

    if route == "metadata":
        result = metadata_agent_executor.invoke(state)
    elif route == "analytics":
        result = analytics_executor.invoke(state)
    else:
        assistant_msg = AIMessage(content="❌ Unknown routing decision. Try a different question.")
        st.session_state.messages.append(assistant_msg)
        with st.chat_message("assistant"):
            st.markdown(assistant_msg.content)
        st.stop()

    # 🧠 Add agent response to history
    output = result.get("output")
    assistant_msg = output if isinstance(output, AIMessage) else AIMessage(content=str(output))
    st.session_state.messages.append(assistant_msg)

    with st.chat_message("assistant"):
        st.markdown(assistant_msg.content)

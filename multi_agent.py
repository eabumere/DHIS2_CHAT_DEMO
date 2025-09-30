# multi_agent.py
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
from typing import List, TypedDict, Optional, Union
import os
import pandas as pd
from PIL import Image
from utils.llm import get_llm
import streamlit as st
# --- LLM for Routing ---
llm = get_llm()
# Load environment variables
load_dotenv()

# --- Import agent executors ---
from agents.metadata_agent import metadata_agent_executor
from agents.analytics_agent import analytics_executor
from agents.tracker_data_agent import tracker_data_executor
from agents.event_data_agent import event_data_executor
from agents.data_entry_agent import data_entry_executor
# from agents.azure_document_intelligence_agent import azure_document_intelligence_executor


# --- State ---
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    last_active_agent: Optional[str]

system_prompt = """
You are a dispatcher agent for a DHIS2 assistant.
Your job is to decide which specialized agent should handle the user's request.

There are five available agents:

1. **Metadata Agent** (`metadata_agent`)
   - Use for operations like creating, updating, deleting, or retrieving metadata.
   - e.g., "Create a new dataset", "What is the UID for ANC visits?"

2. **Analytics Agent** (`analytics_agent`)
   - Use for analytical queries.
   - e.g., "Show me a trend of malaria cases", "What's the total for Bo district?"

3. **Data Entry Agent** (`data_entry_agent`)
   - Use for entering or updating aggregate data values.
   - Handles follow-up actions and confirmations related to previously started data entry operations.

4. **Event Data Agent** (`event_data_agent`)
   - Use for entering data for event-based (non-tracker) programs.
   - e.g., "Record a malaria event in Kenema on March 5th."

5. **Tracker Data Agent** (`tracker_data_agent`)
   - Use for tracked entity operations (registration-based).
   - e.g., "Register a pregnant woman for ANC", "Record a follow-up visit for ID XYZ."

ROUTING RULES:
- If the user is confirming a previous operation (e.g., "yes", "confirm", "proceed", "submit", "delete it"),
  route to the **last_active_agent** (provided in context).
- Route to `metadata_agent` only for explicit metadata CRUD operations.
- Route to `data_entry_agent` for aggregate data submissions, updates, or deletions.
- Route to `analytics_agent`, `event_data_agent`, or `tracker_data_agent` for their respective domains.
- If the request is ambiguous, return `clarify_agent` so the system can ask the user directly.

Respond with **only one** of the following values (and nothing else):
- `metadata_agent`
- `analytics_agent`
- `data_entry_agent`
- `event_data_agent`
- `tracker_data_agent`
- `clarify_agent`
"""




# 6. **Azure Document Intelligence Agent** (`azure_document_intelligence_agent`)
#    - Use for enterprise-grade document processing with Azure services:
#      - Custom model training for facility register layouts
#      - High-accuracy handwriting recognition with Azure Document Intelligence
#      - Automated DHIS2 mapping and validation
#      - Production-ready document processing pipeline
#      - e.g., "Train a custom model for facility registers", "Process with Azure Document Intelligence", "Deploy model version"
# - `azure_document_intelligence_agent`
# --- Prompt Template ---
router_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages")
])


# --- Routing Decision ---
def routing_decision(state: AgentState) -> str:
    """Decides which agent to route the request to based on last user message."""
    if not state.get("messages"):
        return "metadata"

    last_user_msg = state["messages"][-1]
    user_text = last_user_msg.content.strip().lower()
    # print(f"last message {last_user_msg}")

    # --- Step 1: Detect confirmations or numeric selections ---
    confirmation_keywords = {"yes", "confirm", "proceed", "submit", "delete", "ok", "sure", "proceed with submitting this data", "no"}
    numeric_selection = user_text.isdigit()
    looks_like_uid = len(user_text) == 11 and user_text.isalnum()
    last_agent = state.get("last_active_agent")
    if user_text in confirmation_keywords or numeric_selection or looks_like_uid:
        last_agent = state.get("last_active_agent")
        print(f"present agent: {last_agent}")
        if last_agent:
            print(f"Routing to last active agent due to confirmation/selection/UID: {last_agent}")
            return last_agent
        else:
            # Instead of defaulting, instruct Streamlit to ask user for clarification
            return "clarify_agent"

    # --- Step 2: Ask LLM for classification ---
    full_prompt = router_prompt.invoke({"messages": [last_user_msg]})
    response = llm.invoke(full_prompt)

    decision_raw = response.content.strip().lower()
    decision = decision_raw.strip("`'\"")

    print(f"Router Input: {last_user_msg.content}")
    print(f"Router LLM Raw Response: {repr(response.content)}")
    print(f"Router Decision (normalized): {repr(decision)}")

    valid_routes = {
        "metadata_agent": "metadata",
        "analytics_agent": "analytics",
        "data_entry_agent": "data_entry",
        "event_data_agent": "event_data",
        "tracker_data_agent": "tracker_data",
    }

    if decision in valid_routes:
        chosen = valid_routes[decision]
        state["last_active_agent"] = chosen  # persist context
        st.session_state.last_active_agent = chosen
        return chosen

    # --- Step 3: Keyword fallback ---
    keyword_routes = {
        "analytics_agent": ["report", "chart", "trend", "visual", "analysis", "insight"],
        "metadata_agent": ["data element", "dataset", "program", "org unit", "indicator",
                           "create metadata", "update metadata", "delete metadata"],
        "data_entry_agent": ["aggregate", "data entry", "submit", "enter value",
                             "create aggregate data", "update data", "delete data"],
        "event_data_agent": ["event", "single event", "event program", "malaria event"],
        "tracker_data_agent": ["tracked", "tei", "enrollment", "visit", "follow-up", "patients", "enroll", "individual", "Register"],
    }

    for agent, keywords in keyword_routes.items():
        if any(k in user_text for k in keywords):
            chosen = valid_routes[agent]
            state["last_active_agent"] = chosen  # persist context
            print(f"Routing based on keyword fallback to: {agent}")
            return chosen

    # --- Step 4: Could not classify ---
    print("No valid routing found. Asking user to clarify explicitly.")
    return "clarify_agent"



# --- Agent Nodes ---
def dispatcher_node(state: AgentState) -> AgentState:
    # Pass through unchanged, routing will be done by the graph
    return state

def metadata_node(state: AgentState) -> AgentState:
    # Send messages to metadata agent executor and append response
    result = metadata_agent_executor.invoke({"messages": state["messages"]})
    output = result.get("output")
    # Append AIMessage response to messages list in state
    state["messages"].append(output if isinstance(output, AIMessage) else AIMessage(content=str(output)))
    return state

def analytics_node(state: AgentState) -> AgentState:
    result = analytics_executor.invoke({"messages": state["messages"]})
    output = result.get("output")
    state["messages"].append(output if isinstance(output, AIMessage) else AIMessage(content=str(output)))
    return state

def tracker_data_node(state: AgentState) -> AgentState:
    result = tracker_data_executor.invoke({"messages": state["messages"]})
    output = result.get("output")
    state["messages"].append(output if isinstance(output, AIMessage) else AIMessage(content=str(output)))
    return state

def event_data_node(state: AgentState) -> AgentState:
    result = event_data_executor.invoke({"messages": state["messages"]})
    output = result.get("output")
    state["messages"].append(output if isinstance(output, AIMessage) else AIMessage(content=str(output)))
    return state

def data_entry_node(state: AgentState) -> AgentState:
    result = data_entry_executor.invoke({
        "messages": state["messages"]})
    output = result.get("output")
    state["messages"].append(output if isinstance(output, AIMessage) else AIMessage(content=str(output)))
    return state

# def azure_document_intelligence_node(state: AgentState) -> AgentState:
#     result = azure_document_intelligence_executor.invoke({"messages": state["messages"]})
#     output = result.get("output")
#     state["messages"].append(output if isinstance(output, AIMessage) else AIMessage(content=str(output)))
#     return state


# --- Graph Setup ---
graph = StateGraph(AgentState)
graph.add_node("dispatcher", dispatcher_node)
graph.add_node("metadata", metadata_node)
graph.add_node("analytics", analytics_node)
graph.add_node("data_entry", data_entry_node)
graph.add_node("event_data", event_data_node)
graph.add_node("tracker_data", tracker_data_node)
# graph.add_node("azure_document_intelligence", azure_document_intelligence_node)

graph.set_entry_point("dispatcher")

graph.add_conditional_edges("dispatcher", routing_decision, {
    "metadata": "metadata",
    "analytics": "analytics",
    "data_entry": "data_entry",
    "event_data": "event_data",
    "tracker_data": "tracker_data"
})
#"azure_document_intelligence": "azure_document_intelligence"
# Terminal edges from agent nodes to END
for node in ["metadata", "analytics", "data_entry", "event_data", "tracker_data"]:
    graph.add_edge(node, END)
 #"azure_document_intelligence"
compiled_multi_agent = graph.compile()

# --- Session Memory ---
session_histories = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]


# --- Final Executor ---
dispatcher_executor = RunnableWithMessageHistory(
    compiled_multi_agent,
    get_session_history,
    input_messages_key="messages",
    history_messages_key="messages"
)

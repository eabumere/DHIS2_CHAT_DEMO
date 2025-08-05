# multi_agent.py
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import AzureChatOpenAI
from langchain.memory import ChatMessageHistory
from dotenv import load_dotenv
from typing import List, TypedDict, Optional, Union
import os
import pandas as pd
from PIL import Image
from utils.llm import get_llm
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


# --- State ---
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]

# --- Routing Prompt ---
system_prompt = """
You are a dispatcher agent for a DHIS2 assistant.
Your job is to decide which specialized agent should handle the user’s request.

There are five available agents:

1. **Metadata Agent** (`metadata_agent`)
   - Use for operations like creating, updating, deleting, or retrieving metadata:
     - Examples: datasets, programs, data elements, org units, indicators, categories.
     - e.g., “Create a new dataset”, “What is the UID for ANC visits?”

2. **Analytics Agent** (`analytics_agent`)
   - Use for analytical queries:
     - Generating charts, reports, trends, or summaries.
     - Computing totals, averages, or performing breakdowns.
     - e.g., “Show me a trend of malaria cases”, “What’s the total for Bo district?”

3. **Data Entry Agent** (`data_entry_agent`)
   - Use for entering or updating aggregate data values:
     - Standard data sets or data elements (not program-based).
     - e.g., “Enter 25 malaria cases for Bombali for January.”

4. **Event Data Agent** (`event_data_agent`)
   - Use for entering data for event-based (non-tracker) programs:
     - Typically no registration, just single events.
     - e.g., “Record a malaria event in Kenema on March 5th.”

5. **Tracker Data Agent** (`tracker_data_agent`)
   - Use for tracked entity operations (registration-based):
     - Includes persons, follow-up visits, tracked events.
     - e.g., “Register a pregnant woman for ANC”, “Record a follow-up visit for ID XYZ.”

Respond with **only one** of the following values (and nothing else):
- `metadata_agent`
- `analytics_agent`
- `data_entry_agent`
- `event_data_agent`
- `tracker_data_agent`
"""

# --- Prompt Template ---
router_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages")
])


# --- Routing Decision ---
def routing_decision(state: AgentState) -> str:
    """Decides which agent to route the request to based on last user message."""
    if state["messages"]:
        last_user_msg = state["messages"][-1]

        # Prepare prompt to classify message
        full_prompt = router_prompt.invoke({"messages": [last_user_msg]})
        response = llm.invoke(full_prompt)

        # Clean and normalize the LLM decision carefully
        decision_raw = response.content.strip().lower()

        # Remove backticks or quotes surrounding the response if any
        decision = decision_raw.strip("`'\"")

        # Debugging with repr to show hidden chars if any
        print(f"Router Input: {last_user_msg.content}")
        print(f"Router LLM Raw Response: {repr(response.content)}")
        print(f"Router Decision (normalized): {repr(decision)}")

        # Valid routes map LLM output to graph node names
        valid_routes = {
            "metadata_agent": "metadata",
            "analytics_agent": "analytics",
            "data_entry_agent": "data_entry",
            "event_data_agent": "event_data",
            "tracker_data_agent": "tracker_data"
        }

        # Strict match on cleaned decision
        if decision in valid_routes:
            return valid_routes[decision]

        # Fallback: keyword-based routing if LLM response invalid or unexpected
        text = last_user_msg.content.lower()

        keyword_routes = {
            "analytics_agent": ["report", "chart", "trend", "visual", "analysis", "insight"],
            "metadata_agent": [
                "data element", "dataset", "program", "org unit", "indicator",
                "create metadata", "update metadata", "delete metadata"
            ],
            "data_entry_agent": ["aggregate", "aggregate data entry", "data entry"
                "form", "submit", "enter value", "fill form",
                "create data", "update data", "delete data"
            ],
            "event_data_agent": [
                "event", "single event", "event program", "malaria event", "non-tracker event"
            ],
            "tracker_data_agent": [
                "tracked", "tei", "enrollment", "enrol", "visit", "follow-up", "tracked entity"
            ]
        }

        def keyword_fallback(text: str) -> str:
            for agent, keywords in keyword_routes.items():
                if any(k in text for k in keywords):
                    print(f"Routing based on keyword fallback to: {agent}")
                    return valid_routes.get(agent, "metadata")
            return "metadata"

        return keyword_fallback(text)

    # Final fallback if no messages
    return "metadata"


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


# --- Graph Setup ---
graph = StateGraph(AgentState)
graph.add_node("dispatcher", dispatcher_node)
graph.add_node("metadata", metadata_node)
graph.add_node("analytics", analytics_node)
graph.add_node("data_entry", data_entry_node)
graph.add_node("event_data", event_data_node)
graph.add_node("tracker_data", tracker_data_node)

graph.set_entry_point("dispatcher")

graph.add_conditional_edges("dispatcher", routing_decision, {
    "metadata": "metadata",
    "analytics": "analytics",
    "data_entry": "data_entry",
    "event_data": "event_data",
    "tracker_data": "tracker_data"
})

# Terminal edges from agent nodes to END
for node in ["metadata", "analytics", "data_entry", "event_data", "tracker_data"]:
    graph.add_edge(node, END)

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

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import AzureChatOpenAI
from langchain.memory import ChatMessageHistory
from dotenv import load_dotenv
from typing import List, TypedDict
import os

# Import your executors that handle metadata and analytics respectively
from agents.metadata_agent import openai_executor as metadata_agent_executor
from agents.analytics_agent import analytics_executor

# Load environment variables for API keys, endpoints, etc.
load_dotenv()


# TypedDict to hold message state (chat history)
class AgentState(TypedDict):
    messages: List[BaseMessage]


# Improved Dispatcher prompt to decide which agent to route to
system_prompt = """
You are a dispatcher agent for a DHIS2 assistant.
Your job is to decide whether a user request is related to:

1. **Metadata operations** (e.g., create/update/delete datasets, data elements, org units, programs, etc.)
2. **Analytics queries** (e.g., data value summaries, charts, reports, insights)

If the user request is about creating, updating, deleting, or retrieving any metadata, respond with exactly `metadata_agent`.
If the request is about analyzing data or generating reports, respond with exactly `analytics_agent`.

Respond ONLY with the agent name: `metadata_agent` or `analytics_agent`.
"""

# Create a ChatPromptTemplate for routing
router_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages")
])

# Initialize the Azure OpenAI Chat model used for routing decision
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT_4O"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    model="gpt-4",
    temperature=0
)


def routing_decision(state: AgentState) -> str:
    """Decides which agent to route the request to based on last user message."""
    if state["messages"]:
        last_user_msg = state["messages"][-1]
        # Prepare prompt to classify message
        full_prompt = router_prompt.invoke({"messages": [last_user_msg]})
        response = llm.invoke(full_prompt)
        decision = response.content.strip().lower()

        # Debug output
        print(f"Router Input: {last_user_msg.content}")
        print(f"Router LLM Raw Response: {response.content}")
        print(f"Router Decision: {decision}")

        # Basic keyword routing fallback in case LLM output is unexpected
        if "analytics" in decision or "analytics_agent" in decision:
            return "analytics"
        elif "metadata" in decision or "metadata_agent" in decision:
            return "metadata"
        else:
            # Keyword fallback based on input text if LLM unclear
            text = last_user_msg.content.lower()
            metadata_keywords = ["create", "update", "delete", "data element", "dataset", "program", "org unit"]
            analytics_keywords = ["analytics", "report", "chart", "summary", "insight"]
            if any(k in text for k in analytics_keywords):
                print("Routing based on keyword fallback to analytics")
                return "analytics"
            else:
                print("Routing based on keyword fallback to metadata")
                return "metadata"
    # Default fallback
    return "metadata"


def dispatcher_node(state: AgentState) -> AgentState:
    # Just pass state unchanged in dispatcher
    return state


def metadata_node(state: AgentState) -> AgentState:
    # Send messages to metadata agent executor and append response
    result = metadata_agent_executor.invoke({"messages": state["messages"]})
    output = result.get("output")
    # Append AIMessage response to messages list in state
    state["messages"].append(output if isinstance(output, AIMessage) else AIMessage(content=str(output)))
    return state


def analytics_node(state: AgentState) -> AgentState:
    # Send messages to analytics agent executor and append response
    result = analytics_executor.invoke({"messages": state["messages"]})
    output = result.get("output")
    # Append AIMessage response to messages list in state
    state["messages"].append(output if isinstance(output, AIMessage) else AIMessage(content=str(output)))
    return state


# Build the state graph with dispatcher, metadata, and analytics nodes
graph = StateGraph(AgentState)
graph.add_node("dispatcher", dispatcher_node)
graph.add_node("metadata", metadata_node)
graph.add_node("analytics", analytics_node)

# Set entry point of the graph to dispatcher
graph.set_entry_point("dispatcher")

# Add conditional routing edges from dispatcher
graph.add_conditional_edges("dispatcher", routing_decision, {
    "metadata": "metadata",
    "analytics": "analytics"
})

# Terminal edges from metadata and analytics
graph.add_edge("metadata", END)
graph.add_edge("analytics", END)

# Compile graph to get runnable
compiled_multi_agent = graph.compile()

# In-memory chat history store per session
session_histories = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """Retrieve or create a chat history object for the session."""
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]


# Wrap compiled multi-agent graph in a Runnable with message history support
dispatcher_executor = RunnableWithMessageHistory(
    compiled_multi_agent,
    get_session_history,  # Returns ChatMessageHistory
    input_messages_key="messages",
    history_messages_key="messages"
)

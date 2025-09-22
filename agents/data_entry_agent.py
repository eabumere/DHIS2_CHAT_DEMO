# agents/data_entry_agent.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .tools.data_entry_tools import submit_aggregate_data, suggest_column_mapping
from dotenv import load_dotenv
from typing import List, TypedDict, Optional, Any
import streamlit as st
from utils.llm import get_llm
from PIL import Image

load_dotenv()

class AgentState(TypedDict):
    messages: List[BaseMessage]
    output: Optional[AIMessage]
    dataframe_columns: Optional[Any]  # for df, typically pandas.DataFrame
    pdf_text: Optional[str]
    image: Optional[Image.Image]
    word_text: Optional[str]
    ppt_text: Optional[str]



system_prompt = """
You are a DHIS2 data entry assistant. Your job is to help users submit, update, and delete data in DHIS2.

AVAILABLE TOOLS:
1. suggest_column_mapping - Use this to map CSV columns to DHIS2 fields
2. submit_aggregate_data - Use this to submit, update, or delete data

WORKFLOW:
1. When user uploads data, use suggest_column_mapping to create column mappings
2. When user wants to submit/import data, call submit_aggregate_data with preview_only=True first
3. When user confirms, call submit_aggregate_data with preview_only=False
4. When user wants to delete data, call submit_aggregate_data with params="DELETE"

IMPORTANT RULES:
- ALWAYS use the tools. Never just describe what you will do.
- For any data operation (submit/update/delete), you MUST call submit_aggregate_data
- If user says "use it", "submit", "import", "delete", "update" - call the appropriate tool immediately
- Do not ask for clarification unless absolutely necessary
- Show previews before destructive operations

RESPONSE FORMAT:
- When calling tools, just call them directly
- When showing results, be concise and clear
- Never include internal instructions in your responses to users
"""

llm = get_llm()
tools = [submit_aggregate_data, suggest_column_mapping]
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
openai_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=False,  # Reduce verbosity to prevent prompt leakage
    return_intermediate_steps=False,  # Don't return intermediate steps to prevent confusion
    handle_parsing_errors=True,
    max_iterations=3  # Reduce iterations to prevent loops
)

def agent_node(state: AgentState) -> AgentState:
    try:
        messages = state.get("messages", [])
        
        # Clear session state if this is a new conversation or new data upload
        if messages and len(messages) == 1:  # First user message
            if "suggested_columns_data_entry" in st.session_state:
                del st.session_state["suggested_columns_data_entry"]
        
        # Check if we need to generate or regenerate column mapping
        should_generate_mapping = False
        
        # Case 1: No mapping exists
        if "raw_data_df_uploaded" in st.session_state and not st.session_state.get("suggested_columns_data_entry"):
            should_generate_mapping = True
        
        # Case 2: Mapping exists but we're starting a new operation (check last user message)
        elif "raw_data_df_uploaded" in st.session_state and st.session_state.get("suggested_columns_data_entry"):
            if messages and hasattr(messages[-1], 'content'):
                last_message = messages[-1].content.lower()
                # If user is asking for a new operation, regenerate mapping
                if any(keyword in last_message for keyword in ["submit", "delete", "update", "import", "post", "send", "use", "sure", "proceed"]):
                    should_generate_mapping = True
        
        # Generate column mapping if needed
        if should_generate_mapping:
            column_mapping = suggest_column_mapping.invoke({
                "columns": state.get("dataframe_columns", [])
            })
            st.session_state.suggested_columns_data_entry = column_mapping

            # Add context message
            context_msg = AIMessage(content=f"""
                Available data: CSV with columns {state.get("dataframe_columns", [])}
                Suggested mapping: {column_mapping}
                Use submit_aggregate_data tool to process this data.
            """)
            messages.append(context_msg)
        
        # If we have column mapping but no context message, add it
        elif "raw_data_df_uploaded" in st.session_state and st.session_state.get("suggested_columns_data_entry") and not any("Available data: CSV with columns" in msg.content for msg in messages if hasattr(msg, 'content')):
            column_mapping = st.session_state.get("suggested_columns_data_entry")
            context_msg = AIMessage(content=f"""
            Available data: CSV with columns {state.get("dataframe_columns", [])}
            Suggested mapping: {column_mapping}
            Use submit_aggregate_data tool to process this data.
            """)
            messages.append(context_msg)
        
        state["messages"] = messages
        
        # Force tool usage by checking if user wants an operation
        if messages and hasattr(messages[-1], 'content'):
            last_message = messages[-1].content.lower()
            if any(keyword in last_message for keyword in ["submit", "delete", "update", "import", "post", "send", "use", "proceed"]):
                # Add a system message to force tool usage
                force_tool_msg = AIMessage(content="You must call submit_aggregate_data tool now. Do not respond with text only.")
                messages.append(force_tool_msg)
                state["messages"] = messages
        
        result = openai_executor.invoke(state)  # Pass full state with all messages
        return {
            "messages": result["messages"],
            "output": result["output"]
        }
    except Exception as e:
        # Log full error for debugging
        print(f"❌ Error: {str(e)}")

        # Friendly message for UI
        if "token" in str(e).lower() or "rate limit" in str(e).lower():
            user_friendly_message = "❌ The system is currently overloaded (token limit exceeded). Please wait and try again, or contact your administrator."
        else:
            user_friendly_message = "❌ An unexpected error occurred. Please contact your administrator."

        # Append the friendly error message to chat history
        messages = state.get("messages", []) + [AIMessage(content=user_friendly_message)]

        return {
            "messages": messages,
            "output": AIMessage(content=user_friendly_message)
        }


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)
data_entry_executor = graph.compile()

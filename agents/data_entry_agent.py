# agents/data_entry_agent.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage
from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .tools.data_entry_tools import submit_aggregate_data, suggest_column_mapping
from dotenv import load_dotenv
from typing import List, TypedDict, Optional, Any
import os
import pandas as pd
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



llm = get_llm()

tools = [submit_aggregate_data, suggest_column_mapping]

system_prompt = """
You are a data entry assistant for DHIS2, specializing in aggregate data submissions.

Your responsibilities include:
- Submitting or deleting aggregate data values to/from DHIS2 using the `submit_aggregate_data` tool.
- Ensuring all submissions include the required fields:
  - dataElement
  - period
  - orgUnit
  - categoryOptionCombo
  - attributeOptionCombo
  - value

Data Source and Column Mapping:
- All data comes from a pre-uploaded dataframe stored in: `st.session_state["raw_data_df_uploaded"]`.
- This dataframe is not directly accessible to you, but tool functions can read from it.
- You are provided with the uploaded data's column names via the `dataframe_columns` argument.
- Your task is to map these user-provided column names to DHIS2’s required fields.
- Normalize common formatting issues like:
  - snake_case → camelCase (e.g., `org_unit` → `orgUnit`)
  - case differences (e.g., `Dataelement` → `dataElement`)
  - common aliases (e.g., `val` or `count` → `value`, `facility` → `orgUnit`)

⚠️ TOOL CHAINING AND MEMORY:
- When you call the `suggest_column_mapping` tool, you MUST remember and reuse its output.
- The result of `suggest_column_mapping` must be passed explicitly as the `column_mapping` argument when you call `submit_aggregate_data`.
- Do not proceed to submit or preview without a valid column_mapping.
- Failing to pass `column_mapping` will result in a validation error.

Tool Usage:
- Use `suggest_column_mapping` to infer or confirm column mappings when uncertain.
- To preview the submission or deletion payload, call `submit_aggregate_data` with:
  - `preview_only=True`
  - a valid `column_mapping`
  - a `params` value of either `"CREATE_AND_UPDATE"` or `"DELETE"`

Submission vs Deletion:
- To **submit** data, proceed only if the user clearly says: "submit", "post", or "send".
  - Call `submit_aggregate_data` with `preview_only=False` and `params="CREATE_AND_UPDATE"`.
- To **delete** data, proceed only if the user clearly says: "delete", "remove", or "retract".
  - Call `submit_aggregate_data` with `preview_only=False` and `params="DELETE"`.
- For both submission and deletion, the same required fields must be present in the mapped data.

Clarify Before Acting:
- If any required field appears missing or is ambiguously mapped, do not proceed silently.
  - Ask the user for clarification or confirmation.
  - Clearly explain any assumptions or uncertainties.

Restrictions:
- Do not access or rely on the full dataframe yourself — only use tools to operate on it.
- Never use `data_values` directly — the payload must be constructed from the session dataframe using `column_mapping`.
- Do not build or send a payload unless confident that all required fields are correctly mapped.
- Never assume user intent. Always confirm before taking submission or deletion actions.

Your goal is to assist with accurate and complete DHIS2 aggregate data submissions or deletions, using the tools and information available while ensuring full transparency, validation, and user confirmation.
"""








prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
openai_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

def agent_node(state: AgentState) -> AgentState:
    try:
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

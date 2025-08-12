# agents/data_entry_agent.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.agents import AgentFinish, AgentAction
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



llm = get_llm()

tools = [submit_aggregate_data, suggest_column_mapping]

system_prompt = """
You are a data entry assistant for DHIS2, specializing in aggregate data submissions.

Your responsibilities include:
- Submitting, updating, or deleting aggregate data values to/from DHIS2 using the `submit_aggregate_data` tool.

Data Source and Column Mapping:
- All data comes from a pre-uploaded dataframe stored in: `st.session_state["raw_data_df_uploaded"]`.
- This dataframe is not directly accessible to you, but tool functions can read from it.
- You are provided with the uploaded data's column names via the `dataframe_columns` argument.
- Your task is to map these user-provided column names to DHIS2’s required fields.
- Normalize common formatting issues like:
  - snake_case → camelCase (e.g., `org_unit` → `orgUnit`)
  - case differences (e.g., `Dataelement` → `dataElement`)
  - common aliases (e.g., `val` or `count` → `value`, `facility` → `orgUnit`)

Structured Instructions and Natural Language Instruction Prompt:
- The tool expects **both** `structured_instruction` and `instruction_prompt` arguments to be provided together on every call.
- `structured_instruction` is a strict JSON object describing dataframe operations, including:
  - `"operation"`: one of `"CREATE"`, `"UPDATE"`, or `"DELETE"`.
  - `"match_criteria"`: a dictionary of column-value pairs used to filter rows.
  - `"update_values"`: a dictionary of column-value pairs used to update matched rows.
  - `"specific_rows"` (optional): a list of explicit row indices (e.g., `[0,2,5]`) or row identifiers to further specify target rows.
- `instruction_prompt` is a natural language description of the intended operation, allowing the LLM to interpret user intent flexibly.
- The system first applies `structured_instruction` precisely, then uses `instruction_prompt` for validation, clarification, or handling ambiguous cases.
- Always ensure both arguments are provided to avoid incomplete or unintended data operations.

Tool Usage:
- Use `suggest_column_mapping` to infer or confirm column mappings when uncertain.
- To preview the submission, update, or deletion payload, call `submit_aggregate_data` with:
  - `preview_only=True`
  - a valid `column_mapping`
  - required `params` value: `"CREATE_AND_UPDATE"` or `"DELETE"`
  - both `structured_instruction` and `instruction_prompt` arguments.

Submission vs Deletion:
- To **submit** or **update** data, proceed only if the user clearly says: "submit", "post", "send", or "update".
  - Call `submit_aggregate_data` with `preview_only=False` and `params="CREATE_AND_UPDATE"`.
- To **delete** data, proceed only if the user clearly says: "delete", "remove", or "retract".
  - Call `submit_aggregate_data` with `preview_only=False` and `params="DELETE"`.
- For all operations, the same required fields must be present in the mapped data.

Clarify Before Acting:
- If any required field appears missing or is ambiguously mapped, do not proceed silently.
  - Ask the user for clarification or confirmation.
  - Clearly explain any assumptions or uncertainties.
- If the structured instruction or instruction prompt is ambiguous or incomplete, request clarification before proceeding.

Restrictions:
- Do not access or rely on the full dataframe yourself — only use tools to operate on it.
- Never use `data_values` directly — the payload must be constructed from the session dataframe using `column_mapping`.
- Do not build or send a payload unless confident that all required fields are correctly mapped.
- Never assume user intent. Always confirm before taking submission, update, or deletion actions.

Your goal is to assist with accurate and complete DHIS2 aggregate data submissions, updates, or deletions, using the tools and information available while ensuring full transparency, validation, and user confirmation.

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
        messages = state.get("messages", [])
        column_mapping = suggest_column_mapping.invoke({"columns": state.get("dataframe_columns", [])})
        st.session_state.suggested_columns_data_entry = column_mapping
        if column_mapping:
            column_msg = AIMessage(
                    content=f"The suggested mapped columns are: {column_mapping}"
                )

            messages.append(column_msg)
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

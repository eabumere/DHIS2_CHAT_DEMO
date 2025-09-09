# agents/tracker_data_agent.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage
from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .tools.tracker_tools import process_scanned_register
from typing import List, TypedDict, Optional, Any
from dotenv import load_dotenv
from PIL import Image
import os
from utils.llm import get_llm

load_dotenv()

class AgentState(TypedDict):
    messages: List[BaseMessage]
    output: Optional[AIMessage]
    dataframe_columns: Optional[Any]
    pdf_file: Optional[str]
    image: Optional[Image.Image]
    word_text: Optional[str]
    ppt_text: Optional[str]

llm = get_llm()

# Only keep the PDF processing tool
tools = [process_scanned_register]

system_prompt = """
You are an AI assistant that processes scanned facility registers into structured patient-level data.

Available tool:
- `process_scanned_register(pdf_file: str)`: Given the local path to a PDF file, split into pages, extract tables, and merge by 'ART No Patient ID:'.

Context:
- The uploaded PDF is already stored locally. Its path is provided in the `pdf_file` field of state.
- Do NOT ask the user to upload the PDF again.
- Always call `process_scanned_register` with the given `pdf_file`.

Workflow:
1. When the user asks to process or register patients:
   - Call `process_scanned_register` with the provided `pdf_file`.
   - Return the structured patient-level dataset.

Constraints:
- Do not fabricate data.
- Use only extracted values.
- Keep patient data private.
"""


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
openai_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def agent_node(state: AgentState) -> AgentState:
    try:
        result = openai_executor.invoke(state)  # Run agent
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
tracker_data_executor = graph.compile()

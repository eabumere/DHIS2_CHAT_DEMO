# agents/event_data_agent.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage
from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .tools.event_tools import record_event_data
from dotenv import load_dotenv
from typing import List, TypedDict
import os
from utils.llm import get_llm

load_dotenv()

class AgentState(TypedDict):
    messages: List[BaseMessage]

llm = get_llm()

tools = [record_event_data]

system_prompt = """
You manage data entry for DHIS2 event-based (non-tracker) programs.
Use the appropriate tool to:
- Record events with required fields (orgUnit, program, eventDate, etc.)
- Provide confirmation or error details from DHIS2
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
        result = openai_executor.invoke(state)
        return {"messages": result["messages"]}
    except Exception as e:
        messages = state["messages"] + [AIMessage(content=f"❌ Error: {e}")]
        return {"messages": messages}

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)
event_data_executor = graph.compile()

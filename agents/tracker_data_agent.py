# agents/tracker_data_agent.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage
from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .tools.tracker_tools import register_tracked_entity, record_followup_visit
from dotenv import load_dotenv
from typing import List, TypedDict
import os
from utils.llm import get_llm

load_dotenv()

class AgentState(TypedDict):
    messages: List[BaseMessage]

llm = get_llm()

tools = [register_tracked_entity, record_followup_visit]

system_prompt = """
You handle DHIS2 tracker data operations.
Supported operations include:
- Registering new tracked entities (e.g., pregnant women for ANC)
- Recording follow-up visits or updates for tracked entities

Always use the appropriate tool and provide accurate feedback based on DHIS2 responses.
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
tracker_data_executor = graph.compile()

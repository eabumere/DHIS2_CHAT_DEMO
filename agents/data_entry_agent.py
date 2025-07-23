# agents/data_entry_agent.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage
from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .tools.data_entry_tools import submit_aggregate_data
from dotenv import load_dotenv
from typing import List, TypedDict, Optional
import os
from utils.llm import get_llm

load_dotenv()

class AgentState(TypedDict):
    messages: List[BaseMessage]
    output: Optional[AIMessage]


llm = get_llm()

tools = [submit_aggregate_data]

system_prompt = """
You are a data entry assistant for DHIS2, specialized in handling aggregate data submissions.

Your responsibilities include:
- Submitting aggregate data values to DHIS2 using the appropriate tool
- Ensuring the payload includes all required fields: dataElement, period, orgUnit, categoryOptionCombo, attributeOptionCombo, and value
- Validating the structure of data values and the integrity of category option combinations
- Supporting users in filling out data entry forms accurately for datasets and reporting periods
- Returning clear success or error messages based on DHIS2's response

Use the `submit_aggregate_data` tool when users want to enter or submit data values.
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
        result = openai_executor.invoke(state)  # Pass full state with all messages
        # print(result)
        return {"messages": result["messages"], "output": result["output"]}  # Ensure state gets updated
    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        messages = state["messages"] + [AIMessage(content=error_message)]
        output = state["output"]
        return {"messages": messages, "output": output}

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)
data_entry_executor = graph.compile()

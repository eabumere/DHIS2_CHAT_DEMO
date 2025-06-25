# analytics_agent.py

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage
from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .tools.analytics_tools import query_analytics, get_analytics_metadata
from dotenv import load_dotenv
from typing import List, TypedDict
import os

# Load environment variables
load_dotenv()


# Define the state structure for the graph
class AgentState(TypedDict):
    messages: List[BaseMessage]


# Set up the LLM (Azure OpenAI GPT-4)
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT_4O"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    model="gpt-4",
    temperature=0,
    max_tokens=4000,
)

# Tools available to the analytics agent
tools = [query_analytics, get_analytics_metadata]

# Instruction prompt for analytics tasks
system_prompt_text = """You are an assistant specialized in querying DHIS2 analytics.

You can:
- Query analytics data by forming valid requests.
- Retrieve metadata related to analytics like dimensions and indicators.

Only call the provided tools and return actual results from DHIS2. Do not fabricate data.

Use JSON payloads when making analytics queries.

Be clear, concise, and accurate.
"""

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Define the graph node logic
def agent_node(state: AgentState) -> AgentState:
    try:
        last_message = state["messages"][-1]
        result = agent_executor.invoke({"messages": [last_message]})

        # Improved response extraction
        if isinstance(result, dict) and isinstance(result.get("output"), AIMessage):
            response = result["output"].content
        elif isinstance(result, dict):
            response = str(result.get("output", result))
        else:
            response = str(result)

        messages = state["messages"] + [AIMessage(content=response)]
        return {"messages": messages}
    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        messages = state["messages"] + [AIMessage(content=error_message)]
        return {"messages": messages}


# Build the stateful agent graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

# Compile the graph and expose it
analytics_executor = graph.compile()

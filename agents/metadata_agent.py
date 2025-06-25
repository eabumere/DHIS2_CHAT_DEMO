from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage
from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .tools.metadata_tools import create_metadata, get_dhis2_metadata, delete_metadata, update_metadata
from dotenv import load_dotenv
from typing import List, TypedDict
import os

# Load environment variables from .env
load_dotenv()

# Define state structure
class AgentState(TypedDict):
    messages: List[BaseMessage]

# --- Azure OpenAI LLM setup ---
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT_4O"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    model="gpt-4",
    temperature=0,
    max_tokens=4000,
)


# --- Agent and Tools Setup ---
tools = [create_metadata, get_dhis2_metadata, update_metadata]
enable_delete = os.getenv("ENABLE_DELETE_TOOL") == "true"
if enable_delete:
    tools.append(delete_metadata)

system_prompt_text = f"""You are a helpful assistant for interacting with DHIS2.

You support operations on DHIS2 metadata and data, including:
- Retrieving metadata (e.g., organisation units, data elements, programs)
- Creating metadata by providing a valid payload
{'- Deleting metadata when requested' if enable_delete else '- Deleting metadata is currently not supported'}
- Posting data values (e.g., tracked entities, aggregate reports)
- Querying analytics (e.g., using filters like orgUnit, period, dimensions)

Always call the relevant tool and return only the actual results retrieved from DHIS2. Never make up or simulate metadata results.

Use the appropriate tool for the user’s intent. When creating or posting data, return your response in the format:
    ({{{{"endpoint"}}}}, {{{{payload_dict}}}})

If filters or parameters are mentioned (e.g., "page 2 of org units in level 2 under Rwanda"), convert them into query parameters and pass them to the metadata tool.

When asked to find the most recent or "last created" or "last updated" metadata item, use query parameters like:
- `order=created:desc` for the most recently created item
- `order=lastUpdated:desc` for the most recently updated item
- `pageSize=1` to limit to just the latest
- `fields=id,name,created,lastUpdated` to simplify the result

Infer whether to sort by `created` or `lastUpdated` based on the user's phrasing.

Only use tools provided: `get_dhis2_metadata`, `create_metadata`, `update_metadata`{', `delete_metadata`' if enable_delete else ''}. If a request goes beyond these, inform the user.

Be concise, helpful, and accurate.
"""



# --- Prompt Template ---
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])




agent = create_openai_tools_agent(llm, tools, prompt)
openai_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Agent Node ---
def agent_node(state: AgentState) -> AgentState:
    try:
        last_message = state["messages"][-1]
        result = openai_executor.invoke({"messages": [last_message]})
        response = result.get("output", str(result))
        messages = state["messages"] + [AIMessage(content=response)]
        return {"messages": messages}
    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        messages = state["messages"] + [AIMessage(content=error_message)]
        return {"messages": messages}

# --- Graph ---
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)
compiled_graph = graph.compile()

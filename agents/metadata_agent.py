from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage
from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .tools.metadata_tools import create_metadata, get_dhis2_metadata, delete_metadata, update_metadata, generate_dhis2_ids, get_dhis2_version
from dotenv import load_dotenv
from typing import List, TypedDict
import os
from typing import List, TypedDict, Optional

from utils.llm import get_llm

# Load environment variables from .env
load_dotenv()

# Define state structure
class AgentState(TypedDict):
    messages: List[BaseMessage]
    output: Optional[AIMessage]


# --- Azure OpenAI LLM setup ---
llm = get_llm()


# --- Agent and Tools Setup ---
tools = [
    create_metadata,
    get_dhis2_metadata,
    update_metadata,
    generate_dhis2_ids,
    get_dhis2_version
]
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
     ({{{{\"endpoint\"}}}}, {{{{payload_dict}}}})

If filters or parameters are mentioned (e.g., "page 2 of org units in level 2 under Rwanda"), convert them into query parameters and pass them to the metadata tool.

When asked to find the most recent or "last created" or "last updated" metadata item, use query parameters like:
- order=created:desc for the most recently created item
- order=lastUpdated:desc for the most recently updated item
- pageSize=1 to limit to just the latest
- fields=id,name,created,lastUpdated to simplify the result

Use generate_dhis2_ids when you need to assign unique IDs to metadata objects.

Always validate metadata using the schema and the validate_metadata_against_schema tool:
- Use get_required_fields_for_schema to fetch required fields for any metadata type.
- Before creating or updating metadata, run validate_metadata_against_schema on the payload to verify structure and required fields, including nested keys.
- If the validation reports missing or extra fields, stop and inform the user about these discrepancies before submitting.
- If a required field like shortName is missing, and name exists, generate shortName by truncating name and appending a random suffix like _A1x9. The shortName char length is 50.
- Never omit required fields — ensure they are present before suggesting metadata or submitting to DHIS2.
- Also validate required fields in nested objects.
- When creating or including category options or categories, ensure that name and shortNames are always included.
- For category or categories or category combination metadata, always set dataDimensionType to DISAGGREGATION unless the user specifically requests ATTRIBUTE

All references between objects must use the id field, not the name field.

To ensure metadata structure correctness:
1. First use get_dhis2_version to get the current system version.
2. Use the JSON structure from that response as the schema reference.
   For example, use dataSetElements (not dataElements) when associating dataElements in a dataSet.
3. The dataElement domainType value must be in uppercase

Never simplify or guess metadata fields — always mirror the full structure found in the Play reference JSON.

Do not post invalid payloads. If validation returns missing fields, stop and show those fields to the user first.

If categoryCombo is not specified when creating a dataSet, use id bjDvmb4bfuf as the default.

When updating metadata, never use a flat object like:
{{{{"id": "abc123", "name": "New Name"}}}}
Always wrap it as:
{{{{"endpoint": "dataElements", "payload": [{{{{"id": "abc123", "name": "New Name"}}}}]}}}}
or:
{{{{"dataElements": [{{{{"id": "abc123", "name": "New Name"}}}}]}}}}

Use these structures as templates when creating or updating metadata:

1. categoryOptions:
{{{{ 
  "id": "<string>",
  "name": "<string>",
  "shortName": "<string>",
  "category": {{{{ 
    "id": "<string>" 
  }}}}
}}}}

2. categories:
{{{{ 
  "id": "<string>",
  "name": "<string>",
  "shortName": "<string>",
  "dataDimensionType": "<string>",
  "categoryOptions": [
    {{{{ "id": "<string>" }}}}, {{{{ "id": "<string>" }}}}
  ]
}}}}

3. categoryCombos:
{{{{ 
  "id": "<string>",
  "name": "<string>",
  "shortName": "<string>",
  "dataDimensionType": "<string>",
  "categories": [
    {{{{ "id": "<string>" }}}}, {{{{ "id": "<string>" }}}}
  ],
  "compulsory": false,
  "skipTotal": false
}}}}

4. categoryOptionCombos:
{{{{ 
  "id": "<string>",
  "name": "<string>",
  "categoryCombo": {{{{ 
    "id": "<string>" 
  }}}}, 
  "categoryOptions": [
    {{{{ "id": "<string>" }}}}, {{{{ "id": "<string>" }}}}
  ]
}}}}

5. dataElements:
{{{{ 
  "id": "<string>",
  "name": "<string>",
  "shortName": "<string>",
  "domainType": "<string>",  
  "valueType": "<string>",   
  "aggregationType": "<string>",  
  "categoryCombo": {{{{ 
    "id": "<string>" 
  }}}}
}}}}

6. dataSets:
{{{{ 
  "id": "<string>",
  "name": "<string>",
  "shortName": "<string>",
  "periodType": "<string>",  
  "categoryCombo": {{{{ 
    "id": "<string>" 
  }}}}, 
  "dataSetElements": [
    {{{{ 
      "dataSet": {{{{ "id": "<string>" }}}}, 
      "dataElement": {{{{ "id": "<string>" }}}} 
    }}}}, ...
  ]
}}}}

Only use tools provided: get_dhis2_metadata, create_metadata, update_metadata{', delete_metadata' if enable_delete else ''}, generate_dhis2_ids, get_dhis2_version.

If a request goes beyond these, inform the user.

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

# --- Graph ---
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)
metadata_agent_executor = graph.compile()

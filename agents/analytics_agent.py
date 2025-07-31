# analytics_agent.py

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage
from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .tools.analytics_tools import query_analytics, search_metadata, get_organisation_units, compute_total, compute_average, compute_max, compute_min, get_data_elements
from dotenv import load_dotenv
from typing import List, TypedDict, Optional
import os
from utils.llm import get_llm
# Load environment variables
load_dotenv()


# Define the state structure for the graph
class AgentState(TypedDict):
    messages: List[BaseMessage]
    output: Optional[AIMessage]
    raw_data: dict  # or: Any, if it can vary
    metadata_result: Optional[dict]  # For search_metadata tool result
    selected_metadata_id: Optional[str]  # Explicit metadata id to skip re-query



# Set up the LLM (Azure OpenAI GPT-4)
llm = get_llm()

# Tools available to the analytics agent
tools = [query_analytics, search_metadata, get_organisation_units, compute_total, compute_average, compute_max, compute_min, get_data_elements]

# Instruction prompt for analytics tasks
system_prompt_text = """
You are a DHIS2 analytics assistant. Your job is to query DHIS2 correctly using the provided tools.

Your main goals:
- Interpret user queries.
- Identify the correct metadata (e.g., indicator or data element).
- Use that metadata ID to fetch analytics data via the tools provided.
- Respond clearly and accurately with user-friendly summaries of the result.

---

### Metadata Lookup Rules

- If the user gives natural terms (e.g., "maternal deaths") or user gives CODE (e.g., "HTS_TST"), you should call `search_metadata` to find the best match.
- However, **if `metadata_result.selected` or `selected_metadata_id` is provided**, **you must skip** calling `search_metadata`.
- In this case, **treat `metadata_result.selected` as final** and use its `id` directly as input to `query_analytics`.
- Do not attempt to re-infer or guess a different metadata match. Assume it is confirmed by the user.
- If multiple metadata matches are returned by `search_metadata`, allow the user (via UI) to choose — and then re-run the flow with `metadata_result.selected` set.

---

### Org Unit Logic

- If the user provides a location like "Bo District" or "national", call `get_organisation_units` to look up the correct orgUnit ID.
- If no location is given, default to the root org unit (i.e., level 1 in DHIS2).

---

### Analytics Queries

- Use `query_analytics` with these required inputs:
  - `indicators`: List of IDs (e.g., from selected metadata)
  - `doc_type`: Must be set to `"indicator"`, `"dataElement"`, or `"programIndicator"` based on the metadata result
  - `periods`: Must be derived from temporal context or default to `LAST_12_MONTHS`
  - `org_units`: Based on resolved location or fallback to root
  - `disaggregations`: A list of disaggregation terms or categories based on user intent (see logic below)

#### Disaggregation Handling

- If the metadata has **category combinations** (disaggregations), inspect the user query to determine how to populate the `disaggregations` field.

Use the following logic to populate the `disaggregations` list:

1. **Specific Category Options Mentioned**  
   If the user mentions groups such as:  
   > "MSM", "FSW", "Female", "Transgender", etc.  
   → Return a list of the **exact strings** mentioned:  
   `["MSM", "FSW"]`

2. **Disaggregation Category Mentioned**  
   If the user says:  
   > "Break down by sex", "Group by age", "Disaggregate by population"  
   → Return the **category name(s)** as a list:  
   `["Sex", "Age Group", "All population"]`

3. **All Disaggregations Requested**  
   If the user explicitly requests:  
   > "Include all disaggregations"  
   → Return:  
   `["all"]`

4. **No Disaggregations Mentioned**  
   If there’s no indication of disaggregation intent:  
   → Return:  
   `["None"]`

Do **not infer disaggregations** unless the user clearly requests them. Only extract what is present in the prompt.

---

### Output Guidelines

- If the user asks for a **total**, respond clearly with the **summed value** (e.g., “There were 123 maternal deaths in the past 12 months.”).
- If they request a **trend** or **chart**, structure the output to support that (e.g., JSON timeseries).
- Respond in **natural, friendly language first**, and optionally follow with data or tables if needed.
- Keep answers concise, factual, and relevant unless the user asks for more detail or raw output.

---

### DO NOT:

- Do not call `search_metadata` if `metadata_result.selected` or `selected_metadata_id` is present.
- Do not fabricate IDs or values. Only use values returned by tools.
- Do not guess metadata or orgUnit IDs from names — use tool lookups instead.
- Do not hardcode disaggregation IDs. Always extract them from the metadata.

---

### Special Cases

- If the user asks for a total, average, max, or min:
  - Use: `compute_total`, `compute_average`, `compute_max`, or `compute_min`.
  - Do not manually calculate these in Python or LLM.

Your job is to act like an intelligent bridge between human-friendly input and machine-structured data tools.
"""

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)


# # Define the graph node logic
# def agent_node(state: AgentState) -> AgentState:
#     try:
#         last_message = state["messages"][-1]
#         result = agent_executor.invoke({"messages": [last_message]})
#
#         # Improved response extraction
#         if isinstance(result, dict) and isinstance(result.get("output"), AIMessage):
#             response = result["output"].content
#         elif isinstance(result, dict):
#             response = str(result.get("output", result))
#         else:
#             response = str(result)
#
#         messages = state["messages"] + [AIMessage(content=response)]
#         return {"messages": messages}
#     except Exception as e:
#         error_message = f"❌ Error: {str(e)}"
#         messages = state["messages"] + [AIMessage(content=error_message)]
#         return {"messages": messages}





# def agent_node(state: AgentState) -> AgentState:
#     try:
#         result = agent_executor.invoke(state)
#
#         # 🌟 Ensure response is returned in the "output" key
#         if isinstance(result, dict) and "output" in result:
#             response_msg = result["output"]
#         elif isinstance(result, dict) and "messages" in result:
#             last_msg = result["messages"][-1]
#             response_msg = (
#                 last_msg if isinstance(last_msg, AIMessage)
#                 else AIMessage(content=str(last_msg))
#             )
#         else:
#             response_msg = AIMessage(content="⚠️ No response generated by the agent.")
#
#         # ✅ Match the metadata executor return signature
#         return {
#             "messages": state["messages"] + [response_msg],
#             "output": response_msg
#         }
#
#     except Exception as e:
#         error_msg = AIMessage(content=f"❌ Error: {e}")
#         return {
#             "messages": state["messages"] + [error_msg],
#             "output": error_msg
#         }



def agent_node(state: AgentState) -> AgentState:
    try:
        result = agent_executor.invoke(state)

        # Extract latest message to show to the user
        if isinstance(result, dict):
            response_msg = None
            tool_output = None
            metadata_tool_result = None  # ✅ initialize safely


            # Final response message
            if "output" in result:
                response_msg = result["output"]
            elif "messages" in result:
                last_msg = result["messages"][-1]
                response_msg = (
                    last_msg if isinstance(last_msg, AIMessage)
                    else AIMessage(content=str(last_msg))
                )

            # Capture tool output if available
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if isinstance(step, tuple):
                        if step[0].tool == "query_analytics":
                            tool_output = step[1]
                        elif step[0].tool == "search_metadata":
                            metadata_tool_result = step[1]

            # Add both message and raw tool output to state
            # return {
            #     "messages": state["messages"] + [response_msg],
            #     "output": {
            #         "message": response_msg,
            #         "tool_data": tool_output  # 💡 You can now access full dict later
            #     }
            # }
            # print(tool_output)
            return {
                "messages": state["messages"] + [response_msg],
                "output": response_msg,
                "raw_data": tool_output,
                "metadata_result": metadata_tool_result  # <-- Include structured search_metadata result

            }

        else:
            return {
                "messages": state["messages"] + [AIMessage(content="⚠️ Invalid result format.")],
                "output": None
            }

    except Exception as e:
        error_msg = AIMessage(content=f"❌ Error: {e}")
        return {
            "messages": state["messages"] + [error_msg],
            "output": error_msg
        }


# Build the stateful agent graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

# Compile the graph and expose it
analytics_executor = graph.compile()


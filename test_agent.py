from langchain_core.messages import HumanMessage
from agent import compiled_graph  # Ensure this is the compiled LangGraph from agent.py

# Sample test prompt: Creating two data elements
test_prompt = """
Create two data elements.
The first one is called AbumereDE1TestUnique with code AbuDE1CODE123, value type TEXT, and domain type AGGREGATE.
The second one is called AbumereDE2TestUnique with code AbuDE2CODE123, value type NUMBER, and domain type AGGREGATE.
"""

# Construct initial state with the user message
initial_state = {
    "messages": [HumanMessage(content=test_prompt)]
}

# Run the agent graph
print("Running agent...\n")
final_state = compiled_graph.invoke(initial_state)

# Extract and print the assistant's final message
final_response = final_state["messages"][-1]
print("\n--- Final Agent Response ---\n")
print(final_response.content)

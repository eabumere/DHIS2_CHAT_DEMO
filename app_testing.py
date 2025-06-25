import os
from langchain_core.messages import HumanMessage, AIMessage
from multi_agent import routing_decision, metadata_agent_executor, analytics_executor, get_session_history


def test_message_routing(user_input: str):
    print(f"\nUser input:\n{user_input}\n{'-' * 40}")

    # Build state with last user message
    state = {"messages": [HumanMessage(content=user_input)]}

    # Get routing decision
    route = routing_decision(state)
    print(f"Routing decision: {route}")

    # Call the chosen executor
    if route == "metadata":
        result = metadata_agent_executor.invoke({"messages": state["messages"]})
    elif route == "analytics":
        result = analytics_executor.invoke({"messages": state["messages"]})
    else:
        print("Unknown routing result")
        return

    output = result.get("output")
    if isinstance(output, AIMessage):
        print("Agent response:")
        print(output.content)
    else:
        print("Agent response (raw):")
        print(output)


if __name__ == "__main__":
    # Test cases:
    test_cases = [
        "Create Data Element with name AbumereDE1TestUnique with code AbuDE1CODE123 and shortname AbumereDE1TestUnique, value type INTEGER, and domain type AGGREGATE.",
        "Show me analytics data value summaries for last quarter.",
        "List level 2 organisation units available in the system.",
        "Create a new organisation unit named TestingTheMic with code TEST_MIC, short name TestMic, opening date 2023-01-01, and assign it under parent unit with ID nLbABkQlwaT.",
        "How are you today?",
    ]

    for case in test_cases:
        test_message_routing(case)

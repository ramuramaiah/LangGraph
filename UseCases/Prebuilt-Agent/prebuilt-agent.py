import operator
from typing import Annotated, List

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

if __name__ == "__main__":
    load_dotenv()

    print("Hello LangGraph")

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

    # Augment the LLM with tools
    tools = [add, multiply, divide]
    # Pass in:
    # (1) the augmented LLM with tools
    # (2) the tools list (which is used to create the tool node)
    pre_built_agent = create_react_agent(llm, tools=tools)

    # Show the agent
    display(Image(pre_built_agent.get_graph().draw_mermaid_png()))

    # Invoke
    messages = [HumanMessage(content="Add 3 and 4."),
                HumanMessage(content="Multiply 3 and 4."),
                HumanMessage(content="Divide 4 by 2.")
                ]
    
    agent_messages = pre_built_agent.invoke({"messages": messages})
    for m in agent_messages["messages"]:
        m.pretty_print()
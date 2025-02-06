from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# Schema for structured output to use as routing logic
class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        None, description="The next step in the routing process"
    )


# State
class State(TypedDict):
    input: str
    decision: str
    output: str


if __name__ == "__main__":
    load_dotenv()

    print("Hello LangGraph")

    # Nodes
    def llm_call_1(state: State):
        """Write a story"""

        result = llm.invoke(state["input"])
        return {"output": result.content}


    def llm_call_2(state: State):
        """Write a joke"""

        result = llm.invoke(state["input"])
        return {"output": result.content}


    def llm_call_3(state: State):
        """Write a poem"""

        result = llm.invoke(state["input"])
        return {"output": result.content}


    def llm_call_router(state: State):
        """Route the input to the appropriate node"""

        # Run the augmented LLM with structured output to serve as routing logic
        decision = router.invoke(
            [
                SystemMessage(
                    content="Route the input to story, joke, or poem based on the user's request."
                ),
                HumanMessage(content=state["input"]),
            ]
        )

        return {"decision": decision.step}


    # Conditional edge function to route to the appropriate node
    def route_decision(state: State):
        # Return the node name you want to visit next
        if state["decision"] == "story":
            return "llm_call_1"
        elif state["decision"] == "joke":
            return "llm_call_2"
        elif state["decision"] == "poem":
            return "llm_call_3"

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

    # Augment the LLM with schema for structured output
    router = llm.with_structured_output(Route)

    # Build workflow
    router_builder = StateGraph(State)

    # Add nodes
    router_builder.add_node("llm_call_1", llm_call_1)
    router_builder.add_node("llm_call_2", llm_call_2)
    router_builder.add_node("llm_call_3", llm_call_3)
    router_builder.add_node("llm_call_router", llm_call_router)

    # Add edges to connect nodes
    router_builder.add_edge(START, "llm_call_router")
    router_builder.add_conditional_edges(
        "llm_call_router",
        route_decision,
        {  # Name returned by route_decision : Name of next node to visit
            "llm_call_1": "llm_call_1",
            "llm_call_2": "llm_call_2",
            "llm_call_3": "llm_call_3",
        },
    )
    router_builder.add_edge("llm_call_1", END)
    router_builder.add_edge("llm_call_2", END)
    router_builder.add_edge("llm_call_3", END)

    # Compile workflow
    router_workflow = router_builder.compile()

    # Show the workflow
    display(Image(router_workflow.get_graph().draw_mermaid_png()))

    # Invoke
    state = router_workflow.invoke({"input": "Write me a poem about cats"})
    print(state["output"])
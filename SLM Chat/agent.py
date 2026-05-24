
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import tools

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOllama(model="llama3.2", temperature=0).bind_tools(tools)

def chatbot_node(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def route_after_chatbot(state: State) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", route_after_chatbot, {"tools": "tools", "__end__": END})
workflow.add_edge("tools", "chatbot")

agent_app = workflow.compile()



from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage
)
from pathlib import Path

from langchain.tools import tool

from langgraph.checkpoint.memory import InMemorySaver

from langgraph.errors import GraphRecursionError

from rag import KNOWLEDGE_DIR, retrieve



load_dotenv()

model=ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

@tool
def search_knowledge(question:str)->str:
    """搜索 AI 学习助手的本地知识库。"""
    return retrieve(question)


@tool
def list_knowledge_sources() -> list[str]:
    """查看当前知识库中有哪些文件。"""
    
    return [
        file_path.name
        for file_path in KNOWLEDGE_DIR.glob("*.txt")
    ]


def chatbot(state:MessagesState):
    response=model_with_tools.invoke(
        state["messages"]
    )
    return{
        "messages":[response]
    }

tools=[
    search_knowledge,
    list_knowledge_sources
]

model_with_tools=model.bind_tools(tools)

builder=StateGraph(MessagesState)

builder.add_node(
    "chatbot",
    chatbot
)

builder.add_node(
    "tools",
    ToolNode(tools)
)

builder.add_edge(
    START,
    "chatbot"
)

builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

builder.add_edge(
    "tools",
    "chatbot"
)

memory=InMemorySaver()

graph=builder.compile(
    checkpointer=memory
)

config={
        "configurable": {
            "thread_id": "test-normal"
        },
        "recursion_limit":10
    }

try:
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="1+1等于多少？"
                )
            ]
        },
        config=config
    )

    print(result["messages"][-1].content)

except GraphRecursionError:
    print("Agent 执行步骤过多，已停止")



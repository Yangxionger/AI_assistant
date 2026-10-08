import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage
)

from langchain.tools import tool


load_dotenv()

model=ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


@tool
def search_knowledge(question:str)->str:
    """搜索 AI 学习助手的本地知识库。"""
    return f"模拟搜索结果：{question}"

messages = [
    HumanMessage(
        content="请从知识库查询什么是 RAG"
    )
]

model_with_tools=model.bind_tools(
    [search_knowledge]
)

response = model_with_tools.invoke(
    messages
)
messages.append(response)

if response.tool_calls:
    tool_call=response.tool_calls[0]


tool_result=search_knowledge.invoke(
    tool_call["args"]
)

tool_message=ToolMessage(
    content=tool_result,
    tool_call_id=tool_call["id"]
)

messages.append(tool_message)

final_response=model_with_tools.invoke(
    messages
)

print(final_response.content)


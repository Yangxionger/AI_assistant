from llm import chat_with_tools
from rag import retrieve, list_knowledge_sources
import json

import logging

logger=logging.getLogger(__name__)

def list_knowledge_source():
    return list_knowledge_sources()

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索 AI 学习助手的本地知识库",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "需要在知识库中搜索的问题"
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_knowledge_sources",
            "description": "查看 AI 学习助手知识库中有哪些文件",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

tool_functions = {
    "search_knowledge": retrieve,
    "list_knowledge_sources": list_knowledge_source
}

def execute_tool(tool_call):
    tool_name=tool_call.function.name

    try:
        arguments=json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return f"工具 {tool_name} 的参数格式错误"

    if tool_name not in tool_functions:
        return f"不存在工具：{tool_name}"

    tool_function = tool_functions[tool_name]
    tool_result = tool_function(**arguments)

    if isinstance(tool_result, str):
        tool_content = tool_result
    else:
        tool_content = json.dumps(
            tool_result,
            ensure_ascii=False
        )
    return tool_content


def run_agent(user_input, messages=None,max_steps=5):
    if messages is None:
        messages=[]
    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )
    logger.info("Agent started")
        
    for step in range(max_steps):
        logger.info("Agent step: %s", step + 1)

        message = chat_with_tools(
            messages=messages,
            tools=tools
        )

        if not message.tool_calls:
            logger.info("Agent finished without more tool calls")
            messages.append(message)
            return message.content,messages

        messages.append(message)

        for tool_call in message.tool_calls:
            logger.info(
                "Calling tool: %s",
                tool_call.function.name
            )
            
            tool_content=execute_tool(tool_call)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_content
            })

    logger.warning("Agent reached max_steps")

    return "Agent 达到最大执行步数，任务未完成。",messages


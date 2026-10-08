import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / ".env"
# Keep the legacy configuration working until a project .env is created.
if not ENV_FILE.exists():
    ENV_FILE = ROOT_DIR.parents[1] / ".env"
load_dotenv(ENV_FILE)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def chat_with_llm(question: str, system_prompt: str) -> str:

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ],
        stream=False
    )

    return response.choices[0].message.content

def chat_with_tools(messages, tools):
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        tools=tools,
        stream=False
    )

    return response.choices[0].message

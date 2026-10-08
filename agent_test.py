from agent import run_agent

history=[]
answer,history= run_agent(
    "我的知识库里有哪些文件？",
    messages=history
)
print(answer)
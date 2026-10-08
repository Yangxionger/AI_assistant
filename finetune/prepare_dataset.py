from datasets import load_dataset
import random

dataset = load_dataset(
    "m-a-p/COIG-CQIA",
    "segmentfault",
    split="train"
)

print("数据数量：", len(dataset))
print("字段：", dataset.column_names)

print("\n第一条数据：")
print(dataset[0])

from collections import Counter


print("\nanswer_from 分布：")
print(
    Counter(dataset["answer_from"])
)

print("\nhuman_verified 分布：")
print(
    Counter(dataset["human_verified"])
)

print("总数据量：", len(dataset))



major_types = []
minor_types = []

for item in dataset:
    task_type = item["task_type"]

    major_types.extend(
        task_type.get("major", [])
    )

    minor_types.extend(
        task_type.get("minor", [])
    )

print("\n===== task_type major =====")
print(Counter(major_types))

print("\n===== task_type minor =====")
print(Counter(minor_types))

domains = []

for item in dataset:
    domain = item["domain"]

    domains.extend(domain)

print("\n===== domain =====")
print(Counter(domains))

print("\n===== 随机抽样 10 条 =====")

for index in random.sample(range(len(dataset)), 10):
    item = dataset[index]

    print("\n--------------------")
    print("编号：", index)
    print("领域：", item["domain"])
    print("问题：", item["instruction"])
    print("回答：", item["output"][:500])


output_lengths = [
    len(item["output"])
    for item in dataset
]

print("\n===== 回答长度 =====")
print("最短回答长度：", min(output_lengths))
print("最长回答长度：", max(output_lengths))
print(
    "平均回答长度：",
    round(sum(output_lengths) / len(output_lengths), 2)
)

questions = [
    item["instruction"].strip()
    for item in dataset
]

print("\n===== 重复问题 =====")
print("原始数量：", len(questions))
print("去重后数量：", len(set(questions)))
print(
    "重复数量：",
    len(questions) - len(set(questions))
)

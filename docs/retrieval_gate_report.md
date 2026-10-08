# Retrieval gate v1 验收报告

先测量，后接入 gate。当前 5 份原始资料、16 个 points；22 条有答案问题、22 条无答案问题。未修改 finetune，没有 Web Search。

| 组别 | 总题数 | 有分数 | min | mean | median | max |
|---|---:|---:|---:|---:|---:|---:|
| positive | 22 | 21 | 0.003065 | 0.818223 | 0.942226 | 0.999617 |
| negative | 22 | 11 | 0.001560 | 0.075015 | 0.030604 | 0.474124 |

无候选分数记 null，不混入均值；positive 1 条、negative 11 条无候选。分数区间重叠 [0.003065, 0.474124]。该区间是 min/max 区间交集，不代表每一个值都有样本。

## 阈值与误差

选 0.5，高于实测无答案问题最大分数 0.474124。0.4 会误放行 Git 历史密钥清理问题；0.5 与 0.6 的校准集判断相同，0.7 会增加有答案问题拒绝数量。因此采用 0.5，优先降低错误放行风险。不是通用阈值，不能解释为 50% 有答案。

本次环境 CrossEncoder.predict 使用 Sigmoid，记录的是原有 predict 输出，而非 logits。实现没有改变激活函数或分数尺度。

按知识库是否有答案标签计算：FP=0/22，FN=4/22，正例放行18/22。FN：p04(JSON约束)、p09(commit未push)、p10(Git三阶段)、p12(功能隔离合并)。p09 和 p12 在原向量检索阶段已未检索到正确 Git 资料；p04 和 p10 是保守阈值新增拒绝。

人工核对 Top-3 原文：19/22 正例有对应答案证据。p13 的状态码问题在 fastapi.txt 中也有正确答案，不能只限定 http.txt。p14(外部API异常为何成为500) 得分0.9584，但检索内容仅解释一般500状态码，没有该问题的异常处理链，因此被错误放行。这是另一个“命中结果是否含答案”的误判，不能用 FP=0 掩盖。阈值只能过滤低相关结果，不能证明高分结果完整回答问题。

以上是同一小型校准集上的结果，不是独立测试集泛化性能；上传更多文档、修改 chunk、embedding 或 reranker 后应重新评估。

## 实现

rag.py 新增 retrieve_with_confidence；最高分<0.5 或无候选时 hits=[] / no_hit。retrieve 保持列表接口并返回门控后的 hits；_retrieve 保留原始检索供离线评估。ask_with_rag 在 no_hit 分支直接返回固定提示，不调用 LLM。

main.py 的 RagResponse 和 ask 新增 retrieval_status，内部 top_score 不暴露到正式 API。普通 Agent 和 LangGraph 的检索工具仍拿 list[dict]，低分返回空列表，不改变图结构。/agent 的整体决策仍由其原有模型控制，本轮未改成 /ask 的固定拒答策略。

## 验收

4 组真实测试全部通过，耗时28.422秒。知识库内问题和同义改写 hit；完全无关问题及困难无答案问题 no_hit；no_hit sources=[]、context为空、LLM调用为0。TXT/MD/PDF 上传、重复上传、同名更新、失败回滚、PDF页码、引用去重以及两种 Agent 工具均通过。测试资料清理完毕，原始知识库保留，finetune大小和mtime检查不变。

题目和逐题结果：evaluation/retrieval_eval_set.json、evaluation/retrieval_gate_baseline.json/.md。验收：docs/retrieval_gate_test_result.json、docs/retrieval_gate_acceptance_console.log。

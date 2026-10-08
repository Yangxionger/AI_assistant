# 第一轮项目整理结果

本轮只整理展示文档、配置模板、Git 忽略规则及确认未使用的入口内容。没有重构 RAG/LangGraph/Web Search，没有修改 finetune 实验结果，也没有执行 git add/commit/push。

## 修改与归档

- 重写 README.md：项目问题与功能、架构、入库/问答流程、Local/Web/None、技术栈、目录、安装与配置、启动、接口差异、PDF、gate 校准、Tavily、sources、独立 QLoRA 实验、验收与边界。
- 新建 .env.example：只有公开配置示例及密钥占位符，没有读取复制真实 Key。
- 完善 .gitignore：忽略 .env、上传、数据库、状态、work、虚拟环境、模型缓存/权重、checkpoint、finetune/output、训练语料/备份、日志和临时文件。保留 .env.example、代码、五份示例知识、PDF fixtures、评估题及精选 JSON/Markdown 报告。
- main.py 只移除未使用的 run_agent import 和 agent_histories，AST 检查没有实际读取引用。
- 将 langchain_test.py 原样移动到 examples/legacy/langchain_test.py，移动前后 SHA256 相同。
- agent.py、agent_test.py、langgraph_test.py 暂保留：旧 Agent 仍有验收测试依赖；两个演示依赖根目录模块，直接移动会破坏导入。本轮不改它们的运行方式。
- requirements.txt 已检查，覆盖当前后端；保持不变。requirements-qlora.txt 仍为空，不随意锁定实验依赖，README 给出独立环境补齐建议。

没有直接删除历史文件、数据、缓存或实验结果。Git 原有 knowledge.txt 删除状态不是本轮操作；已有 llm/rag/requirements 等未提交修改也完整保留。

## 回归结果

真实 embedding、reranker、Qdrant、DeepSeek、Tavily 回归，共 6 个 unittest 测试，0 failures / 0 errors：

| 检查 | 结果 |
|---|---|
| TXT/MD 上传、检索、重复、更新与校验 | passed |
| 文字 PDF、页码、重复、异常及失败回滚 | passed |
| 本地命中、同义问法、gate/no_hit | passed |
| /agent local | hit/local；Web 调用 0 次 |
| /agent Web | no_hit/web；真实 Tavily HTTP 成功，来源来自 metadata |
| /agent none | no_hit/none；sources=[]，明确无可靠资料提示 |
| 本地引用去重、TXT/MD 无页码、Agent metadata | passed |
| Tavily provider 数据结构控制测试 | passed |
| 原知识库与 finetune 保留检查 | passed |
| threshold | 保持 0.5 |
| TLS/SSL EOF | 本轮 0 次 |

详见 docs/cleanup_v1_regression_result.json。控制测试不会替代真实 Web 验收。本轮使用 TestClient 直接调用应用，开始测试时没有运行中的 uvicorn，不需要停止或重启已有后端；未把服务留在后台。

## 建议提交范围（仅列表，没有暂存）

- README.md、.env.example、.gitignore、requirements.txt。
- main.py、rag.py、llm.py、langgraph_agent.py、web_search.py。
- agent.py、agent_test.py、langgraph_test.py、examples/legacy/langchain_test.py：明确为历史演示/兼容测试资产。
- knowledge/fastapi.txt、git.txt、http.txt、python.txt、rag.txt。
- tests 下的源代码与 tests/fixtures/text_learning.pdf、image_only.pdf。
- evaluation/retrieval_eval_set.json、retrieval_gate_baseline.json、retrieval_gate_baseline.md。
- docs/cleanup_v1_report.md、cleanup_v1_regression_result.json、tavily_live_acceptance_result.json、retrieval_gate_report.md，以及按展示需要选取的上传/PDF/来源验收 JSON。历史阶段报告不代表当前路由，不建议一次提交全部失败尝试产物。
- finetune 顶层准备/训练/评估源码、configs、formal_v1_comparison.json/.md、3B 与 concise 的精选回答和评估摘要。保留为独立实验，不提交 adapter、checkpoint、训练语料、数据备份和日志。
- requirements-qlora.txt 当前为空，不将其作为可用安装清单；待基于验证环境补齐后再用于复现说明。

发布时还需明确 knowledge.txt 原有迁移是否应记录；本轮不操作其 Git 状态。.gitignore 不会取消已跟踪文件，也不删除本地文件。

## 仍然保留的边界

单用户、单进程 Qdrant、内存 LangGraph checkpoint；没有 OCR、多用户认证、Docker、云部署或前端。检索分数与 Web 摘要不能保证完整答案；本地引用是参考资料，LLM 允许知识补充。TXT/MD 的失败恢复不等同 PDF 补偿回滚。后端依赖尚未完整锁版本，QLoRA 安装清单待补齐。

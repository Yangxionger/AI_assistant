# Retrieval gate calibration

No LLM evaluation. Evidence is checked using source and hand-authored answer terms.

Missing candidates have null scores; they are not counted as zero.

Statistics: {"positive": {"scored_count": 21, "min": 0.0030650540720671415, "mean": 0.818223336523044, "median": 0.9422255754470825, "max": 0.9996167421340942}, "negative": {"scored_count": 11, "min": 0.0015603185165673494, "mean": 0.07501485342667862, "median": 0.030603770166635513, "max": 0.474124014377594}, "overlap": [0.0030650540720671415, 0.474124014377594], "recommended_threshold": 0.5, "false_positive_ids": [], "false_negative_ids": ["p04", "p09", "p10", "p12"]}

| ID | label | question | top1 | source | evidence in Top-3 |
|---|---|---|---|---|---|
| p01 | positive | 浏览器页面和接口端口不同，被拦住了请求，FastAPI里要配置什么？ | 0.9658381938934326 | fastapi.txt | True |
| p02 | positive | 我不想手写接口说明，FastAPI能自动提供什么文档工具？ | 0.9370375275611877 | fastapi.txt | True |
| p03 | positive | 提交的JSON不符合请求模型，为什么会出现422？ | 0.9861582517623901 | fastapi.txt | True |
| p04 | positive | 后端怎样约束用户提交的JSON结构？ | 0.011805185116827488 | fastapi.txt | True |
| p05 | positive | 在FastAPI中怎么声明一个读取接口和一个提交接口？ | 0.8863531947135925 | fastapi.txt | True |
| p06 | positive | 前端的问题怎样经FastAPI交给大模型并拿到结果？ | 0.9951978325843811 | fastapi.txt | True |
| p07 | positive | 文本切块过大和过小分别会有什么检索问题？ | 0.9159353971481323 | fastapi.txt | True |
| p08 | positive | 为什么知识库文档不是整篇生成一个向量，而是拆成小块？ | 0.9945927262306213 | fastapi.txt | True |
| p09 | positive | 代码已经commit但网络断了没push，本地历史还在吗？ | 0.0030650540720671415 | fastapi.txt | False |
| p10 | positive | git add、commit、push各自操作哪个阶段？ | 0.3831051290035248 | git.txt | True |
| p11 | positive | 怎么只提交这个功能的改动，让提交记录更清楚？ | 0.9414587020874023 | git.txt | True |
| p12 | positive | 开发新功能时如何隔离改动，完成后怎么合回去？ | None | None | False |
| p13 | positive | 接口返回200、404和500分别意味着什么？ | 0.9978581070899963 | fastapi.txt | True |
| p14 | positive | 后端请求外部服务失败，为什么浏览器最后看到的是500？ | 0.9584000110626221 | fastapi.txt | False |
| p15 | positive | 查资源和提交数据一般分别选哪种HTTP方法？ | 0.6961559653282166 | http.txt | True |
| p16 | positive | 想在Python列表的末尾增加一个元素，要怎么做？ | 0.9813374280929565 | python.txt | True |
| p17 | positive | Python存名字和年龄这种对应关系，适合用什么结构？ | 0.6832443475723267 | python.txt | True |
| p18 | positive | 读取一个UTF-8文本文件，pathlib有什么方便方法？ | 0.9988924860954285 | python.txt | True |
| p19 | positive | 文件不存在时怎样避免Python程序直接崩掉？ | 0.964408814907074 | python.txt | True |
| p20 | positive | RAG为什么要在生成答案之前找外部资料？ | 0.9422255754470825 | rag.txt | True |
| p21 | positive | 检索里Top-K和threshold分别控制什么？ | 0.9400033950805664 | rag.txt | True |
| p22 | positive | FAISS里的IndexFlatIP和HNSW在搜索方式上有何区别？ | 0.9996167421340942 | rag.txt | True |
| n01 | negative | MySQL发生死锁时如何定位事务并解决锁顺序问题？ | None | None | False |
| n02 | negative | Linux inode耗尽而磁盘还有空间时应该怎么排查？ | None | None | False |
| n03 | negative | 如何计算TCP拥塞窗口在慢启动中的变化？ | None | None | False |
| n04 | negative | Java垃圾收集器G1如何决定Mixed GC的触发时间？ | None | None | False |
| n05 | negative | C++的移动构造函数如何配合右值引用减少拷贝？ | None | None | False |
| n06 | negative | CSS Grid如何让一个元素跨三列并响应式布局？ | None | None | False |
| n07 | negative | 操作系统如何避免银行家算法中的不安全状态？ | None | None | False |
| n08 | negative | 红黑树删除节点后有哪些旋转和重新染色情形？ | None | None | False |
| n09 | negative | 如何用PyTorch实现Transformer多头注意力？ | None | None | False |
| n10 | negative | 训练QLoRA时NF4量化误差如何计算？ | 0.004393762908875942 | fastapi.txt | False |
| n11 | negative | Git rebase冲突时如何使用reflog恢复丢失提交？ | 0.13705956935882568 | git.txt | False |
| n12 | negative | FastAPI如何实现OAuth2刷新令牌轮换与撤销？ | 0.055499061942100525 | fastapi.txt | False |
| n13 | negative | Python装饰器怎样保留闭包变量并处理nonlocal？ | 0.0016542335506528616 | python.txt | False |
| n14 | negative | HTTP/2的HPACK动态表如何编码重复请求头？ | 0.011714889667928219 | fastapi.txt | False |
| n15 | negative | RAG怎样训练ColBERT晚期交互检索模型？ | 0.030603770166635513 | rag.txt | False |
| n16 | negative | Python asyncio任务取消如何传播CancelledError？ | 0.0015603185165673494 | python.txt | False |
| n17 | negative | FastAPI上传文件时怎样进行病毒检测和恶意PDF隔离？ | 0.0614093542098999 | fastapi.txt | False |
| n18 | negative | 请给出HNSW的M和efConstruction在百万向量下的最优参数及内存估算。 | 0.038629282265901566 | rag.txt | False |
| n19 | negative | 怎么给HTTP接口设计支付幂等键以及跨库事务补偿？ | 0.0085151307284832 | fastapi.txt | False |
| n20 | negative | 怎么把Git仓库历史里的泄漏密钥彻底移除？ | 0.474124014377594 | git.txt | False |
| n21 | negative | 量子纠缠实验怎样检验贝尔不等式？ | None | None | False |
| n22 | negative | 做戚风蛋糕时蛋白打发不足该怎么补救？ | None | None | False |

Human review: p13 has valid evidence in fastapi.txt (alternative source). p14 is an accepted but incomplete retrieval despite its high score. Answer evidence missing: p09, p12, p14.

Threshold sweep:
[
  {
    "threshold": 0.1,
    "false_positive_count": 2,
    "false_negative_count": 3
  },
  {
    "threshold": 0.3,
    "false_positive_count": 1,
    "false_negative_count": 3
  },
  {
    "threshold": 0.4,
    "false_positive_count": 1,
    "false_negative_count": 4
  },
  {
    "threshold": 0.5,
    "false_positive_count": 0,
    "false_negative_count": 4
  },
  {
    "threshold": 0.6,
    "false_positive_count": 0,
    "false_negative_count": 4
  },
  {
    "threshold": 0.7,
    "false_positive_count": 0,
    "false_negative_count": 6
  },
  {
    "threshold": 0.9,
    "false_positive_count": 0,
    "false_negative_count": 7
  },
  {
    "threshold": 0.96,
    "false_positive_count": 0,
    "false_negative_count": 13
  }
]

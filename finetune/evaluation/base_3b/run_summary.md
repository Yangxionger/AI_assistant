# Qwen2.5-3B Base 下载、硬件与评估报告

完整下载，两个 safetensors 分片均通过官方 SHA-256 校验。tokenizer 正常；4-bit NF4、bfloat16 compute、device_map=auto；实际参数与 buffer 均在 cuda:0；CUDA 正常，无 OOM。

模型快照：`D:\Code\huggingface_cache\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots\aa8e72537993ba99e69dfaafa59ed015b17504d1`

HF_HOME：`D:\Code\huggingface_cache`；HF_HUB_CACHE：`D:\Code\huggingface_cache\hub`。用户环境变量已持久设置；本次进程使用 D 盘缓存。

| 项目 | 结果 |
|---|---|
| 完整模型快照文件大小 | 6,183,451,098 bytes（5.759 GiB） |
| 3B 缓存目录总大小，含保留的续传文件 | 11,706,004,142 bytes（10.902 GiB） |
| 其中残片与续传文件 | 5.143 GiB |
| 成功生成 | 18/18 |
| 平均生成 tokens（含停止 token，如有） | 512 |
| 平均字符数 | 969.56 |
| 达到 512 token 上限 | 18/18 |
| 空回答 | 0 |
| 整卡峰值（nvidia-smi，每秒采样，含桌面等占用） | 3728 MiB（3.641 GiB） |
| PyTorch 峰值 allocated / reserved | 2161.16 / 2222.0 MiB |
| 总推理时间 | 406.28 秒 |
| 评估时间，含加载 | 415.69 秒 |

复用 v1 的相同 18 题；逐题验证 prompt 文本和 token IDs 一致；chat template 与完整生成参数一致；do_sample=False、max_new_tokens=512。未加载 LoRA，未训练，未自动评分。formal_v1 全部文件及原 v1 比较文件经哈希核对保持不变。

## 输出注意事项

18 题均达到 token 上限，可能截断。第 7 题的重复行标记来自代码围栏，第 14 题来自 Java 右花括号；第 17 题末尾存在一个 U+FFFD 替代字符，位于截断处。原始答案及自动标记均原样保存，质量由人工比较。

首次加载报告读取 hf_device_map 时遇到 AttributeError，无 OOM；独立评估脚本仅增加实际张量设备检查作为兼容处理，重跑通过。首次失败报告保留为 load_report_before_device_fix.json。

## C 盘旧缓存：只检查，未删除

目录：`C:\Users\杨宇屏\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct`

总文件大小：269,744,141 bytes（257.248 MiB）。

| 具体路径 | 文件大小（bytes） |
|---|---:|
| `C:\Users\杨宇屏\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct\blobs\67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2.42781964.incomplete` | 268,069,638 |
| `C:\Users\杨宇屏\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct\blobs\a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d.f7c775c0.incomplete` | 0 |
| `C:\Users\杨宇屏\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct\trees\aa8e72537993ba99e69dfaafa59ed015b17504d1.json` | 1,761 |
| `C:\Users\杨宇屏\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots\aa8e72537993ba99e69dfaafa59ed015b17504d1\config.json` | 661 |
| `C:\Users\杨宇屏\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots\aa8e72537993ba99e69dfaafa59ed015b17504d1\generation_config.json` | 242 |
| `C:\Users\杨宇屏\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots\aa8e72537993ba99e69dfaafa59ed015b17504d1\merges.txt` | 1,671,839 |
| `C:\Users\杨宇屏\.cache\huggingface\hub\.locks\models--Qwen--Qwen2.5-3B-Instruct\67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2.lock` | 0 |
| `C:\Users\杨宇屏\.cache\huggingface\hub\.locks\models--Qwen--Qwen2.5-3B-Instruct\a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d.lock` | 0 |

共享 Xet 缓存：`C:\Users\杨宇屏\.cache\huggingface\xet`，379,329 bytes。它是共享缓存，无法把全部内容归因于本次 3B 下载。

D 盘续传文件也全部保留。大小按文件长度汇总；不包含其他模型目录。没有删除任何旧缓存。

## 文件

- `D:\Code\huawei-ai-24weeks\projects\ai-study-assistant\finetune\evaluation\base_3b\base_3b_answers.json`
- `D:\Code\huawei-ai-24weeks\projects\ai-study-assistant\finetune\evaluation\base_3b\base_3b_answers.md`
- `D:\Code\huawei-ai-24weeks\projects\ai-study-assistant\finetune\evaluation\base_3b\download_report.json`
- `D:\Code\huawei-ai-24weeks\projects\ai-study-assistant\finetune\evaluation\base_3b\cache_verification.json`
- `D:\Code\huawei-ai-24weeks\projects\ai-study-assistant\finetune\evaluation\base_3b\load_report.json`
- `D:\Code\huawei-ai-24weeks\projects\ai-study-assistant\finetune\evaluation\base_3b\evaluation_report.json`
- `D:\Code\huawei-ai-24weeks\projects\ai-study-assistant\finetune\evaluation\base_3b\evaluation_console.log`

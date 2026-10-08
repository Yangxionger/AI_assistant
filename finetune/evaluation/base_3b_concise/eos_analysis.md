# 3B Base EOS 诊断与简洁提示评估

EOS 配置正常。旧评估的生成预算为 512 tokens，原提示没有篇幅要求，答案普遍较长。下面的实测支持长度预算不足这一解释；没有强制插入 EOS，也没有按字符截断输出。

## EOS 实测证据

- 生成 EOS IDs 为 151645（<|im_end|>）和 151643（<|endoftext|>），与模型原生配置一致；tokenizer EOS 为 151645。
- min_length/min_new_tokens 没有设置额外最小长度；没有 suppress_tokens/begin_suppress_tokens 或 forced_eos_token_id；chat template 哈希与原评估一致。
- 简短控制题：请只回复一个字：好。回答 `好`，共 2 tokens，末尾 ID=151645，自然 EOS。
- 原第 1 题：仅把诊断生成上限提高到 1536；实际在 536 tokens 自然 EOS。前 512 tokens 解码结果与旧回答完全相同，直接验证该旧回答被上限截断。
- 延长上限的诊断只测了原第 1 题，其他原题在更高上限下的最终 EOS 步数没有逐题测量。正式简洁评估仍使用 512 token 上限。

## 新一轮设置

复用相同 18 道原题，在每道 user 消息末尾追加：

> 请将回答控制在200～350字，直接回答问题。

原模板与完整生成参数保持一致：do_sample=False、max_new_tokens=512；4-bit NF4、bfloat16 compute、device_map=auto，实际张量均在 cuda:0；没有 LoRA，没有训练，没有自动评分。

本轮改变了 prompt，比较旧 0.5B/LoRA/3B 结果时需考虑新增的篇幅要求。

## 结果

| 指标 | 原 3B 提示 | 简洁提示 |
|---|---:|---:|
| 成功生成 | 18/18 | 18/18 |
| 自然 EOS | 0/18 | 18/18 |
| 触及长度上限 | 18/18 | 0/18 |
| 平均生成 tokens（含停止 token） | 512 | 199.78 |
| 平均字符数（含空白） | 969.56 | 351.78 |
| 推理时间，秒 | 406.28 | 164.39 |

去除空白后平均字符数 335.28；200～350 字范围内 9/18。超过 350 的题号：[1, 4, 10, 11, 12, 14, 16, 18]；低于 200 的题号：[6]。字数按去空白 Unicode 字符数统计，代码和标点计入；未做任何输出裁剪。

空回答 0；异常标记 {}。无 OOM；整卡每秒采样峰值 3891 MiB，PyTorch allocated/reserved 峰值 2141.22/2192.0 MiB。

旧 3B 答案、旧 0.5B 比较文件以及 formal_v1 全部文件经哈希核对保持不变；HF_HOME 继续为 D:\Code\huggingface_cache；未删除任何缓存。

## 文件

- base_3b_answers.json：原题、有效 user 内容、原始回答、完整生成 token IDs、末尾 token、停止原因和统计。
- base_3b_answers.md：供人工阅读的 18 题原始回答。
- eos_diagnostics.json：简短控制题、原第 1 题延长输出及完整诊断 token IDs。
- evaluation_report.json：硬件与文件保护记录。
- evaluation_console.log：完整终端日志。

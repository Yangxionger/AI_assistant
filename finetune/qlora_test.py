from numpy import gradient
import torch

from datasets import Dataset
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

#SFT数据
dataset = Dataset.from_list([
    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 RAG？"
            },
            {
                "role": "assistant",
                "content": "RAG 是检索增强生成，通过先检索相关知识，再结合知识生成答案。"
            }
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "什么是 Embedding？"
            },
            {
                "role": "assistant",
                "content": "Embedding 是把文本等信息转换成向量表示的方法。"
            }
        ]
    }
])

#4bit量化
quantization_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#LoRA
lora_config=LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
         "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    task_type="CAUSAL_LM"
)

#训练规则
training_args=SFTConfig(
    output_dir="./qwen_lora_output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_step=2e-4,
    logging_steps=1
)

#训练总管
trainer=SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    quantization_config=quantization_config
)
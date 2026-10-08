from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel


BASE_DIR = Path(__file__).resolve().parent

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_DIR = BASE_DIR / "output" / "final_adapter"


# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# 2. 4-bit 加载原始基础模型
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto"
)

# 3. 准备同一个问题
messages = [
    {
        "role": "user",
        "content": "什么是 RAG？"
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(base_model.device)


# 4. 先测试原始 Qwen
base_model.eval()

with torch.no_grad():
    base_outputs = base_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

base_generated_tokens = base_outputs[0][
    inputs["input_ids"].shape[1]:
]

base_answer = tokenizer.decode(
    base_generated_tokens,
    skip_special_tokens=True
)

# 5. 加载 LoRA Adapter
lora_model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_DIR)
)

lora_model.eval()


# 6. 再让 LoRA 模型回答同一个问题
with torch.no_grad():
    lora_outputs = lora_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

lora_generated_tokens = lora_outputs[0][
    inputs["input_ids"].shape[1]:
]

lora_answer = tokenizer.decode(
    lora_generated_tokens,
    skip_special_tokens=True
)


# 7. 对比
print("\n========== 原始 Qwen ==========")
print(base_answer)

print("\n========== LoRA 微调后 ==========")
print(lora_answer)
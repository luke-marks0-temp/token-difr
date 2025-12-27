# %%
"""Simple vLLM demo script - run inference on a prompt multiple times."""

import os

os.environ.setdefault("VLLM_USE_V1", "1")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# %%
# Configuration - edit these values
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

model = LLM(
    model=MODEL,
    tensor_parallel_size=1,
    max_model_len=4096,
)

# %%
PROMPT = "How should I boil an egg?"
NUM_RUNS = 10
MAX_TOKENS = 64
TEMPERATURE = 1.0
TOP_K = 50
TOP_P = 1.0
SEED = 42

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL)

messages = [{"role": "user", "content": PROMPT}]
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

token_prompts = [{"prompt_token_ids": prompt_token_ids} for _ in range(NUM_RUNS)]


sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    top_k=TOP_K,
    top_p=TOP_P,
    seed=SEED,
)

# %%
outputs = model.generate(token_prompts, sampling_params=sampling_params)

# %%
print(f"Model: {MODEL}")
print(f"Prompt: {PROMPT}")
print()
for idx, output in enumerate(outputs, start=1):
    completion_ids = output.outputs[0].token_ids
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    print(f"Run {idx}: {completion_text}\n")

# %%
first_output = outputs[0]
print(first_output)
# %%

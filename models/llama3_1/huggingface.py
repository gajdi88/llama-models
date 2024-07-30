# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "./Meta-Llama-3.1-8B-Instruct"

device = torch.device('cuda')

# transfer model


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

pipeline.model.to(device)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
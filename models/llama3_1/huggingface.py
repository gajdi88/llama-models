# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import transformers
import torch
import asyncio

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")

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

async def chatbot():
    messages = []

    while True:
        try:
            content_in = input('>>> ')
            if content_in:
                messages.append({"role": "user", "content": content_in})

                # Combine messages into a single input string
                input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

                # Generate response
                outputs = pipeline(input_text, max_new_tokens=256)

                content_out = outputs[0]["generated_text"]
                print(content_out)

                messages.append({"role": "assistant", "content": content_out})
            else:
                break
        except (KeyboardInterrupt, EOFError):
            print("Exiting chat...")
            break

# Run the chatbot
asyncio.run(chatbot())
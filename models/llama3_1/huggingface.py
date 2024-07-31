import transformers
import torch
import asyncio
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
device = torch.device('cuda')

# Load the model and tokenizer
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

model.to(device)

async def chatbot():
    messages = []

    while True:
        try:
            content_in = input('>>> ')
            if content_in:
                messages.append({"role": "user", "content": content_in})

                # Combine messages into a single input string
                input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
                attention_mask = torch.ones(input_ids.shape, device=device)  # Create attention mask

                # Generate response token-by-token
                output_ids = input_ids
                for _ in range(256):  # Limit to 256 tokens
                    outputs = model.generate(
                        output_ids,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        max_new_tokens=1,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    next_token_id = outputs[0, -1].unsqueeze(0).unsqueeze(0)
                    output_ids = torch.cat((output_ids, next_token_id), dim=1)

                    # Convert next_token_id tensor to a single token ID and decode
                    next_token = tokenizer.decode(next_token_id.item())
                    print(next_token, end='', flush=True)

                    if next_token == tokenizer.eos_token:
                        break

                output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                messages.append({"role": "assistant", "content": output_text})
                print()
            else:
                break
        except (KeyboardInterrupt, EOFError):
            print("Exiting chat...")
            break

# Run the chatbot
asyncio.run(chatbot())
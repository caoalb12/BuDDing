import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
hf_token = os.getenv("HF_TOKEN")

print("[LOG]: Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, token=hf_token)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

print(f"[LOG]: Model loaded on: {device}")

def generate_response(prompt, omission_set, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    original_layers = model.model.layers
    model.model.layers = torch.nn.ModuleList([
        layer for i, layer in enumerate(original_layers) if i not in omission_set
    ])

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Restore original layers after generation
    model.model.layers = original_layers
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "Explain quantum computing in simple terms:"
    print(f"\nPrompt: {prompt}\n")
    print("Generating response...\n")
    response = generate_response(prompt, omission_set=[])
    print(f"Response:\n{response}")
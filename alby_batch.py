import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# LOAD LLAMA MODEL + TOKENIZER
# ============================================================

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16
).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


# ============================================================
# CORE FUNCTION:
# BATCHED, LAYER-BY-LAYER PROCESSING
# WITHOUT SKIPPING LAYERS — INSTEAD,
# ONLY APPLY LAYER TO PROMPTS NOT IN ITS OMISSION SET
# ============================================================

def batched_layerwise_forward(prompts, omission_sets):
    """
    Implements the architecture:

    Layer 0:
        A -> L0
        B -> L0
        C -> L0

    Layer 1:
        B -> L1
        C -> L1

    Layer 2:
        A -> L2
        C -> L2
    ...
    """

    # Tokenize all prompts, but do NOT batch them into one tensor.
    encoded = [
        tokenizer(p, return_tensors="pt").to(device)
        for p in prompts
    ]

    # Initialize hidden states per prompt
    hidden_states = []
    attn_masks = []
    position_ids = []

    for enc in encoded:
        ids = enc["input_ids"]
        mask = enc["attention_mask"].to(torch.bool)

        hs = model.model.embed_tokens(ids)
        hidden_states.append(hs)
        attn_masks.append(mask)

        seq_len = hs.size(1)
        pid = torch.arange(seq_len, device=device).unsqueeze(0)
        position_ids.append(pid)

    # Process each layer independently per prompt
    for layer_index, layer in enumerate(model.model.layers):
        for i in range(len(prompts)):
            if layer_index in omission_sets[i]:
                continue  # skip this layer for this prompt

            hs = hidden_states[i]
            mask = attn_masks[i]
            pid = position_ids[i]

            out = layer(
                hs,
                mask,
                pid,
                None,   # past_key_values
                False,  # output_attentions
                False,  # use_cache
                None    # position_embeddings
            )[0]

            hidden_states[i] = out

    # Final norm + output token for each prompt
    outputs = []
    for i in range(len(prompts)):
        hs = model.model.norm(hidden_states[i])
        logits = model.lm_head(hs)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
        out_ids = torch.cat([encoded[i]["input_ids"], next_token], dim=1)
        decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        outputs.append(decoded)

    return outputs


# ============================================================
# TEST MAIN
# ============================================================

def main():
    prompts = [
        "Explain quantum computing simply.",
        "What is the capital of France?",
        "Write a poem about GPUs."
    ]

    omission_sets = [
        [1, 3],   # A
        [2, 4],   # B
        [5, 6]    # C
    ]

    outputs = batched_layerwise_forward(prompts, omission_sets)

    for i, out in enumerate(outputs):
        print(f"Prompt {i}: {prompts[i]}")
        print(f"Omission Set: {omission_sets[i]}")
        print(f"Output:\n{out}")
        print("-" * 60)


if __name__ == "__main__":
    main()
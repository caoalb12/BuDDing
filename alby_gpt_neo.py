import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPTNeoForCausalLM
from transformers.cache_utils import DynamicCache


# ---------------------------------------------------------
# Device helper
# ---------------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# One dynamic KV-aware step through GPT-Neo with layer skipping
# ---------------------------------------------------------
def dynamic_step(
    token_id: torch.Tensor,
    position_id: int,
    cache: DynamicCache,
    omission_set: set[int],
    model,
    device: torch.device,
):
    """
    Run ONE decoding step for ONE prompt with GPT-Neo, allowing per-layer skipping.

    token_id:   [1, 1] int64 tensor (last generated token)
    position_id: int scalar (absolute position of this token)
    cache:      DynamicCache object (shared across all layers)
    omission_set: set of layer indices to SKIP for this example
    model:      GPTNeoForCausalLM
    """

    transformer = model.transformer
    layers = transformer.h  # list of GPTNeoBlock

    # ---- Embedding for this single token ----
    # token embedding
    token_emb = transformer.wte(token_id)  # [1, 1, hidden]
    # position embedding
    pos_ids = torch.tensor([[position_id]], device=device)
    pos_emb = transformer.wpe(pos_ids)    # [1, 1, hidden]

    hidden_states = transformer.drop(token_emb + pos_emb)  # [1, 1, hidden]

    # ---- Run through blocks with per-layer skipping ----
    for layer_idx, block in enumerate(layers):
        # SKIP this layer for this prompt
        if layer_idx in omission_set:
            continue

        # In newer Transformers, GPTNeoBlock expects `layer_past` to be the
        # *cache object itself*. It will internally call
        #   layer_past.update(key, value, self.layer_id, cache_kwargs)
        # so we must pass the DynamicCache, not a tensor/tuple.
        outputs = block(
            hidden_states,
            layer_past=cache,
            attention_mask=None,
            head_mask=None,
            use_cache=True,
            output_attentions=False,
        )

        # Newer GPT-Neo returns: (hidden_states, ...) but NOT (hidden, present)
        hidden_states = outputs[0]  # [1, 1, hidden]

    # Final layer norm
    hidden_states = transformer.ln_f(hidden_states)  # [1, 1, hidden]

    # We DO NOT touch `cache` ourselves; it is updated inside the blocks.
    return hidden_states, cache


# ---------------------------------------------------------
# Sampling helper – 4 decoding modes
# ---------------------------------------------------------
def sample_from_logits(logits, mode="greedy", top_k=40, top_p=0.9, temperature=1.0):
    """
    logits: [vocab_size]
    mode: 'greedy', 'top_k', 'top_p', 'temperature'
    returns: int token_id
    """
    if mode == "greedy":
        return int(torch.argmax(logits))

    # Apply temperature for non-greedy modes
    if temperature != 1.0:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)

    if mode == "top_k":
        k = min(top_k, logits.size(-1))
        values, indices = torch.topk(probs, k)
        idx = torch.multinomial(values, 1)
        return int(indices[idx])

    if mode == "top_p":
        # Sort probabilities
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        # Keep tokens until cumulative prob > top_p
        mask = cumulative <= top_p
        mask[0] = True  # ensure at least one token

        filtered_probs = sorted_probs[mask]
        filtered_indices = sorted_indices[mask]

        filtered_probs = filtered_probs / filtered_probs.sum()
        idx = torch.multinomial(filtered_probs, 1)
        return int(filtered_indices[idx])

    if mode == "temperature":
        idx = torch.multinomial(probs, 1)
        return int(idx)

    # Fallback
    return int(torch.argmax(logits))


# ---------------------------------------------------------
# Full autoregressive generation with dynamic per-layer KV skipping
# ---------------------------------------------------------
def generate_with_dynamic_kv(
    prompt_text,
    omission_set,
    model,
    tokenizer,
    device,
    max_new_tokens=30,
    mode="greedy",
    top_k=40,
    top_p=0.9,
    temperature=1.0,
):
    """
    Autoregressive generation for a single prompt, with:
      - per-layer KV caches
      - per-layer skipping via omission_set
    """
    model.eval()

    # 1) Tokenize prompt
    enc = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]       # [1, L]
    prompt_len = input_ids.size(1)

    num_layers = model.config.num_layers
    past_key_values = [None] * num_layers

    # 2) Feed prompt tokens one by one to build KV cache
    logits_last = None
    for i in range(prompt_len):
        token_step = input_ids[:, i:i+1]  # [1, 1]
        position_id = i                   # absolute position

        hidden, past_key_values = dynamic_step(
            token_step,
            position_id,
            past_key_values,
            omission_set,
            model,
            device,
        )

        if i == prompt_len - 1:
            logits_last = model.lm_head(hidden)[0, 0, :]  # [vocab_size]

    # 3) Generate new tokens
    generated_token_ids = []
    curr_logits = logits_last

    for step in range(max_new_tokens):
        next_token_id = sample_from_logits(
            curr_logits,
            mode=mode,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        # stop on EOS if defined
        if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
            break

        generated_token_ids.append(next_token_id)

        token_step = torch.tensor([[next_token_id]], device=device, dtype=torch.long)
        position_id = prompt_len + step

        hidden, past_key_values = dynamic_step(
            token_step,
            position_id,
            past_key_values,
            omission_set,
            model,
            device,
        )
        curr_logits = model.lm_head(hidden)[0, 0, :]

    # 4) Build final sequence and decode
    if generated_token_ids:
        gen_tensor = torch.tensor(
            generated_token_ids, device=device, dtype=torch.long
        ).unsqueeze(0)
        full_ids = torch.cat([input_ids, gen_tensor], dim=1)
    else:
        full_ids = input_ids

    return tokenizer.decode(full_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------
# MAIN – Run all 4 decoding modes on your 3 prompts
# ---------------------------------------------------------
def main():
    device = get_device()
    print(f"Using device: {device}\n")

    model_name = "EleutherAI/gpt-neo-125M" # Change this to be whatever model you need

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
    model.config.use_cache = True  # make sure caching is enabled

    prompts = {
        "A": "The Eiffel Tower is located in",
        "B": "The capital of Japan is",
        "C": "The largest mammal in the ocean is",
    }

    # Layer indices to skip for each prompt
    omission_sets = {
        "A": [1, 3],
        "B": [2, 4],
        "C": [5, 6],
    }

    decoding_modes = {
        "greedy":      {"mode": "greedy"},
        "top_k":       {"mode": "top_k", "top_k": 40},
        "top_p":       {"mode": "top_p", "top_p": 0.9},
        "temperature": {"mode": "temperature", "temperature": 0.7},
    }

    for mode_name, cfg in decoding_modes.items():
        print("=" * 60)
        print(f"Decoding mode: {mode_name}")
        print("=" * 60)

        for key in prompts:
            text = generate_with_dynamic_kv(
                prompt_text=prompts[key],
                omission_set=omission_sets[key],
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=30,
                **cfg,
            )
            print(f"\nPrompt {key} ({mode_name}):\n  {text}\n")

        print("\n")


if __name__ == "__main__":
    main()
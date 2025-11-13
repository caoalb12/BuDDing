import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
hf_token = os.getenv("HF_TOKEN")

print("[LOG]: Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, token=hf_token)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

print(f"[LOG]: Model loaded on: {device}")


def custom_forward_with_masks(input_ids, attention_mask, layer_masks):
    """
    Custom forward pass with per-sample layer masking.
    
    Args:
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        layer_masks: [batch_size, num_layers] - binary mask (1=use layer, 0=skip layer)
    
    Returns:
        logits: [batch_size, seq_len, vocab_size]
    """
    batch_size, seq_len = input_ids.shape
    num_layers = len(model.model.layers)
    
    # Get embeddings
    hidden_states = model.model.embed_tokens(input_ids)  # [batch, seq, hidden]
    
    # Create position ids
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Create causal attention mask (4D format expected by model)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device),
        diagonal=1
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    # Apply padding mask if needed
    if attention_mask is not None:
        # Expand attention mask: [batch, 1, 1, seq_len]
        expanded_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        # Invert mask (1 -> 0, 0 -> -inf)
        expanded_mask = (1.0 - expanded_mask) * torch.finfo(hidden_states.dtype).min
        # Combine with causal mask
        causal_mask = causal_mask + expanded_mask
    
    # Get rotary embeddings once
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    
    # Process through layers with per-sample masking
    for layer_idx, layer in enumerate(model.model.layers):
        # Check which samples should use this layer
        use_layer = layer_masks[:, layer_idx]  # [batch_size]
        
        if use_layer.any():
            # Process all samples through the layer
            layer_output = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]
            
            # Selectively update hidden states based on mask
            # use_layer: [batch] -> [batch, 1, 1] for broadcasting
            mask = use_layer.view(-1, 1, 1).to(hidden_states.dtype)
            hidden_states = mask * layer_output + (1 - mask) * hidden_states
    
    # Final norm and head
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    
    return logits


def generate_with_per_sample_omissions(prompts, omission_sets, max_new_tokens=64, temperature=0.7):
    """
    Generate responses for multiple prompts, each with its own omission set.
    
    Args:
        prompts: List of prompt strings
        omission_sets: List of sets, one per prompt - layers to OMIT
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        List of generated text strings
    """
    batch_size = len(prompts)
    num_layers = len(model.model.layers)
    
    # Tokenize prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Create layer masks: 1 = use layer, 0 = skip layer
    layer_masks = torch.ones((batch_size, num_layers), dtype=torch.bool, device=device)
    for i, omit_set in enumerate(omission_sets):
        for layer_idx in omit_set:
            layer_masks[i, layer_idx] = 0
    
    # Track which sequences are done
    eos_token_id = tokenizer.eos_token_id
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass with per-sample layer masking
            logits = custom_forward_with_masks(input_ids, attention_mask, layer_masks)
            
            # Get next token logits (last position)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample next tokens
            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Only update unfinished sequences
            next_tokens = next_tokens * unfinished + eos_token_id * (~unfinished)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
            ], dim=-1)
            
            # Check for EOS
            unfinished = unfinished & (next_tokens != eos_token_id)
            if not unfinished.any():
                break
    
    # Decode outputs
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PER-SAMPLE LAYER OMISSION - Single generate call!")
    print("="*70)
    
    # Your exact example
    prompts = [
        "What is the capital of France?",  # Request a
        "Explain quantum computing:",       # Request b  
        "Write a haiku about coding:"       # Request c
    ]
    
    omission_sets = [
        {10, 11, 12},    # a: skip layers 0, 2, 3
        {10, 12, 14},       # b: skip layers 0, 1
        {11, 15, 21}     # c: skip layers 0, 3, 4
    ]
    
    
    print("\nRequest configuration:")
    print("Request a: omit layers", sorted(omission_sets[0]))
    print("Request b: omit layers", sorted(omission_sets[1]))
    print("Request c: omit layers", sorted(omission_sets[2]))
    
    print("\n" + "-"*70)
    print("Generating...")
    print("-"*70)
    
    responses = generate_with_per_sample_omissions(
        prompts, 
        omission_sets, 
        max_new_tokens=50,
        temperature=0.7
    )
    
    for i, (prompt, omit_set, response) in enumerate(zip(prompts, omission_sets, responses)):
        print(f"\n[Request {chr(97+i)}] Omitting layers: {sorted(omit_set)}")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
    
    print("\n" + "="*70)
    # print("Comparison: Same prompt, different omissions")
    # print("="*70)
    
    # same_prompts = ["The meaning of life is"] * 4
    # different_omissions = [
    #     set(),                    # Full model
    #     set(range(0, 8)),        # Skip early layers
    #     set(range(10, 20)),      # Skip middle layers
    #     set(range(24, 32))       # Skip late layers
    # ]
    
    # responses = generate_with_per_sample_omissions(
    #     same_prompts,
    #     different_omissions,
    #     max_new_tokens=30,
    #     temperature=0.7
    # )
    
    # labels = ["Full model", "No early (0-7)", "No middle (10-19)", "No late (24-31)"]
    # for label, omit_set, response in zip(labels, different_omissions, responses):
    #     print(f"\n{label}:")
    #     print(f"  {response}")

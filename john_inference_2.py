import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# Config
# ---------------------------
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # requires HF auth + license
HF_TOKEN = os.getenv("HF_TOKEN")

# Three example prompts (you can change these)
PROMPTS = {
    "a": "Explain quantum computing in simple terms.",
    "b": "Summarize the causes of climate change in one paragraph.",
    "c": "Write a friendly email thanking a colleague for their help."
}

# Per-request layer subsets we want to execute (0-based layer indices)
ROUTES = {
    "a": [0, 2, 3],
    "b": [0, 1],
    "c": [0, 3, 4],
}

# ---------------------------
# Utilities
# ---------------------------

def pick_dtype_and_device():
    if torch.cuda.is_available():
        # bf16 on recent GPUs gives speedups and lower memory
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, torch.device("cuda")
        else:
            return torch.float16, torch.device("cuda")
    else:
        # CPU path prefers fp32 for correctness/perf
        return torch.float32, torch.device("cpu")


def build_additive_causal_mask(lengths, L_max, dtype, device):
    """
    lengths: list of actual sequence lengths per sample (len B)
    L_max:   max length after padding
    Returns: [B, 1, L_max, L_max] additive mask with 0.0 for attend, -inf for masked
    """
    B = len(lengths)
    # Base: upper-triangular mask (prevent attending to future tokens)
    tri = torch.triu(torch.ones((L_max, L_max), dtype=torch.bool, device=device), diagonal=1)
    base = torch.zeros((L_max, L_max), dtype=dtype, device=device)
    base = base.masked_fill(tri, torch.finfo(dtype).min)  # 0 on/below diag, -inf above diag

    out = torch.zeros((B, 1, L_max, L_max), dtype=dtype, device=device)
    for i, S in enumerate(lengths):
        m = base.clone()
        # Mask out padded queries/keys
        if S < L_max:
            m[S:, :] = torch.finfo(dtype).min  # queries beyond S are fully masked
            m[:, S:] = torch.finfo(dtype).min  # keys beyond S are not visible
        out[i, 0] = m
    return out


@torch.no_grad()
def get_rope_tuple(model, hidden_states, position_ids):
    """
    Compute RoPE (cos, sin) like HF's internal path.
    Works across transformers versions by looking in likely places.
    """
    # Llama3.x: rotary_emb often lives inside attention
    attn0 = model.model.layers[0].self_attn
    rotary = getattr(getattr(model.model, "rotary_emb", None), "forward", None)
    if rotary is None:
        rotary = getattr(attn0, "rotary_emb", None)
    if rotary is None:
        return None  # very old versions compute from position_ids internally

    # Some versions expose it as a module with forward(x, position_ids) -> (cos, sin)
    try:
        return rotary(hidden_states, position_ids)
    except TypeError:
        # Fallback in case signature changed; last resort is to let layer handle position_ids directly
        return None


def pad_stack_hidden(hidden_list, L_max):
    """
    hidden_list: list of [1, S_i, D] tensors
    returns batch [B, L_max, D] and the original lengths
    """
    B = len(hidden_list)
    D = hidden_list[0].shape[-1]
    device = hidden_list[0].device
    dtype = hidden_list[0].dtype

    batch = torch.zeros((B, L_max, D), dtype=dtype, device=device)
    lengths = []
    for i, hs in enumerate(hidden_list):
        S = hs.shape[1]
        lengths.append(S)
        batch[i, :S, :] = hs
    return batch, lengths


def unstack_hidden(batched_hidden, lengths):
    """
    Split a [B, L, D] tensor back into list of [1, S_i, D] using per-sample lengths.
    """
    out = []
    for i, S in enumerate(lengths):
        out.append(batched_hidden[i:i+1, :S, :])
    return out


def run_layer_batched(model, layer_idx, names, hiddens_map, pos_ids_map):
    """
    Run model.model.layers[layer_idx] for a group of samples (`names`) as a single batch.
    Updates hiddens_map in-place for each name.
    """
    if not names:
        return

    layer = model.model.layers[layer_idx]
    # Collect per-sample states
    hs_list = [hiddens_map[n] for n in names]
    pos_list = [pos_ids_map[n] for n in names]
    L_max = max(t.shape[1] for t in hs_list)

    # Pad/stack to a batch
    hs_batch, lengths = pad_stack_hidden(hs_list, L_max)
    pos_batch = torch.zeros((len(names), L_max), dtype=torch.long, device=hs_batch.device)
    for i, pid in enumerate(pos_list):
        S = pid.shape[1]
        pos_batch[i, :S] = pid[0]

    # Mask & RoPE
    attn_mask = build_additive_causal_mask(lengths, L_max, dtype=hs_batch.dtype, device=hs_batch.device)
    rope = get_rope_tuple(model, hs_batch, pos_batch)

    # Call the layer (support both APIs)
    try:
        out = layer(hidden_states=hs_batch, attention_mask=attn_mask,
                    position_embeddings=rope, use_cache=False)[0]
    except TypeError:
        out = layer(hidden_states=hs_batch, attention_mask=attn_mask,
                    position_ids=pos_batch, use_cache=False)[0]

    # Unstack results
    outs = unstack_hidden(out, lengths)
    for name, hs_out in zip(names, outs):
        hiddens_map[name] = hs_out


def run_layer_batched_in_stream(model, layer_idx, names, hiddens_map, pos_ids_map, stream):
    """
    Same as run_layer_batched, but uses a dedicated CUDA stream for concurrency.
    """
    if not names:
        return None  # nothing to do

    if not torch.cuda.is_available():
        run_layer_batched(model, layer_idx, names, hiddens_map, pos_ids_map)
        return None

    # Ensure this stream sees work from current stream
    stream.wait_stream(torch.cuda.current_stream())

    event = torch.cuda.Event()
    with torch.cuda.stream(stream):
        run_layer_batched(model, layer_idx, names, hiddens_map, pos_ids_map)
        event.record(stream)

    return event


# ---------------------------
# Main demo: schedule a/b/c
# ---------------------------

@torch.inference_mode()
def main():
    dtype, device = pick_dtype_and_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"[LOG] Loading tokenizer and model on {device} (dtype={dtype})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,         # transformers warns 'torch_dtype' is deprecated in favor of 'dtype' in latest; both work
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    ).to(device).eval()

    # Tokenize each request (no padding – we’ll pad only when batching)
    inputs = {k: tokenizer(v, return_tensors="pt").to(device) for k, v in PROMPTS.items()}
    input_ids = {k: v["input_ids"] for k, v in inputs.items()}

    # Embed once (start of the stack)
    hiddens = {k: model.model.embed_tokens(ids) for k, ids in input_ids.items()}

    # Position ids for each
    pos_ids = {k: torch.arange(hiddens[k].shape[1], device=device).long().unsqueeze(0) for k in hiddens}

    # ---- Stage 0: Layer 0 (a, b, c) as a batch
    print("[S0] Layer 0: a b c (batched)")
    run_layer_batched(model, layer_idx=0, names=["a", "b", "c"], hiddens_map=hiddens, pos_ids_map=pos_ids)

    # ---- Stage 1: Layer 1 (b) in parallel with Layer 2 (a)
    print("[S1] Parallel: Layer 1 (b)  ||  Layer 2 (a)")
    if torch.cuda.is_available():
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        ev1 = run_layer_batched_in_stream(model, 1, ["b"], hiddens, pos_ids, stream1)  # L1 on b
        ev2 = run_layer_batched_in_stream(model, 2, ["a"], hiddens, pos_ids, stream2)  # L2 on a

        # Wait for both before moving on
        cur = torch.cuda.current_stream()
        if ev1 is not None:
            cur.wait_event(ev1)
        if ev2 is not None:
            cur.wait_event(ev2)
        torch.cuda.synchronize()
    else:
        # CPU fallback: sequential
        run_layer_batched(model, 1, ["b"], hiddens, pos_ids)
        run_layer_batched(model, 2, ["a"], hiddens, pos_ids)

    # ---- Stage 2: Layer 3 (a, c) as a batch
    print("[S2] Layer 3: a c (batched)")
    run_layer_batched(model, layer_idx=3, names=["a", "c"], hiddens_map=hiddens, pos_ids_map=pos_ids)

    # ---- Stage 3: Layer 4 (c)
    print("[S3] Layer 4: c")
    run_layer_batched(model, layer_idx=4, names=["c"], hiddens_map=hiddens, pos_ids_map=pos_ids)

    # ---- Head: norm + lm_head for each request (1-token greedy step demo)
    print("[Head] norm + lm_head: a b c")
    next_tokens = {}
    for name in ["a", "b", "c"]:
        final_hidden = hiddens[name]
        normed = model.model.norm(final_hidden)
        logits_last = model.lm_head(normed[:, -1, :])           # [1, vocab]
        next_id = torch.argmax(logits_last, dim=-1).unsqueeze(-1)  # [1, 1]
        # Stitch the token to original inputs to show an output snippet
        out_ids = torch.cat([input_ids[name], next_id], dim=-1)
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        next_tokens[name] = {
            "token_id": int(next_id.item()),
            "text": text
        }

    print("\n=== Outputs (greedy +1 token) ===")
    for k, v in next_tokens.items():
        print(f"{k}: token_id={v['token_id']}\n{text_wrap(v['text'])}\n")


def text_wrap(s, width=100):
    # pretty print
    import textwrap
    return "\n".join(textwrap.wrap(s, width=width))


if __name__ == "__main__":
    main()
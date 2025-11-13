import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Set
from dataclasses import dataclass


@dataclass
class Request:
    """Represents a single inference request"""
    id: str
    input_ids: torch.Tensor
    omission_set: Set[int]
    hidden_state: torch.Tensor = None
    attention_mask: torch.Tensor = None


class DynamicLayerBatcher:
    """
    Handles dynamic batching and routing through transformer layers
    with per-request layer pruning.
    """
    
    def __init__(self, model, device: str = "cuda"):
        """
        Args:
            model: The LLaMA model
            device: Device to run on ("cuda" or "cpu")
        """
        self.model = model
        self.device = device
        self.num_layers = len(model.model.layers)
        
    def _group_requests_by_execution(
        self, 
        requests: List[Request], 
        layer_idx: int
    ) -> tuple[List[Request], List[Request]]:
        """
        Split requests into those that execute this layer vs those that skip it.
        
        Returns:
            (execute_requests, skip_requests)
        """
        execute = []
        skip = []
        
        for req in requests:
            if layer_idx in req.omission_set:
                skip.append(req)
            else:
                execute.append(req)
        
        return execute, skip
    
    def _create_execution_plan(
        self, 
        requests: List[Request]
    ) -> List[tuple[int, List[str]]]:
        """
        Create an execution plan showing which requests execute at each layer.
        
        Returns:
            List of (layer_idx, [request_ids]) tuples
        """
        plan = []
        for layer_idx in range(self.num_layers):
            execute, _ = self._group_requests_by_execution(requests, layer_idx)
            if execute:
                plan.append((layer_idx, [req.id for req in execute]))
        return plan
    
    def forward(
        self, 
        requests: List[Request],
        verbose: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Execute forward pass with dynamic batching and layer pruning.
        
        Args:
            requests: List of Request objects with omission sets
            verbose: Print execution plan
            
        Returns:
            Dictionary mapping request_id -> final hidden state
        """
        if verbose:
            print("\n=== Execution Plan ===")
            plan = self._create_execution_plan(requests)
            for layer_idx, req_ids in plan:
                print(f"Layer {layer_idx:2d}: {req_ids}")
            print()
        
        # Initialize hidden states using model's embedding layer
        for req in requests:
            with torch.no_grad():
                req.hidden_state = self.model.model.embed_tokens(req.input_ids)
        
        # Process each layer
        for layer_idx in range(self.num_layers):
            execute_reqs, skip_reqs = self._group_requests_by_execution(
                requests, layer_idx
            )
            
            if verbose and execute_reqs:
                print(f"Layer {layer_idx:2d}: "
                      f"execute={[r.id for r in execute_reqs]}, "
                      f"skip={[r.id for r in skip_reqs]}")
            
            # Process requests that execute this layer
            if execute_reqs:
                # Batch the hidden states (stack to preserve batch dimension)
                batch_hidden = torch.cat(
                    [req.hidden_state for req in execute_reqs], 
                    dim=0
                )
                
                # Batch attention masks
                batch_mask = torch.cat(
                    [req.attention_mask for req in execute_reqs],
                    dim=0
                )
                
                # Create position_ids for the batch
                batch_size = len(execute_reqs)
                seq_length = batch_hidden.shape[1]
                position_ids = torch.arange(
                    seq_length, 
                    device=self.device
                ).unsqueeze(0).expand(batch_size, -1)
                
                # Compute RoPE embeddings
                position_embeddings = self.model.model.rotary_emb(
                    batch_hidden, 
                    position_ids
                )
                
                # Prepare causal attention mask (4D)
                causal_mask = None
                if batch_mask is not None:
                    # Create causal mask [seq_length, seq_length]
                    causal_attention = torch.tril(
                        torch.ones((seq_length, seq_length), dtype=batch_hidden.dtype, device=self.device)
                    ).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, seq_length]
                    
                    # Expand padding mask to 4D [batch_size, 1, 1, seq_length]
                    padding_mask = batch_mask.unsqueeze(1).unsqueeze(2)
                    
                    # Combine: apply padding to both query and key dimensions
                    # padding_mask broadcasts to [batch_size, 1, seq_length, seq_length]
                    causal_mask = causal_attention * padding_mask * padding_mask.transpose(-1, -2)
                    
                    # Convert to attention mask format (0 -> large negative)
                    causal_mask = (1.0 - causal_mask) * torch.finfo(batch_hidden.dtype).min
                
                # Execute the layer
                layer = self.model.model.layers[layer_idx]
                with torch.no_grad():
                    batch_output = layer(
                        batch_hidden,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        past_key_values=None,
                        output_attentions=False,
                        use_cache=False,
                    )[0]
                
                # Split output back to individual requests
                for i, req in enumerate(execute_reqs):
                    req.hidden_state = batch_output[i:i+1]
            
            # Requests in skip_reqs keep their current hidden_state unchanged
        
        # Apply final norm and return
        results = {}
        for req in requests:
            with torch.no_grad():
                final_hidden = self.model.model.norm(req.hidden_state)
            results[req.id] = final_hidden
        
        return results


def load_model():
    """Load LLaMA 3.1-8B-Instruct model"""
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    hf_token = os.getenv("HF_TOKEN")

    print("[LOG]: Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        token=hf_token
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    print(f"[LOG]: Model loaded on: {device}")
    print(f"[LOG]: Number of layers: {len(model.model.layers)}")
    
    return model, tokenizer, device


def mock_router_predict(input_ids: torch.Tensor, request_id: str) -> Set[int]:
    """
    Mock router that returns predefined omission sets.
    Replace this with your actual router model.
    """
    if request_id == "A":
        return {10, 17, 20}
    elif request_id == "B":
        return {10, 18, 24}
    else:
        return set()


def main():
    # Load model
    model, tokenizer, device = load_model()
    
    # Initialize batcher
    batcher = DynamicLayerBatcher(model, device=device)
    
    # Prepare two sample requests
    text_a = "What is the capital of France?"
    text_b = "Explain quantum computing in simple terms."
    
    print(f"\n[LOG]: Preparing requests...")
    print(f"Request A: {text_a}")
    print(f"Request B: {text_b}")
    
    # Tokenize together with padding to ensure same length
    encoded = tokenizer(
        [text_a, text_b], 
        return_tensors="pt", 
        padding=True,
        truncation=True
    )
    
    input_ids_a = encoded.input_ids[0:1].to(device)
    input_ids_b = encoded.input_ids[1:2].to(device)
    attention_mask_a = encoded.attention_mask[0:1].to(device)
    attention_mask_b = encoded.attention_mask[1:2].to(device)
    
    # Get omission sets from router (mock for now)
    omission_a = mock_router_predict(input_ids_a, "A")
    omission_b = mock_router_predict(input_ids_b, "B")
    
    print(f"\n[LOG]: Omission sets:")
    print(f"Request A: {sorted(omission_a)}")
    print(f"Request B: {sorted(omission_b)}")
    
    # Create requests
    request_a = Request(
        id="A", 
        input_ids=input_ids_a,
        omission_set=omission_a,
        attention_mask=attention_mask_a
    )
    request_b = Request(
        id="B",
        input_ids=input_ids_b,
        omission_set=omission_b,
        attention_mask=attention_mask_b
    )
    
    # Execute batched inference
    print(f"\n[LOG]: Starting batched inference with dynamic layer pruning...")
    results = batcher.forward([request_a, request_b], verbose=True)
    
    # Generate next tokens
    print("\n=== Generation Results ===")
    for req_id, hidden_state in results.items():
        with torch.no_grad():
            logits = model.lm_head(hidden_state)
            next_token_id = logits[:, -1, :].argmax(dim=-1)
            next_token = tokenizer.decode(next_token_id[0])
        
        print(f"Request {req_id}: Next token = '{next_token}'")
        print(f"  Hidden state shape: {hidden_state.shape}")
    
    # Memory usage
    if device == "cuda":
        print(f"\n[LOG]: GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
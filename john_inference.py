import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import warnings
import os

hf_token = os.getenv("HF_TOKEN")

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM

def get_llm(model_name, device_map="auto"):

    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 torch_dtype='auto',
                                                 low_cpu_mem_usage=True,
                                                 dtype=torch.bfloat16, 
                                                 trust_remote_code=True,
                                                 token=hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    model.seqlen = 2048
    model.name = model_name

    print(f"[LOG]: Model loaded on: {device}")

    return model


class OnOff_LlamaDecoderLayer(nn.Module):
    def __init__(self, original_decoder_layer):
        super().__init__()
        self.hidden_size = original_decoder_layer.hidden_size

        self.self_attn = original_decoder_layer.self_attn
        self.mlp = original_decoder_layer.mlp
        self.input_layernorm = original_decoder_layer.input_layernorm
        self.post_attention_layernorm = original_decoder_layer.post_attention_layernorm

        self.pass_layer = False
        self.input = None
        self.output = None

    def turn_off(self):
        self.pass_layer = True

    def turn_on(self):
        self.pass_layer = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        # skip this decoder layer
        if self.pass_layer:
            outputs = (hidden_states,)

            if output_attentions:
                outputs += (None,)

            if use_cache:
                outputs += (past_key_value,)

            return outputs

        # else normal forward
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        self.input = hidden_states
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        if residual.device != hidden_states.device:

            residual = residual.to(hidden_states.device)

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        self.output= hidden_states
        return outputs

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
    AutoConfig,
    AutoModelForCausalLM,
    BertConfig,
    BertModel,
    BertForSequenceClassification
)
from transformers.modeling_outputs import CausalLMOutputWithPast

import csv
import ast
import re

def llama_block_replace(model):
    num_layers = len(model.model.layers)
    for i in range(num_layers):
        model.model.layers[i] = OnOff_LlamaDecoderLayer(model.model.layers[i])
    print("Replacement complete.")

def block_replace(model):
    if 'llama' in model.name.lower() or 'vicuna' in model.name.lower():
        model = llama_block_replace(model)
    else:
        print("ERROR")

    return model

def llama_turn_off(model, block_idx):
    model.model.layers[block_idx].turn_off()

def llama_turn_on(model, block_idx):
    model.model.layers[block_idx].turn_on()

def turn_off(model, block_idx):
    # print(model.name)
    if 'llama' in model.name.lower() or 'vicuna' in model.name.lower():
        llama_turn_off(model, block_idx)

def turn_on(model, block_idx):
    if 'llama' in model.name.lower() or 'vicuna' in model.name.lower():
        llama_turn_on(model, block_idx)

class AdaptiveSLEBForCausalLM(LlamaForCausalLM):
    def __init__(self, config, name=None, base_model=None, router_model=None, router_path=""):
        print('super constructor')
        super().__init__(config)

        print("Setting n_layers as num_hidden layers")
        self.n_layers = config.num_hidden_layers

        print('loading tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B", token=hf_token)

        if router_path != "":
          self.bert_tokenizer = AutoTokenizer.from_pretrained(router_path)

        print('setting base model')
        if base_model is not None:
            self.model = base_model
        else:
            print('no llm model loaded...')

        print('setting router model')
        if router_model is not None:
            self.router = router_model
            self.router.eval()
        else:
            print('no trained router loaded...')

        print('setting name')
        if name is None:
            self.name = 'v23_adaptSLEB'

        print('setting variables')
        self.seqlen = 2048
        self.input_set={}
        # self.excluded_indices=[0,2,4,6,7]

        print('init_count called')
        self.init_count()

    def get_skip_mask(self, router_logits):

        probabilities = router_logits  # shape: (1, 10)

        # excluded_indices = torch.tensor(self.excluded_indices)
        # probabilities[:, excluded_indices] = float('-inf')

        # print(probabilities)
        _, topk_indices = torch.topk(probabilities, 1, dim=-1, largest=True)
        predicted_label = topk_indices.item()  # (1,1) -> int
        # print(predicted_label)
        # if predicted_label in excluded_indices:
        #     print(probabilities)
        #     print(sdfsf)

        self.add_count(predicted_label)
        skip_layer = []

        with open('codes/llama_layer_list_6_advanced_tasks.csv', mode="r", newline="", encoding="utf-8") as f:
        # with open('codes/5_adaptive_cluster/clustered_layer_list_10.csv', mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                row_set_str = row[0]
                row_idx_str = row[1]

                if int(row_idx_str) == predicted_label:
                    my_tuple = ast.literal_eval(row_set_str)
                    skip_layer = list(my_tuple)
                    break

        return skip_layer
    def init_count(self):
        print('initial count!')
        self.count = [0] * 10
        self.total = 0

    def add_count(self, predicted_label):
        self.count[predicted_label] += 1
        self.total += 1
    def print_count(self):
        print('#'*20)
        print('returning count!')
        print(self.count)
        print(self.total)
        print('#'*20)


    def forward(self, input_ids=None, attention_mask=None, skip_layer=None, **kwargs):
        # seq_len = input_ids.size(1)
        # device = input_ids.device

        # if attention_mask is None:
        #     attention_mask = torch.ones((1, seq_len), device=device)

        # if skip_layer is None:
        #     with torch.no_grad():
        #         input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]

        #         answer_index = input_text.find("Answer")

        #         if answer_index != -1:
        #             question_input = input_text[:answer_index]
        #         else:
        #             match = re.search(r":\s*([^\.]*\.)", input_text)
        #             if match:
        #                 question_input = match.group(1).strip()
        #             else:
        #                 words = input_text.split()
        #                 question_input = " ".join(words[:7])

        #         if question_input in self.input_set:
        #             skip_layer = self.input_set[question_input]

        #         else:

        #             self.input_prompt = question_input
        #             bert_inputs = self.bert_tokenizer(
        #                 input_text,
        #                 return_tensors='pt',
        #                 padding=True,
        #                 truncation=True,
        #                 max_length=512
        #             ).to(device)

        #             router_outputs = self.router(
        #                 input_ids=bert_inputs['input_ids'],
        #                 attention_mask=bert_inputs['attention_mask']
        #             )
        #             router_logits = router_outputs.logits  # shape: (batch=1, num_labels=10)

        #             skip_layer = self.get_skip_mask(router_logits)
        #             self.input_set[question_input] = skip_layer
        # else:
        #     skip_mask = skip_mask.to(device)

        skip_mask = index_to_array[0]

        for idx in skip_layer:
            turn_off(self.model, idx)

        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        for idx in skip_layer:
            turn_on(self.model, idx)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=None,
            # cross_attentions=None
        )


def v23_adaptSLEB(model_name="meta-llama/Meta-Llama-3.1-8B", router_path="result/9_llama/router/10/2_onlylog/MSE/5"):
    print('loading base model using get_llm')
    base_model = get_llm(model_name)
    print('updating base_model_name')
    base_model.name = model_name
    print('replacing block models')
    base_model = block_replace(base_model)
    print('getting auto config')
    config = AutoConfig.from_pretrained(model_name, token=hf_token)
    # router_model = CustomBertForSequenceClassification.from_pretrained(router_path,num_labels=10)
    print('creating adaptive sleb class')
    model = AdaptiveSLEBForCausalLM(
        config=config,
        base_model=base_model,
        name='v23_adaptSLEB',
    )

    print('moving to cuda')
    model.to('cuda')
    model.rank = 0
    model.world_size = 1

    print('returning model')
    return model

model = v23_adaptSLEB()

print(model)
# print(tokenizer)
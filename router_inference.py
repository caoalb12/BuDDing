import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Instiatiate the router model with the PuDDing weights.
class RouterClassifier(nn.Module):
    def __init__(self, model_path='router.pt'):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.mlp = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        model_state = torch.load(model_path, map_location='cpu')
        self.load_state_dict(model_state['model_state_dict'])
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_embed = outputs.pooler_output
        return self.mlp(cls_embed)
    
    # Take in a list of prompts, tokenize each, and feed them to the router.
    def predict(self, prompts):
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        self.eval() # turn off training mode

        logits = self.forward(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )

        for idx, prompt in enumerate(prompts):
            omission_set_idx = torch.argmin(logits[idx]).item() # omission set with lowest loss
            print(f"Prompt: {prompt} | Predicted Omission Set Index: {omission_set_idx}")

if __name__ == "__main__":
    router = RouterClassifier()
    
    prompts = [
        "What is the capital city of France?",
        "Which city is the capital of France?",
        "What is capital of the United States?",
        "Should I get some coffee right now?",
    ]
    
    predictions = router.predict(prompts)
import torch
from torch import nn
from transformers import VisualBertConfig
from models.VisualBertResMLP import VisualBertResMLPModel
import torch.nn.functional as F
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisualBertResMLPPrediction(nn.Module):
    def __init__(self, vocab_size, layers, n_heads, num_class = 10, token_size = 26):
        super(VisualBertResMLPPrediction, self).__init__()
        VBconfig = VisualBertConfig(vocab_size= vocab_size, visual_embedding_dim = 512, num_hidden_layers = layers, num_attention_heads = n_heads, hidden_size = 2048)
        self.VisualBertResMLPEncoder = VisualBertResMLPModel(VBconfig, token_size = token_size)
        self.classifier = nn.Linear(VBconfig.hidden_size, num_class)
        self.bbox_embed = MLP(2048, 2048, 4, 3)

    def forward(self, inputs, visual_embeds):
        # prepare visual embedding
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).to(device)

        # append visual features to text
        inputs.update({
                        "visual_embeds": visual_embeds,
                        "visual_token_type_ids": visual_token_type_ids,
                        "visual_attention_mask": visual_attention_mask,
                        "output_attentions": True
                        })
                        
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        inputs['visual_token_type_ids'] = inputs['visual_token_type_ids'].to(device)
        inputs['visual_attention_mask'] = inputs['visual_attention_mask'].to(device)

        # Encoder output
        outputs = self.VisualBertResMLPEncoder(**inputs)
        
        # Classification layer
        classification_outputs = self.classifier(outputs['pooler_output'])
        
        # Detection predition
        bbox_outputs = self.bbox_embed(outputs['pooler_output']).sigmoid() 
        
        return (classification_outputs, bbox_outputs)

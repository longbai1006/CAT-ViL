import torch
from torch import nn
from transformers import VisualBertModel, VisualBertConfig
from timm import create_model
from models.CATViLEmbedding import VisualBertEmbeddings
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeiTPrediction(nn.Module):
    '''
    Data-Efficient Image Transformer VQLA Model
    vocab_size    = tokenizer length
    encoder_layer = 6
    n_heads       = 8
    num_class     = number of class in dataset
    '''
    def __init__(self, vocab_size, layers, n_heads, num_class):
        super(DeiTPrediction, self).__init__()

        self.config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        self.config.visual_embedding_dim = 512
        self.config.vocab_size = vocab_size 
        self.config.num_hidden_layers = layers
        self.config.num_attention_heads = n_heads        

        self.embeddings = VisualBertEmbeddings(config = self.config)
        self.deit = create_model("deit_base_patch16_224", pretrained=True)
        self.classifier = nn.Linear(768, num_class)
        self.bbox_embed = MLP(768, 768, 4, 3)
        
    def forward(self, inputs, visual_embeds):
        # prepare visual embedding
        # append visual features to text
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

        inputs.update({
                        "visual_embeds": visual_embeds,
                        "visual_token_type_ids": visual_token_type_ids,
                        "visual_attention_mask": visual_attention_mask,
                        })
        # Encoder output
        embedding_output = self.embeddings(
            input_ids = inputs['input_ids'].to(device),
            token_type_ids = inputs['token_type_ids'].to(device),
            position_ids = None,
            inputs_embeds = None,
            visual_embeds = inputs['visual_embeds'].to(device),
            visual_token_type_ids = inputs['visual_token_type_ids'].to(device),
            image_text_alignment = None,
        ) 

        outputs = self.deit.blocks(embedding_output)
        outputs = self.deit.norm(outputs)
        outputs = outputs.mean(dim=1)             
        
        # classification layer        
        classification_outputs = self.classifier(outputs)

        # Detection predition
        bbox_outputs = self.bbox_embed(outputs).sigmoid()
                
        return (classification_outputs, bbox_outputs)

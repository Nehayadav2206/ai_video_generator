import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from typing import List, Union

class TextEncoder:
    def __init__(self, model_id="openai/clip-vit-large-patch14"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id)
        self.text_encoder.to(self.device)
        self.text_encoder.eval()
        
        self.max_length = self.tokenizer.model_max_length
    
    def encode_text(self, prompts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text prompts to embeddings"""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Tokenize
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
        
        return text_embeddings
    
    def encode_prompt_pair(self, positive_prompt: str, negative_prompt: str = "") -> tuple:
        """Encode positive and negative prompts for classifier-free guidance"""
        if not negative_prompt:
            negative_prompt = ""
        
        positive_embeddings = self.encode_text(positive_prompt)
        negative_embeddings = self.encode_text(negative_prompt)
        
        return positive_embeddings, negative_embeddings
    
    def create_attention_mask(self, prompts: Union[str, List[str]]) -> torch.Tensor:
        """Create attention mask for text inputs"""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return text_inputs.attention_mask.to(self.device)
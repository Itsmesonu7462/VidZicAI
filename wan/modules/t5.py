import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizers import HuggingfaceTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

__all__ = [
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
]

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize T5 model and tokenizer from Hugging Face
hf_model = T5ForConditionalGeneration.from_pretrained("google/umt5-xxl").to(device)
hf_tokenizer = T5Tokenizer.from_pretrained("google/umt5-xxl")

def fp16_clamp(x):
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x

class T5EncoderModel:

    def __init__(
        self,
        text_len,
        dtype=torch.bfloat16,
        device=device,  # Auto-detect GPU or default to CPU
        checkpoint_path=None,
        tokenizer_path=None,
        shard_fn=None,
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = torch.device(device)  # Ensuring compatibility
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # Load model & tokenizer from Hugging Face instead of checkpoint
        self.model = hf_model.eval().requires_grad_(False).to(self.device)
        self.tokenizer = hf_tokenizer

        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)

    def __call__(self, texts, device=None):
        device = device or self.device  # Default to model's device
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.text_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
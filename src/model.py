"""Implement a model for predicting text entropies from whisper features."""
from typing import Literal
from math import sqrt
import random
import torch
from torch import Tensor
from torch import nn
from transformers import WhisperModel

torch.manual_seed(1797)
random.seed(1797)

class EntropyWhisper(nn.Module):
    """A module trained to predict text entropy from whisper features."""
    def __init__(self, checkpoint):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(checkpoint)
        for param in self.whisper.parameters():
            param.requires_grad = False
        d_model = self.whisper.config.d_model
        self.w = nn.Parameter(torch.Tensor(d_model), requires_grad=True)
        nn.init.normal_(self.w, mean=0, std=sqrt(2 / (2 * d_model)))
    
    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            # extract Whisper contextual representation of the input speech
            whisper_outputs = self.whisper.encoder(x)
            hidden_states = whisper_outputs.last_hidden_state # out dim in [batch_size, seq_len, d_model]
        pooled = hidden_states.mean(1) # out dim in [batch_size, d_model]
        return pooled @ self.w
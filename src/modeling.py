# src/modeling.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class HalluModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float, proj_dim: int, pooling_strategy: str = "mean"):
        super().__init__()
        if pooling_strategy not in ["mean", "attention"]:
            raise ValueError("pooling_strategy must be one of 'mean' or 'attention'")

        self.pooling_strategy = pooling_strategy
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_labels),
        )
        
        if self.pooling_strategy == "attention":
            self.attention_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 1, bias=False)
            )

        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, proj_dim),
        )

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom
        
    def _attention_pool(self, last_hidden_state, attention_mask):
        attention_scores = self.attention_head(last_hidden_state).squeeze(-1)
        
        mask = attention_mask.to(attention_scores.dtype)
        adder = (1.0 - mask) * -10000.0
        attention_scores = attention_scores + adder
        
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # --- THAY ĐỔI CUỐI CÙNG Ở ĐÂY ---
        # Ép kiểu các tensor lên float32 trước khi nhân và cộng dồn để tránh tràn số
        pooled_output = torch.sum(
            attention_weights.unsqueeze(-1).float() * last_hidden_state.float(), 
            dim=1
        )
        
        return pooled_output

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        
        if self.pooling_strategy == "mean":
            pooled = self._mean_pool(hidden, attention_mask)
        elif self.pooling_strategy == "attention":
            pooled = self._attention_pool(hidden, attention_mask)
        
        logits = self.classifier(pooled)
        return logits
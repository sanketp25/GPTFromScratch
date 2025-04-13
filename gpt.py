import torch
import torch.nn as nn
import tiktoken
from attention import MultiHeadAttention

GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    def forward(self, x):
        return x

class DummyGPTModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.token_emb(in_idx)
        pos_indices = torch.arange(seq_len).to(in_idx.device)
        pos_embeds = self.pos_emb(pos_indices)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits



tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
# print(batch) tensor([[6109, 3626, 6100,  345],
        #              [6109, 1110, 6622,  257]])


torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits) # two batches, 4 words (tokens) each of size 50257


"""
Implementing Layer Normalization. This ensures the ouputs have a mean of 0 and variance of 1
Helps in convergence.
Used before and after Multi-head attention

"""


class LayerNorm(nn.Module):
    def __init__(self, emb_dim) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # embedding dimension
        var = x.var(dim=-1, keepdim=True, unbiased=False) # unbiased is basicallyu divsion by n instead of n-1
        norm_x = (x -mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


"""
scale, shift are model parameters that LLM can use to improve model parameters
x.shape = batch_size, num_tokens, embedding_size
n: DoF
"""

torch.manual_seed(123) 
"""
The below line creates a batch of 2 samples, containing 5 features eacj
"""
batch_example = torch.randn(2, 5)
# layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
# out = layer(batch_example)
# print(out)
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True, unbiased=True) # basel's correction
print("Mean: \n",mean)
print("Variance: \n",var)

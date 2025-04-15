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



"""
Implementing the Gelu Activation function, improve performance as compared to ReLU, 
ReLU has a sharp corner at zero and outputs 0 for values <=0. 
GELU allows small, non-zero values for negative values. 
Has zero gradient at x = -0.75

"""

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (
            x + 0.044715 * torch.pow(x, 3))
            )
    )


# Using GELU in feedforward network

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # 2, 3, 768 --> 2,3,3072 (batch, num_Tokens, emd_dim)
            GELU(), # 2,3,3072 ->  2,3,3072
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]) # 2,3, 3072 --> _ ,_, 768
        )
    def forward(self, x):
        return self.layers(x)    

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)    


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and layer_output == x.shape:
                x += layer_output
            else: x = layer_output
        return x

layer_sizes = [3,3,3,3,3,1]
sample_input = torch.tensor([[1.,0.,-1.]])
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, False)


def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
   
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
print_gradients(model_without_shortcut, sample_input) 
"""
layers.0.0.weight has gradient mean of 0.0005015344941057265
layers.1.0.weight has gradient mean of 0.0003395547391846776
layers.2.0.weight has gradient mean of 0.00463462620973587
layers.3.0.weight has gradient mean of 0.014383410103619099
layers.4.0.weight has gradient mean of 0.07439854741096497

As we progress from layer 4 to layer 0, gradients are reducing, resulting into vanishing gradient problem

"""           
print("-"*50)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)
print("-"*50)
print("Transformer Block")
print("-"*50)

"""
Creating the transformer block

"""

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        print("shape of x: ",x.shape)
        x = self.att(x)
        x = self.drop_shortcut(x) + shortcut

        shortcut = x
        x = self.norm2(x)    
        x = self.ff(x)
        x = self.drop_shortcut(x) + shortcut
        return x
    
torch.manual_seed(123)
x = torch.rand(2, 1024, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)    





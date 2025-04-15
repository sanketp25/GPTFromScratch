import torch
import tiktoken
from gpt import GPTModel, generate_text_simple

GPT_CONFIG_124M = {
"vocab_size": 50257,
"context_length": 256,
"emb_dim": 768,
"n_heads": 12,
"n_layers": 12,
"drop_rate": 0.1,
"qkv_bias": False
}
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special ={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
model=model,
idx=text_to_token_ids(start_context, tokenizer),
max_new_tokens=10,
context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# basic training 

inputs = torch.tensor(
    [[16833, 3626, 6100], # ["every effort moves",
    [40, 1107, 588]]) # "I really like"]
targets = torch.tensor(
    [[3626, 6100, 345 ], # [" effort moves you",
    [1107, 588, 11311]]) # " really like chocolate"]

with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)
print(probas.shape) # 2x3x50257 --> batch, token and vocab size
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)
print("Shape of token_ids", token_ids.shape) # 2x3x1

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:"
f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
"""
Indexing is used here, iterative way is:

target_probas_1 = []
for i in range(3):  # 3 tokens in the sequence
    correct_token_id = targets[0, i]
    prob = probas[0, i, correct_token_id]
    target_probas_1.append(prob)

target_probas_1 = torch.stack(target_probas_1)

from the given vocab, the correct follow up token is probas[batch, token_id, correct_next_token_id]


"""
print("Text 1:", target_probas_1)
text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

"""
Working with logarithms of probability scores is more manageable in mathematical
optimization than handling the scores directly

"""
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
""" 
In deep learning, the commmon practice isn't to push the average log probability up to 0 but rather to bring the
negative average log probability down to 0.
"""
neg_avg_log_probas = avg_log_probas * -1 # creating negative probability
print(neg_avg_log_probas) # tensor(10.9728)

print("Logits shape:", logits.shape) # 2x3x50257
print("Targets shape:", targets.shape) #2x3


"""
Applying cross entropy loss function, flatten the tensors over batch dimension.

Because nn.CrossEntropyLoss doesn’t operate over 3D (batch x seq_len x vocab) — it only works over (N, C) + (N,)

N = number of samples
C = number of classes
"""
logits_flat = logits.flatten(0,1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss) # tensor(10.9728)

"""
Loss generated using cross_entropy is the same as neg_avg_log_probas.
This means, we can directly use out of gpt model and perform cross_entropy

"""
import torch
import tiktoken
from gpt import GPTModel, generate_text_simple
from embeddings import create_dataloader_v1

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

"""
Converts texts to tokens.

Inputs: 
    input text: str
    tokenizer: tokenizer (bpe)
Returns:
    tokenized tensor, with batch dimension.    
"""

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special ={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # adds the batch dimension
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
This means, we can directly use output of gpt model and perform cross_entropy

"""

with open("./verdict.txt","r",encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = tokenizer.encode(text_data)
print("Characters: ",total_characters)
print("Tokens: ",len(total_tokens))

train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)
train_loader = create_dataloader_v1(
    txt=train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"], # 256
    stride=GPT_CONFIG_124M["context_length"], # 256
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    txt=val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"], # 256
    stride=GPT_CONFIG_124M["context_length"], # 256
    drop_last=False,
    shuffle=False,
    num_workers=0
)

print("Train loader:")
for x, y in train_loader: # executes 9 times
    print(x.shape, y.shape) 
print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)


"""
Calculates loss of the given batch
"""
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    logits_flat = logits.flatten(0,1)
    targets_flat = target_batch.flatten()
    # print("Flattened logits:", logits_flat.shape) # 512x50257
    # print("Flattened targets:", targets_flat.shape) # 512
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    return loss    

"""
Compute train and validation loss
"""

def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0
    if not len(data_loader):
        return float("nan") 
    elif not num_batches:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device) 
            print("Loss after function is called: ",loss)   
            total_loss += loss.item()
        else: break
    return total_loss / num_batches        
                              

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)    

"""
After generating text, finding a function for evaluating the result and creating a function to 
calculate train and validation loss, now is the time to train the model.

"""

def train_model_simple(model, train_loader, val_loader, device, optimizer, num_epochs,
                    eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, "
                        f"Val loss {val_loss:.3f}"
                )
        generate_and_print_sample(model, tokenizer, device, start_context) 
    return train_losses, val_losses, track_tokens_seen           

   
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model, idx=encoded,max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()    

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device,
                                                        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
                                                        start_context="Every effort moves you", tokenizer=tokenizer)


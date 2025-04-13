import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from torch.nn import Embedding

class GPTDatasetV1(Dataset):
    """
    Of the type Pytorch Dataset.

    Creates Dataset for efficient DataLoader application

    Args:
        txt: string -> Contains  text to be tokenized
        tokenizer: embeddings
        max_length,stride -> int 
    
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        self.txt = txt
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        token_ids = self.tokenizer.encode(self.txt)
        for i in range(0,len(token_ids) - self.max_length, self.stride):
            input_chunk = token_ids[i:i+self.max_length]
            target_chunk = token_ids[i+1:i+self.max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)   
         
    def __getitem__(self, index):
        return self.input_ids[index],self.target_ids[index]
    
def create_dataloader_v1(txt,batch_size = 4, max_length =256, stride=128,shuffle=True, drop_last = True,num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt,tokenizer,max_length,stride)
    datalaoder = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )
    return datalaoder

with open("./verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()

max_length = 4
batch_size = 8
stride = 4
dataloader = create_dataloader_v1(raw_text, batch_size=batch_size, max_length=max_length, stride=stride, shuffle=False)
data_iter = iter(dataloader)
inputs,target = next(data_iter)
print(f"Inputs: \n {inputs}")
print(f"Targets: \n {target}")
print("Next Batch...")
# second = next(data_iter)
# print(second)


"""
Here a matrix is created of the dimensions batch_size * max_length (8x4)
This matrix consists of token ids, coming from GPT2, tiktoken

To create token embedings, we need to create embeddings through pytorch.
These embeddings are trained using backpropogation

"""
vocab_size, output_dim = 50257, 256
embedding_layer = Embedding(vocab_size,output_dim)
print(embedding_layer.weight)

token_embeddings = embedding_layer(inputs)
"""
Creates vector of 256 dimension for each word, in words of length 4 in a batch size of 8
"""

print(token_embeddings.shape) # 8x4x256


"""
POSITIONAL EMBEDDINGS:

Absolute:

embeddings are directly associated with specific positions in a sequence. For each posi-
tion in the input sequence, a unique embedding is added to the token's embedding to
convey its exact location. eg: 
token_embeddings = 1 1 1   1 1 1   1 1 1
pos =          1.1 1.2 1.3        3.1 3.2 3.3


Relative:

emphasis of relative positional embeddings is on the relative position or distance between tokens. This means
the model learns the relationships in terms of “how far apart” rather than “at which
exact position.” The advantage here is that the model can generalize better to sequences
of varying lengths, even if it hasnt seen such lengths during training.



****GPT uses absolute postional *****
"""

context_lenth = max_length # 4 in our case
pos_embedding_layer = Embedding(context_lenth, output_dim)  # 4 x 256 in this case
"""
In the above line, a lookup is created of the size 4 x 256, in the next line we are getting the embedding vectors corresponding to the
indices. i.e vector at position 0, vector at 1 and so on.  
"""
pos_embeddings = pos_embedding_layer(torch.arange(context_lenth)) # torch.arange(4) = [0, 1, 2, 3]
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print("The final embeddings are: ")
print(input_embeddings.shape)


        
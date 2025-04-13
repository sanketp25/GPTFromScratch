import torch
import torch.nn as nn

inputs = torch.tensor(
[[0.43, 0.15, 0.89],  # YOUR      x1
 [0.55, 0.87, 0.66], # journey
 [0.57, 0.85, 0.64], # starts
 [0.22, 0.58, 0.33], # with
 [0.77, 0.25, 0.10], # one
 [0.05, 0.80, 0.55]] # step       x6
)


"""
This is a simple attention mechanism, does not have trainable weights

"""

query = inputs[1] # this serves as the query, multiplies the embedding vectors of all other inputs
attn_scores_2 = torch.empty(inputs.shape[0])  # creates a vector of size 6, initialized to 0
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
# print(attn_scores_2) #  tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])  
# higher the values of attn_scores[i], the more it focuses on the element

# shape of attn_scores_2 is 6, => normalization across the single column
attn_weights_2 = torch.softmax(attn_scores_2,dim=0)

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
# print(context_vec_2)  # --> z2    

"""

Basically what we are doing is:

1. Calculate attn score by taking dot product of every input with every other input,
in this case, input 2 with all inputs from 1,6.
2. Normalize the score using softmax
3. Initialize the context vector of same shape as query
4. Taking the dot product of attn scores and inputs


this is done for one sample, now we do it for matrix

"""



attn_scores = inputs @ inputs.T
# print(attn_scores)
# print("-"*50)

attn_weights = torch.softmax(attn_scores,dim=-1) # sum across each row is 1
# print(attn_weights)


all_context_vecs = attn_weights @ inputs
# print("-"*50)
# print(all_context_vecs)





"""
Creating a self attention mechanism with trainable parmas, will calculate q,k,v

"""


x_2 = inputs[1] # second embedding
d_in = inputs.shape[1] # this is of shape 3
d_out = 2  # to avoid confusion, generally is of dimension equal to d_in



torch.manual_seed(42)

"""
torch.nn.Parameter: Wraps the tensor so that it can be registered as a model parameter.
torch.rand(d_in,d_out): Creates a tensor of size 3x2 (here) with random values (0 to 1) following uniform distribution.
requires_grad=False : Dont update weights

	“Make this tensor part of the model, but don't train it.”
"""
W_query = torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad=False)


query_2 = x_2 @ W_query    # (1x3) (3x2) --> (1x2)
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print("Query 2: ",query_2)


"""
Here we have calculate query, key and value ofr 1 vector, i.e input 1,
 after this we will do it for all inputs, calculate dot product
"""

keys = inputs @ W_key # 6x3 X 3x2 --> 6x2
values = inputs @ W_value
# print(f"Keys shape: {keys.shape} \n Values Shape: {values.shape}") # torch.Size([6, 2]) 

"""
Begin calculating attention scores, as usual start with single vector
"""

keys_2 = keys[1]
# print("----Keys_2----")
# print(keys_2)
# print("Shape of keys_2: ",key_2.shape)

# attn_score_22 = query_2 @ keys_2  
attn_score_22 = query_2.dot(keys_2)
# print(attn_score_22)


"""
Matrix, however this still corresponding to input 2

input 2 with respect to all the other inputs, essentially
"""

attn_scores_2 = query_2 @ keys.T   #1x2 X 2x6
# print(attn_scores_2) # 1x6

"""
Now calculate attention weights
i.e normalize the scores
"""

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 /d_k ** 0.5 ,dim=-1)
# print(attn_weights_2) # 1x6 

"""
Calculate context Vectors

"""

context_vec_2 = attn_weights_2 @ values    # ---> 1x6 X 6x2
# print(context_vec_2) # 1x2


# Now we implement the matrix multiplication

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out) -> None:
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out)) # 3x2
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        """
        x: input embeddings

        Returns: context vector of size x.shape[0], d_out 
        """    
        keys = x @ self.W_key   # 6x3 X 3x2 ---> 6x2
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T # 6x2 X 2x6 ---> 6x6
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values # 6x6 X 6x2 --> 6x2
        return context_vec 
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
# print("Context vectors")
# print(sa_v1(inputs))


""" 
Perfoming the same with nn.Linear module, has 2 advantages:

1. nn.Linear layers, effectively perform matrix multiplication when
the bias units are disabled.
2. Instead of manually implementing nn.Parameter(torch.rand(...)) is that nn.Linear
has an optimized weight initialization scheme, contributing to more stable and
effective model training.
"""



class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False) -> None:
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # since this pytorch's linear layer, weight is of the type [d_out, d_in]
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        """
        x: input embeddings

        Returns: context vector of size x.shape[0], d_out 

        """    
        keys = self.W_key(x)  
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T # 6x2 X 2x6 ---> 6x6
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values # 6x6 X 6x2 --> 6x2
        return context_vec 
    
torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)
# print("Context vectors")
# print(sa_v2(inputs))


"""
Causal Attention

Now we mask the attn weights that the model is yet to see

"""
# inputs: 6x3
# w_* : 3x2
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
values = sa_v2.W_value(inputs)

attn_scores = queries @ keys.T # 6x2 X 2x6 --> 6x6
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
print("Keys.shape[-1]: ",keys.shape[-1])
print(attn_weights)


"""
Creating the max using tril function
"""
print("attn_scores.shape[0]: ",attn_scores.shape[0])
context_length = attn_scores.shape[0] # 6
mask_simple = torch.tril(torch.ones(context_length, context_length)) # lower triangle --> tri + l
print(mask_simple) # 6x6 lower matrix is created

"""
Once mask is ready, use it to zero-out the values from the attn_weights
"""
masked_simple = attn_weights * mask_simple   # this is element wise multiplication, not a dot product
print(masked_simple)

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

"""
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x.sum(dim=-1))           # tensor([ 6, 15])      --> shape [2]
print(x.sum(dim=-1, keepdim=True))  # tensor([[ 6], [15]])  --> shape [2, 1]
"""

# Not using softmax

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print(mask)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim = -1)
print(attn_weights)

"""
Applying Dropout 
"""

torch.manual_seed(42)
dropout = torch.nn.Dropout(0.5)
# eg = torch.ones(6, 6)
# print(eg)
# print(dropout(eg))

print(dropout(attn_weights))



"""
Implementing the above in a class

"""

class CausalAttention(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 context_length,
                 dropout,
                 qkv_bias = False   
                 ) -> None:
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias= qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', 
                             torch.triu(torch.ones(context_length, context_length),diagonal=1)
                             )
    def forward(self, x):
        b, num_tokens, d_in = x.shape    # batch, input shape --> 2x6x3 here
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vector = attn_weights @ values
        return context_vector
    
torch.manual_seed(123)
batch = torch.stack((inputs,inputs),dim=0) #2x6x3
print(batch.shape)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)    

"""
Multi Head attention -- Simple

This stacking the context vectors of multiple causal llms alongside each other.

"""


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, 
                 d_in,
                 d_out,
                 context_length,
                 dropout,
                 num_heads,
                 qkv_bias=False
                 ) -> None:
        super().__init__()
        self.heads=nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2 
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)   


"""
Now implementing the Multihead attention for efficient matrix multiplication in parallel.

"""

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 d_in,
                 d_out,
                 context_length,
                 dropout,
                 num_heads,
                 qkv_bias=False
                 ) -> None:
        super().__init__()
        assert (d_out % num_heads ==0,"d_out must be divisible by num_heads")
        self.d_out = d_out
        self.num_heads=num_heads
        self.head_dim = d_out // num_heads  #2/2 = 1
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias= qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length),diagonal=1)) 
    def forward(self, x):
        b, num_tokens, d_in = x.shape # 2x6x3
        queries = self.W_query(x) #2x6x2
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim) # 2x6x2x1
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        #Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)

        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values =values.transpose(1,2)

        # this done to perfom calculations on the heads

        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:context_length, :context_length]
        attn_scores.masked_fill(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1,2) # b, num_tokens, n_heads, head_dim --> original
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # required 2x6x2, flatten the vector
        context_vec = self.out_proj(context_vec) # optional, but used in LLM arch
        return context_vec

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)




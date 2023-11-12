# Resources:
# https://github.com/karpathy/ng-video-lecture
# https://www.youtube.com/watch?v=kCc8FmEb1nY
# https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=EcVIDWAZEtjN

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu' #allows to run on GPU (if available) - uses nvidia CUDA, makes it a lot faster
eval_iters = 200
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# character-level tokenizer: (usually models use sub-word tokenizer)
# now we need to tokenize our characters, which is basically mapping them to integers 
# based on some sort of vocabulary 
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

test_string = "yo"
print("'yo' encoded into tokens: (each character is a token) ", encode(test_string)) #encode characters into tokens (integers)
print("decoded tokens: ", decode(encode(test_string))) #decode tokens into characters

# let's now encode the entire text dataset and store it into a torch.Tensor
#Train and test splits
data = torch.tensor(encode(text), dtype=torch.long) #store our encoded dataset in a tensor
#print(data[:100]) # the 100 characters we looked at earier will to the GPT look like this

# Let's now split up the data into train and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

#x = train_data[:block_size]
#y = train_data[1:block_size+1]
#for t in range(block_size):
#    context = x[:t+1] #this is input, the context that our model gets (from 1 to 8 characters)
#    target = y[t] #this is expected output? what should model generate, based on the context?
#    print(f"when input is {context} the target: {target}")

#GPUs are great at parallel processing so let's add multiple batches to process?

#data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#let's feed our input to our neural network (the simplest one - bigram language model is used here)
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) #this measures quality of logits in respect to targets? 
            #so basically we have the correct output (targets) so we can calculate how well we are predicting

        return logits, loss
    
    #generate function for our model, that will take (Batch, Time) and extend it to (B, T+1), (B, T+2) and so on
    #basically continue the generation of all batch dimensions in time dimension for max_new_tokens
    def generate(self, idx, max_new_tokens): 
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

#let's see what our unoptimized model spits out
context = torch.zeros((1, 1), dtype=torch.long, device=device)  
print("-----------------------------------------------------------------")                             
print("PREDICTED TEXT - UNOPTIMIZED: \n", decode(m.generate(context, max_new_tokens=500)[0].tolist())) 
print("-----------------------------------------------------------------")
#so let's actually train the model (in a sec)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#Training Loop:
for iter in range(max_iters): # increase number of steps for good results... #more steps, the smaller loss.item() gets?
        # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    
    #zero-out gradients
    optimizer.zero_grad(set_to_none=True)
    
    #get gradients for all parameters and then update them - WE ARE OPTIMIZING 
    loss.backward()
    optimizer.step()

print("final loss: ", loss.item()) #you can experiment with the number of steps in optimization loop and see what loss u get

#lets try out our model AFTER optimization (u can also compare results here with different optimization steps num)
context = torch.zeros((1, 1), dtype=torch.long, device=device) #Batch = 1, Time = 1. We create 1 by 1 tensor which holds a 0
                                            #0 is a newline character, we use it as the first thing that we feed into model

#we feed in our idx as input, we ask for 100 tokens to be generated, we then index[0]throw to get 1 dimension batch 
#we basically get time steps, single dimensional batch, then convert to list, then decode and we have our predicted text!                                         
print("-----------------------------------------------------------------")
print("PREDICTED TEXT - OPTIMIZED: \n", decode(m.generate(context, max_new_tokens=500)[0].tolist())) 
print("-----------------------------------------------------------------")
#with 10k optimization steps I got something that resembles words, much better than before but still no good enough

#our model looks ONLY AT THE LAST CHARACTER to make predictions! it doesn't communicate with other tokens also






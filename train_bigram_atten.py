import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint
from data_preprocessing import (read_file,
                                 vocab_file,
                                 Tokenizer)


#############CONSTANTS###################
train_split = 0.8
batchsize = 8
context = 16
embedding_dims = 32
num_heads = 4
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

################################
def generate_batch( documents: list, 
                    batchsize: int, 
                    context: int):

    docu_len = documents.shape[0]

    # select a random index each document
    time_idx = [torch.randint(docu_len - context, (1,)) for i in range(batchsize)]
    samp_docs = [documents[t: t+context] for t in time_idx]

    x = torch.stack(samp_docs)
    # shift the target by one position
    y = torch.stack([documents[t+1: t+context+1] for t in time_idx])

    x = x.to(device)
    y = y.to(device)
    
    return x, y

class Head(nn.Module):
    """ One self attention head """
    def __init__(self, n_embed, head_size):
        super().__init__()
        self.query_layer = nn.Linear(n_embed, head_size, bias=False)
        self.key_layer = nn.Linear(n_embed, head_size, bias=False)
        self.value_layer = nn.Linear(n_embed, head_size, bias=False)
        # lower triangular matrix of a torch.ones
        self.register_buffer('mask', torch.tril(torch.ones(context, context)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key_layer(x)
        q = self.query_layer(x)
        
        # compute self attention scores ("affinities")
        wei = q@k.transpose(-2, -1) * C**-0.5
        # stop at time step just to be efficient
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value_layer(x)
        out = wei@v

        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, n_embed, num_heads):
        super().__init__()
        self.head_size = n_embed // num_heads
        self.heads = nn.ModuleList([Head(n_embed, self.head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class BigramLanguageAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dims, num_heads):
        super(BigramLanguageAttentionModel, self).__init__()
        self.head_size = embedding_dims // num_heads
        # embed the entire vocabulary size
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dims)
        # embed the position of the word in the context
        self.positional_embedding_table = nn.Embedding(context, embedding_dims)
        self.sa_head = MultiHeadAttention(embedding_dims, num_heads)
        self.lm_head = nn.Linear(embedding_dims, vocab_size)

    
    def forward(self, idx, targets):
        """
        """
        loss = None
        token_embed = self.token_embedding_table(idx)
        pos_embed = self.positional_embedding_table(torch.arange(context, device=device))
        x = token_embed + pos_embed
        x = self.sa_head(x)
        logits = self.lm_head(x)

        if targets is not None:
            # we use view to retain the ordering of the vectors instead of reshape
            logits = logits.view(batchsize * context,  -1)
            targets = targets.view(batchsize * context)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # ensure that when generating, we have a maximum of the length of the context being pedicted
            idx_cond = idx[:, -context:]
            logits, _ = self.forward(idx_cond, None)
            logits_for_last_time_step = logits[:, -1, :]
            probs = F.softmax(logits_for_last_time_step, dim=1)
            # sample from a multinomial distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append to input
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


    def generate_and_show(self, idx, max_new_tokens):
        out = self.generate(idx, max_new_tokens)
        return [tokenizer.decode(x.tolist()) for x in out]


def encode_input(input_string):
    input_string = tokenizer.encode(input_string)
    inp_size = len(input_string)
    if inp_size < context:
        input_string = [0] * (context - inp_size) + input_string

    return torch.tensor(input_string, dtype=torch.long).to(device).reshape(1, -1)
            

if __name__ == '__main__':
    ############# DATA PREPROCESSING ###################
    processed_file_path = 'data/processed/kjv.txt'
    documents = read_file(processed_file_path)
    tokenizer = Tokenizer(None, vocab_file, True)
    # concat all documents into one string
    documents = ["".join(documents)]
    print(f"all documents are a single string: {len(documents)}\n")
    documents_tensor = [torch.tensor(tokenizer.encode(doc), dtype=torch.long) for doc in documents][0]
    
    ########## MODELLING #############
    vocab_size = len(tokenizer.vocabulary)
    epochs = 10000

    # because it is a bigram mode, embedding_dims = vocab_size  
    m = BigramLanguageAttentionModel(vocab_size, embedding_dims, num_heads).to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    print("***************Starting Training *****************")
    for steps in range(epochs):
        xb, yb = generate_batch(documents_tensor, batchsize, context)
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if steps % (epochs/ 10) == 0:
            print(f"Epoch: {steps} ", f"Loss: {loss.item():0.4f}")

    print("***************Training End *****************")
    sample_input = torch.zeros((1,1), dtype=torch.long).to(device)
    num_to_generate = 2000
    pprint(m.generate_and_show(sample_input, num_to_generate)[0])
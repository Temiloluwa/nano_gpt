import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint
from data_prep.tokenizer import (read_file,
                                 vocab_file,
                                 Tokenizer)


#############CONSTANTS###################
train_split = 0.8
batchsize = 8
context = 16
embedding_dims = 32
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


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dims):
        super(BigramLanguageModel, self).__init__()
        self.embedding_table = nn.Embedding(vocab_size, embedding_dims)

    
    def forward(self, idx, targets):
        """
        embedding layer is basically a dense layer with the following differences:
            1. the input is a one-hot encoded tensor
            2. since we want to embed the input, the size of the one-hot encoded tensor
                is the same as the entire vocabulary. We wanna dedicate a single position
                in the tensor to a token. This makes the dense layer weights effectively 
                a lookup table.
        """
        loss = None
        # logits shape (batch, num_tokens_in_sequence or time dimension, embedding_dims)
        logits = self.embedding_table(idx)
        if targets is not None:
            # we use view to retain the ordering of the vectors instead of reshape
            logits = logits.view(batchsize * context,  -1)
            targets = targets.view(batchsize * context)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx, None)
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
    m = BigramLanguageModel(vocab_size, embedding_dims=vocab_size).to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
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
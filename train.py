import torch
import torch.nn as nn
import torch.nn.functional as F

train_split = 0.8
batchsize = 8
context = 16
embedding_dims = 32

from data_preprocessing import (read_file,
                                 vocab_file,
                                 Tokenizer)


def generate_batch( documents: list, 
                    batchsize: int, 
                    context: int):

    # sample documents randomly, one document for one batch item
    batch_idx  = torch.randint(len(documents), (batchsize,))
    batch_docs = [documents[i] for i in batch_idx]
    # select a random index each document
    time_idx = [torch.randint(len(doc) - context, (1,)) for doc in batch_docs]
    x = torch.stack([doc[t: t+context] for doc, t in zip(batch_docs, time_idx)])
    # shift the target by one position
    y = torch.stack([doc[t+1: t+context+1] for doc, t in zip(batch_docs, time_idx)])
    
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
        # logits shape (batch, num_tokens_in_sequence or time dimension, embedding_dims)
        logits = self.embedding_table(idx)
        # we use view to retain the ordering of the vectors instead of reshape
        # the vocabulary size is the number of classes because we predicting a particular token given previous ones
        # this trivial case sets the embedding size as the number of classes
        # in a real network, the inputs are modified such that the last layer has a size equal to the vocabularize size
        # input reshape => (batch * num_tokens_in_sequence or time dimension, embedding_dims or classes )
        # target shape => (batch * num_tokens_in_sequence or time dimension)
        logits = logits.view(batchsize * context,  -1)
        targets = targets.view(batchsize * context)
        loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        

    


if __name__ == '__main__':
    processed_file_path = 'data/processed/kjv.txt'
    documents = read_file(processed_file_path)
    tokenizer = Tokenizer(None, vocab_file)
    documents_tensor = [torch.tensor(tokenizer.encode(doc), dtype=torch.long) for doc in documents]

    xb, yb = generate_batch(documents_tensor, batchsize, context)
    print("input: ", xb)
    print(xb.shape)
    print("output: ", yb)
    print(yb.shape)

    print("-----")

    for b in range(batchsize):
        for t in range(context):
            time_context = xb[b, :t+1]
            target = yb[b,t]
            print(f"when input is '{tokenizer.decode(time_context.tolist())}' and target is '{tokenizer.decode([int(target)])}'")
        
        print("********")

    vocab_size = len(tokenizer.vocabulary)
    # let the embedding dims == vocab_size
    m = BigramLanguageModel(vocab_size, embedding_dims=vocab_size)
    logits, loss = m(xb, yb)
    print(logits.shape)
    print(loss)

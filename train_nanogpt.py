import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_training.utils import *

class Head(nn.Module):
    """
    One self-attention head module.
    
    Args:
        n_embed (int): Size of the input embeddings.
        head_size (int): Size of the attention head.
        dropout (float, optional): Dropout probability
    """
    def __init__(self, n_embed, head_size, dropout):
        super().__init__()
        self.query_layer = nn.Linear(n_embed, head_size, bias=False)
        self.key_layer = nn.Linear(n_embed, head_size, bias=False)
        self.value_layer = nn.Linear(n_embed, head_size, bias=False)
        # Lower triangular matrix of torch.ones
        self.register_buffer('mask', torch.tril(torch.ones(context, context)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the self-attention head.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_size).
        
        Returns:
            torch.Tensor: Output tensor after self-attention of shape (batch_size, sequence_length, head_size).
        """
        k = self.key_layer(x)
        q = self.query_layer(x)
        
        # dk in the paper
        B, T, C = k.shape

        # Compute self-attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        # Apply mask for efficient computation
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value_layer(x)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module.
    
    Args:
        n_embed (int): Size of the input embeddings.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """
    def __init__(self, n_embed, num_heads, dropout):
        super().__init__()
        self.head_size = n_embed // num_heads
        # Each head takes in the full embedding as input
        self.heads = nn.ModuleList([Head(n_embed, self.head_size, dropout) for _ in range(num_heads)])
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embed, n_embed)
    
    def forward(self, x):
        """
        Forward pass of the multi-head self-attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_size).
        
        Returns:
            torch.Tensor: Output tensor after multi-head self-attention of shape (batch_size, sequence_length, embedding_size).
        """
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """
    Feed-forward neural network module used in the Transformer block.
    
    Args:
        n_embed (int): Size of the input embeddings.
        dropout (float): Dropout probability.
    """
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout))

    def forward(self, x):
        """
        Forward pass of the feed-forward neural network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_size).
        
        Returns:
            torch.Tensor: Output tensor after feed-forward computation of shape (batch_size, sequence_length, embedding_size).
        """
        return self.net(x)


class Block(nn.Module):
    """
    Transformer Block: communication followed by computation.
    """
    def __init__(self, n_embed, num_heads, dropout):
        super().__init__()
        self.sa_head = MultiHeadAttention(n_embed, num_heads, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        """
        Forward pass of the Transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_size).
        
        Returns:
            torch.Tensor: Output tensor after applying self-attention and feed-forward computations of shape (batch_size, sequence_length, embedding_size).
        """
        # Apply self-attention and feed-forward computations with skip connections
        x = x + self.sa_head(self.norm1(x))
        x = x + self.ffwd(self.norm2(x))
        return x


class NanoGPTModel(nn.Module):
    """
    Full-Bigram Language Attention Model.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dims (int): Dimensionality of the embedding space.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        num_blocks (int, optional): Number of transformer blocks. Default is 4.
    """
    def __init__(self, vocab_size, embedding_dims, num_heads, dropout, num_blocks=4):
        super(NanoGPTModel, self).__init__()
        self.head_size = embedding_dims // num_heads
        # Embed the entire vocabulary size
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dims)
        # Embed the position of the word in the context
        self.positional_embedding_table = nn.Embedding(context, embedding_dims)
        block_layers = [Block(embedding_dims, num_heads, dropout) for _ in range(num_blocks)] + [nn.LayerNorm(embedding_dims)]
        self.blocks = nn.Sequential(*block_layers)
        self.lm_head = nn.Linear(embedding_dims, vocab_size)
       
    def forward(self, idx, targets):
        """
        Forward pass of the Full-Bigram Language Attention Model.
        
        Args:
            idx (torch.Tensor): Input tensor of token indices of shape (batch_size, sequence_length).
            targets (torch.Tensor): Target token indices for loss computation of shape (batch_size, sequence_length).
        
        Returns:
            torch.Tensor: Logits tensor of shape (batch_size * sequence_length, vocab_size).
            torch.Tensor: Loss tensor (if targets are provided) for backpropagation.
        """
        loss = None
        token_embed = self.token_embedding_table(idx)
        pos_embed = self.positional_embedding_table(torch.arange(context, device=device))
        x = token_embed + pos_embed
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is not None:
            logits = logits.view(batchsize * context,  -1)
            targets = targets.view(batchsize * context)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss


if __name__ == "__main__":
    train_data, val_data = get_data(tokenizer, processed_file_path, train_split)
    gpt_model = NanoGPTModel(vocab_size, 
                            embedding_dims, 
                            n_heads,
                            dropout,
                            n_blocks).to(device)
    optimizer = torch.optim.Adam(gpt_model.parameters(), lr=lr)
    gpt_model = train_model(gpt_model, optimizer, train_data, val_data,  batchsize, context)
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"model_training/models/model_{current_datetime}_ds_{dataset}_voc_{vocab_size}_emb{embedding_dims}_hd{n_heads}_dp_{str(dropout).replace('.', 'p')}_blk{n_blocks}_cxt{context}_lr{str(lr).replace('.', 'p')}_eph{str(epochs)}.pth"
    save_model(gpt_model, save_path)
    print("Saved model @", save_path)

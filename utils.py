import torch
import torch.nn.functional as F
import yaml
from typing import Tuple
from data_preprocessing import process_data, Tokenizer


def load_hyperparameters_from_yaml(yaml_filename):
    """
    Load hyperparameters from a YAML file.

    Args:
        yaml_filename (str): Path to the YAML file containing hyperparameters.

    Returns:
        dict: Dictionary containing the loaded hyperparameters.
    """
    with open(yaml_filename, 'r') as yaml_file:
        hyperparameters = yaml.safe_load(yaml_file)

    return hyperparameters

constants = load_hyperparameters_from_yaml('hyper_params.yaml')
#### DATA PREP ##############################
dataset = constants['dataset']
input_file_path = constants['datasets'][dataset]['input_file_path']
processed_file_path = constants['datasets'][dataset]['processed_file_path']
vocab_file = constants['datasets'][dataset]['vocab_file']
process_data(input_file_path, processed_file_path, dataset)
tokenizer = Tokenizer(processed_file_path, vocab_file)
vocab_size = len(tokenizer.vocabulary)


#### HYPER PARAMS ##############################
train_split = constants['train_split']
batchsize = constants['batchsize']
context = constants['context']
embedding_dims = constants['embedding_dims']
n_heads = constants['n_heads']
lr = constants['lr']
dropout = constants['dropout']
n_heads = constants['n_heads']
n_blocks = constants['n_blocks']
epochs = constants['epochs']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
######################################################

def get_train_data(tokenizer, processed_file_path):
    """
    Read processed documents and tokenize them.

    Args:
        tokenizer: Tokenizer instance.
        processed_file_path (str): Path to the processed file.

    Returns:
        torch.Tensor: Tensor containing encoded documents for training.
    """
    with open(processed_file_path, 'r') as f:
        documents = f.read()

    print(f"All documents are a single string: {len(documents)/1e6:.1f} M tokens")
    documents_tensor = [torch.tensor(tokenizer.encode(doc), dtype=torch.long) for doc in documents]

    return documents_tensor



def generate_batch(documents: list, batchsize: int, context: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of input-output pairs for training.

    Args:
        documents (list): List of document tensors.
        batchsize (int): Size of the batch.
        context (int): Number of context tokens.

    Returns:
        torch.Tensor: Batch of input sequences of shape (batchsize, context).
        torch.Tensor: Batch of target sequences of shape (batchsize, context).
    """
    docu_len = len(documents)
    time_idx = [torch.randint(docu_len - context, (1,)) for _ in range(batchsize)]
    x = torch.stack([torch.tensor(documents[t: t+context], dtype=torch.long) for t in time_idx])
    y = torch.stack([torch.tensor(documents[t+1: t+context+1], dtype=torch.long) for t in time_idx])
    x = x.to(device)
    y = y.to(device)
    return x, y


def train_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                documents_tensor: torch.Tensor, batchsize: int, context: int) -> torch.nn.Module:
    """
    Train the model.

    Args:
        model (torch.nn.Module): PyTorch model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        documents_tensor (torch.Tensor): Tensor of documents.
        batchsize (int): Size of the batch.
        context (int): Number of context tokens.

    Returns:
        torch.nn.Module: Trained model.
    """
    model.train()
    for steps in range(epochs):
        xb, yb = generate_batch(documents_tensor, batchsize, context)
        if steps == 0:
            print(f"input shape: {xb.shape} output shape: {yb.shape}")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if steps % (epochs / 10) == 0:
            print(f"epoch: {steps} , loss: {loss.item():0.3f}")
    return model


def generate(model, idx, max_new_tokens):
    """
    Generate new tokens using the model.

    Args:
        model (nn.Module): Trained PyTorch model.
        idx (torch.Tensor): Starting input tensor.
        max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
        torch.Tensor: Generated sequence of tokens.
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context:]
        logits, _ = model.forward(idx_cond, None)
        logits_for_last_time_step = logits[:, -1, :]
        probs = F.softmax(logits_for_last_time_step, dim=1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)
    return idx


def generate_and_show(model, idx, max_new_tokens):
    """
    Generate and decode tokens using the model.

    Args:
        model (nn.Module): Trained PyTorch model.
        idx (torch.Tensor): Starting input tensor.
        max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
        list: List of generated and decoded strings.
    """
    out = generate(model, idx, max_new_tokens)
    return [tokenizer.decode(x.tolist()) for x in out]


def encode_input(tokenizer, input_string, context):
    """
    Encode an input string using the tokenizer.

    Args:
        tokenizer: Tokenizer instance.
        input_string (str): Input string to be encoded.
        context (int): Number of context tokens.

    Returns:
        torch.Tensor: Encoded input tensor.
    """
    input_string = tokenizer.encode(input_string)
    inp_size = len(input_string)
    if inp_size < context:
        input_string = [0] * (context - inp_size) + input_string
    return torch.tensor(input_string, dtype=torch.long).to(device).view(1, -1)


def save_model(model, model_path):
    """
    Save a PyTorch model's state dictionary to a file.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        model_path (str): The path where the model will be saved.
    """
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def load_model(model, model_path):
    """
    Load a saved PyTorch model's state dictionary and create an instance of the model.

    Args:
        model_class (type): The class of the PyTorch model to be instantiated.
        model_path (str): The path from which the model will be loaded.

    Returns:
        torch.nn.Module: The loaded PyTorch model instance.
    """
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {model_path}")
    return model


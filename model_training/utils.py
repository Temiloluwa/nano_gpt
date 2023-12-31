import torch
import torch.nn.functional as F
import yaml
import numpy as np
import os
import re
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from typing import Union, List, Tuple, Dict
from .data_prep import *


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

config = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
config = load_hyperparameters_from_yaml(config)
#### DATA PREP ##############################
dataset = config['dataset']
input_file_path = os.path.join(os.path.dirname(__file__), f'data/raw/{dataset}.txt')
processed_file_path = os.path.join(os.path.dirname(__file__), f'data/processed/{dataset}.txt') 
vocab_file = os.path.join(os.path.dirname(__file__), f'data/vocab/{dataset}.json') 
process_fn = config['process_fn'][dataset] # get function as string
eval(f'{process_fn}(input_file_path, processed_file_path, dataset)') # evaluate the preprocessor
tokenizer = Tokenizer(processed_file_path, vocab_file)
vocab_size = len(tokenizer.vocabulary)


#### HYPER PARAMS ##############################
train_split = config['train_split']
batchsize = config['batchsize']
context = config['context']
embedding_dims = config['embedding_dims']
n_heads = config['n_heads']
lr = config['lr']
dropout = config['dropout']
n_heads = config['n_heads']
n_blocks = config['n_blocks']
epochs = config['epochs']
eval_epochs = config['eval_epochs']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if config.get('device', None):
    device = config['device']

######################################################

def detach_tensor(torch_tensor: torch.Tensor) -> Union[torch.Tensor, 'numpy.ndarray']:
    """
    Detaches a PyTorch tensor from the computation graph and converts it to a NumPy array if on CUDA device.

    Args:
        torch_tensor (torch.Tensor): The input PyTorch tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: The detached tensor as a NumPy array if on CUDA, otherwise the original tensor.
    """
    if torch_tensor.device.type == 'cuda':
        torch_tensor = torch_tensor.detach().cpu().numpy()
    
    return torch_tensor

def get_model_signature(model: torch.nn.Module, data_tensor: torch.Tensor) -> str:
    """
    Gets the model signature based on provided input tensor.

    Args:
        model (torch.nn.Module): The PyTorch model.
        data_tensor (torch.Tensor): Input data tensor.

    Returns:
        str: The model signature.
    """
    xb, yb = generate_batch(data_tensor, batchsize, context)
    logits, _ = model(xb, yb)
    xb = detach_tensor(xb)
    logits = detach_tensor(logits.view(batchsize, context, -1))
    signature = infer_signature(xb, logits)
    return signature


def get_or_create_experiment(experiment_name):
    """
    Get an existing experiment by name or create a new one if it doesn't exist.

    Args:
        experiment_name (str): The name of the experiment.

    Returns:
        int: The experiment ID.
    """
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if existing_experiment:
        print(f"Using existing experiment '{experiment_name}' (ID: {existing_experiment.experiment_id})")
        return existing_experiment.experiment_id
    else:
        new_experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment '{experiment_name}' (ID: {new_experiment_id})")
        return new_experiment_id


def log_hyper_params(config: Dict[str, float]):
    """
    Logs hyperparameters to the active MLflow run.

    Args:
        config (Dict[str, float]): A dictionary containing hyperparameters.

    Example:
        config = {
            'train_split': 0.8,
            'batchsize': 32,
            'context': 10,
            'embedding_dims': 128,
            'n_heads': 4,
            'lr': 0.001,
            'dropout': 0.2,
            'n_blocks': 3,
            'epochs': 100,
            'eval_epochs': 10
        }
        log_hyper_params(config)
    """
    # Log hyperparameters
    mlflow.log_param("train_split", config['train_split'])
    mlflow.log_param("batchsize", config['batchsize'])
    mlflow.log_param("context", config['context'])
    mlflow.log_param("embedding_dims", config['embedding_dims'])
    mlflow.log_param("n_heads", config['n_heads'])
    mlflow.log_param("lr", config['lr'])
    mlflow.log_param("dropout", config['dropout'])
    mlflow.log_param("n_blocks", config['n_blocks'])
    mlflow.log_param("epochs", config['epochs'])
    mlflow.log_param("eval_epochs", config['eval_epochs'])


def prepare_model(model, train_data, model_name) -> None:
    """
    Prepare an experiment for logging a model with a specified name and signature.

    Args:
        model (Any): The trained model.
        train_data (Any): The training data used for the model.

    Returns:
        None
    """
    signature = get_model_signature(model, train_data)
    mlflow.pytorch.log_model(model, model_name, signature=signature)
    log_hyper_params(config)


def get_data(tokenizer, processed_file_path:str, train_split:float=None):
    """
    Read processed documents and tokenize them.

    Args:
        tokenizer: Tokenizer instance.
        processed_file_path (str): Path to the processed file.
        train_split(float): fraction to be train

    Returns:
        torch.Tensor: Tensor containing encoded documents for training.
    """

    train_tensor = None
    val_tensor = None
    with open(processed_file_path, 'r') as f:
        documents = f.read()

    print(f"The document has: {len(documents)/1e6:.1f} M tokens")
    mlflow.log_param("document_tokens", len(documents))
    documents_tensor = [torch.tensor(tokenizer.encode(doc), dtype=torch.long) for doc in documents]

    if train_split:
        split = int(train_split * len(documents_tensor)) 
        train_tensor = documents_tensor[:split]
        print(f"The train set has: {len(train_tensor)/1e6:.1f} M tokens")
        mlflow.log_param("train_tokens", len(train_tensor))
        val_tensor = documents_tensor[split:]
        print(f"The val set has: {len(val_tensor)/1e6:.1f} M tokens")
        mlflow.log_param("val_tokens", len(val_tensor))
    else:
        train_tensor = documents_tensor

    return train_tensor, val_tensor


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


@torch.no_grad()
def eval_model(model: torch.nn.Module, 
                val_tensor: torch.Tensor,
                batchsize: int, 
                context: int,
                eval_epochs: int) -> float:
    """
    Evaluate the model.

    Args:
        model (torch.nn.Module): PyTorch model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        val_tensor (torch.Tensor): Tensor of documents.
        batchsize (int): Size of the batch.
        context (int): Number of context tokens.

    Returns:
        torch.nn.Module: Trained model.
    """
    model.eval()
    val_losses = []
   
    for steps in range(eval_epochs):
        xb, yb = generate_batch(val_tensor, batchsize, context)
        _, loss = model(xb, yb)
        val_losses.append(loss.item())

    val_loss = np.mean(val_losses)
    
    return val_loss



def train_model(model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                train_tensor: torch.Tensor,
                val_tensor: torch.Tensor, 
                batchsize: int, 
                context: int,
                eval_prompt: str
                ) -> torch.nn.Module:
    """
    Train the model.

    Args:
        model (torch.nn.Module): PyTorch model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        train_tensor (torch.Tensor): Tensor of documents.
        batchsize (int): Size of the batch.
        context (int): Number of context tokens.

    Returns:
        torch.nn.Module: Trained model.
    """
    model.train()
    for steps in range(epochs):
        xb, yb = generate_batch(train_tensor, batchsize, context)
        if steps == 0:
            print(f"input shape: {xb.shape} output shape: {yb.shape}")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if steps % (epochs / 10) == 0:
            print(f"epoch: {steps} , train loss: {loss.item():0.3f}")
            mlflow.log_metric("train_loss", loss.item())

            if val_tensor is not None:
                val_loss = eval_model(model, val_tensor, batchsize, context, eval_epochs)
            print(f"epoch: {steps} , val loss: {val_loss:0.3f} \n ================")
            mlflow.log_metric("val_loss", val_loss.item())
            response = model_generate(model, eval_prompt, 50)
            # https://mlflow.org/docs/latest/python_api/mlflow.llm.html#mlflow.llm.log_predictions
            mlflow.llm.log_predictions([{
                "epoch": steps,
            }], [response], [eval_prompt])

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


def encode_input(tokenizer, input_string, context, device):
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


def parse_model_filename(filename):
    """
    Parse a model filename to extract hyperparameter values.

    Args:
        filename (str): The model filename to be parsed.

    Returns:
        dict or None: A dictionary containing extracted hyperparameter values if the filename matches the expected pattern.
                      None if no match is found.

    Example:
        filename = "model_training/models/model_20230825_135640_ds_shakespeare_tiny_voc_65_emb384_hd6_dp_0p2_blk6_cxt256_lr0p0003_eph1.pth"
        parsed_values = parse_model_filename(filename)
        if parsed_values:
            print(parsed_values)
        else:
            print("No match found")
    """
    pattern = r"model_(\d+)_(\d+)_ds_(\w+)_voc_(\d+)_emb(\d+)_hd(\d+)_dp_([\dp]+)_blk(\d+)_cxt(\d+)_lr([\dp]+)_eph(\d+)\.pth"
    matches = re.match(pattern, filename)

    if matches:
        the_date, the_time, dataset, vocab_size, embedding_dims, n_heads, dropout, n_blocks, context, lr, epochs = matches.groups()
        
        embedding_dims = int(embedding_dims)
        n_heads = int(n_heads)
        dropout = float(dropout.replace('p', '.'))
        n_blocks = int(n_blocks)
        context = int(context)
        lr = float(lr.replace('p', '.'))
        epochs = int(epochs)
        vocab_size = int(vocab_size)

        return {
            "the_date": the_date,
            "the_time": the_time,
            "dataset": dataset,
            "vocab_size": vocab_size,
            "embedding_dims": embedding_dims,
            "n_heads": n_heads,
            "dropout": dropout,
            "n_blocks": n_blocks,
            "context": context,
            "lr": lr,
            "epochs": epochs
        }
    else:
        return None

def model_generate(model, prompt, max_tokens):
    """
    Generate a response using the provided model and prompt.

    Args:
        model (nn.Module): Trained GPT model.
        prompt (str): Input prompt for generating the response.
        max_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: Generated response.
    """
    prompt_encoded = encode_input(tokenizer, prompt, context, device)
    response = generate_and_show(model, prompt_encoded, max_tokens)
    response = response[0].split(prompt)[-1]

    return response
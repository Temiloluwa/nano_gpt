import os
import time
import random
import streamlit as st
from train_nanogpt import NanoGPTModel as GPTModel
from model_training.utils import (parse_model_filename,
                                 generate_and_show,
                                  encode_input,
                                  load_model,
                                  tokenizer,
                                  context)


pages = ["nanogpt", "model_stats"]
datasets = ("kjv", "shakespare_tiny", "shakespare_plays")
favicon_path = "pages/imgs/nano-gpt.ico"
hero_url = "pages/imgs/nano-pgt-hero.jpg"
st.set_page_config(page_title="NanoGPT", page_icon=favicon_path)
device = 'cpu'

def get_checkpoints(dataset: str) -> list[str]:
    """
    Get a list of checkpoints for the specified dataset.

    Args:
        dataset (str): The dataset name ("kjv", "shakespare_tiny", or "shakespare_plays").

    Returns:
        list[str]: List of checkpoint paths for the specified dataset.
    """

    def gen_short_names(path: str) -> str:
        """
        Generate short names for checkpoint paths.

        Args:
            path (str): Full checkpoint path.

        Returns:
            str: Short name for the checkpoint.
        """
        hyp = parse_model_filename(path.split('model_training/models/')[-1])
        if hyp is not None:
            return f"time_{hyp['the_date']+'_'+hyp['the_time']}_emb_{hyp['embedding_dims']}_heads{hyp['n_heads']}_blks{hyp['n_blocks']}_cxt{hyp['context']}_eph{hyp['epochs']}"

    ckpts_path = os.listdir("model_training/models")
    ckpts_path_full = ["model_training/models/" + x for x in ckpts_path]

    ckpts_full = {
        "kjv": [p for p in ckpts_path_full if 'kjv' in p],
        "shakespare_tiny": [p for p in ckpts_path_full if 'shakespare_tiny' in p],
        "shakespare_plays": [p for p in ckpts_path_full if 'shakespare_plays' in p]
    }

    ckpts_short = {
        "kjv": {gen_short_names(x): x for x in ckpts_full["kjv"]},
        "shakespare_tiny": {gen_short_names(x): x for x in ckpts_full["shakespare_tiny"]},
        "shakespare_plays": {gen_short_names(x): x for x in ckpts_full["shakespare_plays"]}
    }
    return ckpts_short[dataset]

    
@st.cache_resource
def model_select(model_path, device):
    """
    Load and return a cached GPT model based on the provided model path and device.

    Args:
        model_path (str): Path to the model checkpoint file.
        device (str): Device to load the model onto ("cpu" or "cuda").

    Returns:
        GPTModel: Loaded GPT model.
    """
    hypparams = parse_model_filename(model_path.split('model_training/models/')[-1])
    vocab_size = hypparams["vocab_size"]
    embedding_dims = hypparams["embedding_dims"]
    n_heads = hypparams["n_heads"]
    dropout = hypparams["dropout"]
    n_blocks = hypparams["n_blocks"]

    gpt_model = GPTModel(
        vocab_size,
        embedding_dims,
        n_heads,
        dropout,
        n_blocks
    )

    gpt_model = load_model(gpt_model, model_path).to(device)

    return gpt_model


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


def prompt_model(model, verbosity):
    """
    Generate responses using the provided model and display them in a chat interface.

    Args:
        model (nn.Module): Trained GPT model.
    """

    st.session_state.prompt_state = {
        "max_tokens": verbosity
    }

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Prompt NanoGPT......."):
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "👩‍💻"})

        with st.chat_message("user", avatar="👩‍💻"):
            st.markdown(prompt)

        with st.chat_message("nanoGPT", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            nano_gpt_response = prompt
            keep_generating = True
            
            while keep_generating: 
                nano_gpt_response = model_generate(model, nano_gpt_response, verbosity)
                for chunk in nano_gpt_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
                if len(full_response) > st.session_state.prompt_state['max_tokens']:
                    st.session_state.prompt_state['max_tokens'] = random.randint(50, verbosity)
                    keep_generating = False
             
        st.session_state.messages.append({"role": "nanoGPT", "content": full_response, "avatar": "🤖"})


def main():
    with st.sidebar:
        dataset = st.selectbox("Select a Dataset", datasets)
        checkpoints = get_checkpoints(dataset)
        verbosity = st.slider("Verbosity", min_value=50, max_value=1000, value=50)
        ckpt = st.radio("Choose a checkpoint", list(checkpoints.keys()))
        nanogpt_model = model_select(checkpoints[ckpt], device)
        

    col1, col2 = st.columns([0.15, 0.85])
    col1.image(hero_url, width=90) 
    col2.markdown("# NanoGPT")
    prompt_model(nanogpt_model, verbosity)


if __name__ == "__main__":
    main()
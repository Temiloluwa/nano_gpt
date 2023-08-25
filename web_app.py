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

if 'previous_device' not in st.session_state:
    st.session_state.previous_device = ''


def get_checkpoints(dataset: str) -> list[str]:
    
    def gen_short_names(path):
        # I got the right split by manually inspecting the string
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
        "kjv": {gen_short_names(x):x for x in ckpts_full["kjv"]},
        "shakespare_tiny": {gen_short_names(x):x for x in ckpts_full["shakespare_tiny"]},
        "shakespare_plays": {gen_short_names(x):x for x in ckpts_full["shakespare_plays"]}
    }
    return ckpts_short[dataset]

    
@st.cache_resource
def model_select(model_path, device):
    hypparams = parse_model_filename(model_path.split('model_training/models/')[-1])
    vocab_size = hypparams["vocab_size"]
    embedding_dims = hypparams["embedding_dims"]
    n_heads = hypparams["n_heads"]
    dropout = hypparams["dropout"]
    n_blocks = hypparams["n_blocks"]


    gpt_model = GPTModel(vocab_size, 
                    embedding_dims, 
                    n_heads,
                   dropout,
                   n_blocks)

    gpt_model = load_model(gpt_model, model_path).to(device)

    return gpt_model


def model_generate(model, prompt, device):
    max_tokens = random.randint(50, 500)
    prompt_encoded  = encode_input(tokenizer, prompt, context, device)
    response = generate_and_show(model, prompt_encoded, max_tokens)
    response = response[0].split(prompt)[-1]

    return response


def prompt_model(model, device):
    """
    Reference:https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
    """
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

            nano_gpt_response = model_generate(model, prompt, device)
            for chunk in nano_gpt_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "nanoGPT", "content": full_response, "avatar": "🤖"})



def main():
    with st.sidebar:
        dataset = st.selectbox("Select a Dataset", datasets)
        checkpoints = get_checkpoints(dataset)
        ckpt = st.radio("Choose a checkpoint", list(checkpoints.keys()))
        device = st.radio("Device", ("cpu", "cuda"))
        st.session_state.device = device
        
        if device != st.session_state.previous_device:
            st.session_state.previous_device = device
            st.experimental_rerun()

        nanogpt_model = model_select(checkpoints[ckpt], device)

    col1, col2 = st.columns([0.15, 0.85])
    col1.image(hero_url, width=90) 
    col2.markdown("# NanoGPT")
    prompt_model(nanogpt_model, device)



main()
import streamlit as st
import random
import time

pages = ["nanogpt", "model_stats"]
datasets = ("kjv", "shakespare_tiny", "shakespare_plays")
favicon_path = "app/imgs/nano-gpt.ico"
hero_url = "app/imgs/nano-pgt-hero.jpg"
st.set_page_config(page_title="NanoGPT", page_icon=favicon_path)


def get_checkpoints(dataset: str) -> list[str]:

    ckpts = {
        "kjv": ["kjv_model_1.pth", "kjv_model_2.pth"],
        "shakespare_tiny": ["shake_model_1.pth", "shahe_model_2.pth"],
        "shakespare_plays": ["plays_model_1.pth"]
    }

    return ckpts[dataset]


def model_generate():
    response = random.choice(
                [
                    "Hello there! How can I assist you today?",
                    "Hi, human! Is there anything I can help you with?",
                    "Do you need help?",
                ]
            )

    return response


def prompt_model():
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

            nano_gpt_response = model_generate()
            for chunk in nano_gpt_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "nanoGPT", "content": full_response, "avatar": "🤖"})

if __name__ == "__main__":
    with st.sidebar:
        dataset = st.selectbox("Select a Dataset", datasets)
        ckpts = get_checkpoints(dataset)
        checkpoints = st.radio("Choose a checkpoint", ckpts)

    
    col1, col2 = st.columns([0.15, 0.85])
    col1.image(hero_url, width=90) 
    col2.markdown("# NanoGPT")
    st.markdown(f"**DATASET**: {dataset.replace('_', ' ').capitalize()}")
    prompt_model()
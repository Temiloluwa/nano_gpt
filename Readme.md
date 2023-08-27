# Nano GPT

NanoGPT is a character level language model inspired by Andrej Karpathy's Tutorial @ [link](https://www.youtube.com/watch?v=kCc8FmEb1nY)


## Features
This repository contains
1. A training script for nanogpt `train_nanogpt.py`
2. A streamlit front end app
3. A config file to customize training or inference.

## Usage
Follow these steps to NanoGPT on your own data and chat with the model using the Streamlit front end.

1. NanoGPT currently supports a single file dataset. Place your data in `/model_training/data/raw/`
2. You can use the default data preprocessing function or create a custom one in `/model_training/data_prep/prep_data.py`
3.  Update the Data Prep part of `config.yaml` with the new dataset name and the it's processing function.
4. Train the model with `python train_nanogpt.py` and run the streamlit app afterwards using `streamlit run nanogpt_app.py`

## MLFlow Tracking

The training script features model tracking with MLFlow. 
You can monitor the training process by starting an mlflow server.

```bash
mlflow server --backend-store-uri sqlite:///model_training/data/mlflow/mlflow.db \
              --registry-store-uri  model_training/models \
              --host localhost\
              --port 5001 
```
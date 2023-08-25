# Nano GPT

Inspired by Andrej Karpathy's Tutorial @ [link](https://www.youtube.com/watch?v=kCc8FmEb1nY)


## MLFlow Tracking
```bash
mlflow server --backend-store-uri sqlite:///model_training/data/mlflow/mlflow.db \
              --registry-store-uri  model_training/models \
              --host localhost\
              --port 5001 
```
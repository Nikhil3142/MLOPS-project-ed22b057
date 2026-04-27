FROM python:3.10-slim
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install mlflow==2.13.0
WORKDIR /mlflow
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/Users/nikhil.narayan/Documents/GOOGLE DRIVE/Sem8/MLOPS/Mlops-Project/MLOPS-project-ed22b057/mlartifacts"]
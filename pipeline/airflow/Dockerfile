FROM apache/airflow:2.9.1-python3.10

USER airflow
RUN pip install --no-cache-dir \
    mlflow==2.13.0 \
    scikit-learn \
    xgboost \
    shap \
    numpy \
    pandas \
    joblib
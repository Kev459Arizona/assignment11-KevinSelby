import mlflow
import pandas as pd
import os
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Set the tracking URI from the environment variable
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Set the MLflow tracking URI
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Set MLflow experiment (you can change this as needed)
mlflow.set_experiment("/assignment-11local")

# Start an MLflow run
with mlflow.start_run():
    # Log some metrics (example values for now)
    mlflow.log_metric("foo", 1)
    mlflow.log_metric("bar", 2)

    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model hyperparameters
    params = {"solver": "lbfgs", "max_iter": 1000, "multi_class": "auto", "random_state": 8888}

    # Train the Logistic Regression model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Predict on the test set
    y_pred = lr.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Log the hyperparameters used in training
    mlflow.log_params(params)

    # Log the accuracy metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag for the model run to describe the training info
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the trained model with the registered model name
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",  # Artifact path within MLflow
        signature=signature,
        input_example=X_train,  # Provide input example to help with model inference
        registered_model_name="basic_lr_iris_model",  # Regis

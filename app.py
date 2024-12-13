from flask import Flask, request, jsonify
import mlflow
import numpy as np
import pandas as pd

# Load model from MLflow
model = mlflow.sklearn.load_model("models:/basic_lr_iris_model/1")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        input_data = request.json.get('data')
        
        # Ensure the data is in the correct format (list of 4 features for a single sample)
        input_array = np.array(input_data).reshape(1, -1)  # Reshape to a 2D array (1 sample with 4 features)

        # Predict using the model
        prediction = model.predict(input_array)
        
        # Return the prediction as a response
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)

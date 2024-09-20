from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
model = joblib.load('mentalHealth.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print(data)
    
    if not data:
        response = {"message": "No data received"}
        return jsonify(response), 400

    # Convert JSON data to a pandas DataFrame
    df = pd.DataFrame([data])  # Ensure data is in a single row
    
    # Save DataFrame to a CSV file
    df.to_csv('prediction.csv', index=False)

    # Load the CSV file for prediction
    csv_data = pd.read_csv('prediction.csv')

    # Make predictions using the model
    predictions = model.predict(csv_data)
    
    print("Received data:", data)
    
    return jsonify(predictions.tolist()), 200

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/members')
def members():
    return {"members":["member1","member2","member3"]}

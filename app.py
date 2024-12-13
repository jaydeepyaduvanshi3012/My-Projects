from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('svm_house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    # Render the home page
    return render_template('index.html')

# Route to accept JSON input from Postman
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request is coming from a form submission or Postman JSON input
    if request.content_type == 'application/json':
        data = request.get_json()  # Fetch JSON data from Postman
        features = [np.array([data['MedInc'], data['HouseAge'], data['AveRooms'], data['AveBedrms'], 
                              data['Population'], data['AveOccup'], data['Latitude'], data['Longitude']])]
    else:
        # Fetch input from HTML form
        data = request.form
        features = [np.array([float(data['MedInc']), float(data['HouseAge']), float(data['AveRooms']), 
                              float(data['AveBedrms']), float(data['Population']), float(data['AveOccup']),
                              float(data['Latitude']), float(data['Longitude'])])]

    # Scale the features
    scaled_features = scaler.transform(features)

    # Predict the house price
    prediction = model.predict(scaled_features)

    # Return response for JSON input
    if request.content_type == 'application/json':
        return jsonify({'predicted_house_price': float(prediction[0])})

    # Return response for HTML form input
    return render_template('index.html', prediction_text=f'Predicted House Price: {prediction[0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)

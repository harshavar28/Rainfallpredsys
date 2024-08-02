import pandas as pd
import joblib

# Load the label encoder, scaler, and imputer
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

# Define the features
features = ['SUBDIVISION', 'YEAR']

# Prediction function
def predict_rainfall(location, year, month):
    # Ensure location formatting consistency
    location = location.strip()

    # Convert input to a DataFrame with correct feature names
    input_data = pd.DataFrame([[location, year]], columns=features)

    # Encode location using the same label encoder
    input_data['SUBDIVISION'] = label_encoder.transform(input_data['SUBDIVISION'])

    # Scale the input data
    input_data = pd.DataFrame(scaler.transform(input_data), columns=features)

    # Load the model
    model = joblib.load(f'model_{month}.pkl')

    # Predict rainfall for the given input
    prediction = model.predict(input_data)

    # Return the predicted rainfall for the given month
    return prediction[0]

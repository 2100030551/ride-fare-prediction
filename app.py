from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

# Load the model and scaler
rf_model = joblib.load('ride_fare_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', fare=None, error=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the data from the POST request
        data = request.form

        # Ensure necessary fields are present and not empty
        required_fields = ['start_location', 'end_location', 'gender', 'vehicle_type']
        for field in required_fields:
            if not data.get(field):
                return render_template('index.html', fare=None,
                                       error=f"Please provide a valid {field.replace('_', ' ')}.")

        # Extract the start and end locations and calculate the distance (simulated here)
        start_location = data['start_location']
        end_location = data['end_location']
        distance_km = np.random.uniform(1, 20)  # Placeholder for actual distance calculation

        # Extract gender and vehicle type (vehicle_type -> vehicle_id)
        gender = int(data['gender'])
        vehicle_id = 0 if data['vehicle_type'].lower() == 'bike' else 1  # Convert vehicle_type to vehicle_id

        # Automatically generate start time (current time) and end time (30 minutes later)
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=30)  # Assuming a 30-minute ride duration

        # Calculate ride duration in minutes
        ride_duration = (end_time - start_time).seconds / 60  # Ride duration in minutes

        # Extract the date-time features (hour and day of the week) from the start time
        hour = start_time.hour
        day_of_week = start_time.weekday()

        # Prepare the features array
        features = np.array([[
            distance_km,  # distance
            hour,  # hour of the day
            data.get('age', 30),  # age (default to 30 if not provided)
            gender,  # gender
            vehicle_id,  # vehicle_id (renamed from vehicle_type)
            data.get('junction', 1),  # junction (default to 1 if not provided)
            ride_duration,  # ride duration
            data.get('time_of_day', 12),  # time of day (default to 12 if not provided)
            day_of_week  # day of the week
        ]])

        # Convert the features into a pandas DataFrame
        feature_columns = ['distance_km', 'hour', 'age', 'gender', 'vehicle_id', 'Junction', 'ride_duration',
                           'time_of_day', 'day_of_week']
        features_df = pd.DataFrame(features, columns=feature_columns)

        # Scale the relevant features (e.g., distance and ride duration)
        features_df[['distance_km', 'ride_duration']] = scaler.transform(features_df[['distance_km', 'ride_duration']])

        # Make the prediction
        prediction = rf_model.predict(features_df)

        # Return the prediction to the user
        fare = prediction[0]
        return render_template('index.html', fare=fare, error=None)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Something went wrong: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)

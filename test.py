import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

# Load the saved model
model = joblib.load('price_prediction_model.pkl')

# Prepare the input data (new car data for prediction)
car_details = {
    'make': ['Suzuki'],  # Car make
    'model': ['Alto'],   # Car model
    'city': ['Karachi'], # City where the car is located
    'year': [2019],      # Year of the car
    'mileage': [45744],  # Mileage of the car in km
    'Engine_displacement': [660], # Engine displacement in cc
}

# Convert the input data into a DataFrame
input_df = pd.DataFrame(car_details)

# Use the model pipeline to predict the price
# Note: The pipeline will automatically apply the preprocessing and prediction steps
predicted_price = model.predict(input_df)

# Output the predicted price
print(f"Predicted Price: Rs. {predicted_price[0]:,.2f}")

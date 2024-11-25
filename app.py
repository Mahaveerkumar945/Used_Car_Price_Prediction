from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('price_prediction_model.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    make = request.form['make']
    model_name = request.form['model']
    city = request.form['city']
    year = int(request.form['year'])
    mileage = int(request.form['mileage'])
    engine_displacement = int(request.form['engine_displacement'])
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'make': [make],
        'model': [model_name],
        'city': [city],
        'year': [year],
        'mileage': [mileage],
        'Engine_displacement': [engine_displacement],
    })
    
    # Predict the price
    predicted_price = model.predict(input_data)
    price = f"Rs. {predicted_price[0]:,.2f}"
    
    return render_template('result.html', price=price)

if __name__ == '__main__':
    app.run(debug=True)

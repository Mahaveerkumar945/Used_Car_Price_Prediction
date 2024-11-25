import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load the dataset
file_path = 'used_cars_Pakistan.csv'  # Adjust the path as needed
data = pd.read_csv(file_path)

# Drop unnecessary columns
data_cleaned = data.drop(columns=['Unnamed: 0', 'Battery'])

# Handle missing values in Engine_displacement by imputing with median
data_cleaned['Engine_displacement'] = data_cleaned['Engine_displacement'].fillna(
    data_cleaned['Engine_displacement'].median()
)

# Sample 10% of the data for faster computation (optional, can be skipped for full data)
data_sampled = data_cleaned.sample(frac=0.1, random_state=42)

# Define features and target
X = data_sampled.drop(columns=['Price_Rs'])
y = data_sampled['Price_Rs']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a column transformer for categorical variables
categorical_features = ['make', 'model', 'city']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessing and model pipeline
preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features)
], remainder='passthrough')

# Initialize the RandomForestRegressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"Model Evaluation:\nMean Absolute Error (MAE): {mae}\nRoot Mean Squared Error (RMSE): {rmse}")

# Save the model for future use
joblib.dump(model, 'price_prediction_model.pkl')
print("Model saved as 'price_prediction_model.pkl'")

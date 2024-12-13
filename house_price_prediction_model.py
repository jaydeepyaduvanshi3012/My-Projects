import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import joblib

# Load California housing dataset
housing = fetch_california_housing()

# Create a DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# Features (X) and Target (y)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the dataset (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train SVM model
svm_model = SVR(kernel='linear')
svm_model.fit(X_train_scaled, y_train)

# Save the model and scaler using joblib
joblib.dump(svm_model, 'svm_house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Optional: Print the model performance on test data
mse = np.mean((svm_model.predict(X_test_scaled) - y_test) ** 2)
print(f"Mean Squared Error: {mse}")

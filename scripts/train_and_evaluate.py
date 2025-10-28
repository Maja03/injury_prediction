import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Step 1: Load data
data_path = os.path.join('data', 'processed_injury_dataset.csv')
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"Data loaded successfully. Shape: {df.shape}")

# Step 2: Prepare features and target
X = df.drop(columns=['p_id2', 'dob', 'season_days_injured'])  # Drop irrelevant columns
y = df['season_days_injured']
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Build and train the model pipeline
pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    RandomForestRegressor(random_state=42)
)
pipeline.fit(X_train, y_train)

# Step 5: Save the trained model
model_path = os.path.join('models', 'trained_model.joblib')
joblib.dump(pipeline, model_path)
print(f'Model saved to {model_path}')

# Step 6: Make predictions and evaluate
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Performance on test set:')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')

# Optional: plot predictions
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Injury Days')
plt.ylabel('Predicted Injury Days')
plt.title('Actual vs. Predicted Injury Days')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.show()

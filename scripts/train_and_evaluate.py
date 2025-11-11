import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json
import numpy as np

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

# Step 7: Save calibration/metadata for conformal intervals and OOD checks
try:
    os.makedirs('models', exist_ok=True)
    # Use a held-out calibration set from the test split residuals (split-conformal style)
    abs_residuals = (y_test - y_pred).abs()
    # 90% interval by default (alpha=0.1); store multiple quantiles for flexibility
    quantiles = {
        'q50': float(abs_residuals.quantile(0.50)),
        'q80': float(abs_residuals.quantile(0.80)),
        'q90': float(abs_residuals.quantile(0.90)),
        'q95': float(abs_residuals.quantile(0.95))
    }

    # Record feature-wise min/max from training set for simple OOD flagging
    feature_ranges = {}
    for col in X_train.columns:
        try:
            feature_ranges[col] = {
                'min': float(pd.to_numeric(X_train[col], errors='coerce').min()),
                'max': float(pd.to_numeric(X_train[col], errors='coerce').max())
            }
        except Exception:
            # Non-numeric columns or issues: skip
            continue

    metadata = {
        'calibration': {
            'abs_residual_quantiles': quantiles,
            'note': 'Split-conformal-style absolute residual quantiles computed on test split'
        },
        'feature_ranges': feature_ranges,
        'metrics': {
            'mse': float(mse),
            'r2': float(r2)
        }
    }

    metadata_path = os.path.join('models', 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved calibration/metadata to {metadata_path}")
except Exception as e:
    print(f"Warning: failed to write calibration/metadata: {e}")

# Step 8: Rolling-origin backtest by start_year (if available)
try:
    if 'start_year' in df.columns:
        years = sorted(df['start_year'].dropna().unique().tolist())
        backtest_rows = []
        for i in range(1, len(years)):
            train_years = years[:i]
            test_year = years[i]
            df_train = df[df['start_year'].isin(train_years)]
            df_test = df[df['start_year'] == test_year]
            if df_train.empty or df_test.empty:
                continue
            X_tr = df_train.drop(columns=['p_id2', 'dob', 'season_days_injured'])
            y_tr = df_train['season_days_injured']
            X_te = df_test.drop(columns=['p_id2', 'dob', 'season_days_injured'])
            y_te = df_test['season_days_injured']

            model_bt = make_pipeline(
                SimpleImputer(strategy='median'),
                RandomForestRegressor(random_state=42)
            )
            model_bt.fit(X_tr, y_tr)
            y_hat = model_bt.predict(X_te)

            for pid, y_true, y_pred in zip(df_test['p_id2'].values, y_te.values, y_hat):
                backtest_rows.append({
                    'player_id': pid,
                    'year': int(test_year),
                    'y_true_days': float(y_true),
                    'y_pred_days': float(y_pred)
                })

        if backtest_rows:
            os.makedirs('models', exist_ok=True)
            bt_csv = os.path.join('models', 'backtest_predictions.csv')
            pd.DataFrame(backtest_rows).to_csv(bt_csv, index=False)
            print(f"Saved rolling backtest predictions to {bt_csv}")

            # Simple decision analysis across thresholds
            bt_df = pd.DataFrame(backtest_rows)
            thresholds = [30, 40, 50, 60, 70]
            decision_stats = []
            for t in thresholds:
                flagged = bt_df[bt_df['y_pred_days'] >= t]
                rate_flagged = len(flagged) / len(bt_df)
                mean_true_flagged = float(flagged['y_true_days'].mean()) if not flagged.empty else np.nan
                decision_stats.append({'threshold': t, 'rate_flagged': rate_flagged, 'mean_true_days_flagged': mean_true_flagged})
            pd.DataFrame(decision_stats).to_csv(os.path.join('models', 'decision_curve_data.csv'), index=False)
            print("Saved decision curve data to models/decision_curve_data.csv")
        else:
            print("Backtest: not enough yearly splits to evaluate")
    else:
        print("Backtest skipped: 'start_year' not found in dataset")
except Exception as e:
    print(f"Warning: rolling backtest failed: {e}")

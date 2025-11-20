import json
import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, f1_score, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = os.path.join('data', 'processed_injury_dataset.csv')
MODELS_DIR = 'models'
REG_MODEL_PATH = os.path.join(MODELS_DIR, 'injury_days_reg.joblib')
CLF_MODEL_PATH = os.path.join(MODELS_DIR, 'injury_flag_clf.joblib')
SHAP_BACKGROUND_PATH = os.path.join(MODELS_DIR, 'shap_background.joblib')
METADATA_PATH = os.path.join(MODELS_DIR, 'metadata.json')
BACKTEST_CSV = os.path.join(MODELS_DIR, 'backtest_predictions.csv')
DECISION_CURVE_CSV = os.path.join(MODELS_DIR, 'decision_curve_data.csv')

CLASSIFICATION_THRESHOLD_DAYS = 30
DECISION_THRESHOLDS_DAYS = [30, 45, 60]
DECISION_THRESHOLDS_PROB = [0.30, 0.50, 0.70]
RANDOM_STATE = 42


def build_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> ColumnTransformer:
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='median'))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ],
        remainder='drop'
    )
    return transformer


def train_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    return Pipeline(
        steps=[
            ('preprocess', preprocessor),
            ('model', model)
        ]
    )


def rolling_origin_evaluation(df: pd.DataFrame,
                              reg_pipeline: Pipeline,
                              clf_pipeline: Pipeline,
                              feature_cols: List[str]) -> pd.DataFrame:
    years = sorted(df['start_year'].dropna().unique().tolist())
    eval_rows: List[Dict] = []

    for i in range(1, len(years)):
        train_years = years[:i]
        test_year = years[i]
        df_train = df[df['start_year'].isin(train_years)].copy()
        df_test = df[df['start_year'] == test_year].copy()

        if df_train.empty or df_test.empty:
            continue

        X_train = df_train[feature_cols]
        y_train_days = df_train['season_days_injured']
        y_train_flag = (y_train_days >= CLASSIFICATION_THRESHOLD_DAYS).astype(int)

        X_test = df_test[feature_cols]
        y_test_days = df_test['season_days_injured']
        y_test_flag = (y_test_days >= CLASSIFICATION_THRESHOLD_DAYS).astype(int)

        reg_pipeline.fit(X_train, y_train_days)
        clf_pipeline.fit(X_train, y_train_flag)

        y_pred_days = reg_pipeline.predict(X_test)
        y_prob_injury = clf_pipeline.predict_proba(X_test)[:, 1]
        y_pred_flag = (y_prob_injury >= 0.5).astype(int)

        for idx in range(len(df_test)):
            eval_rows.append({
                'player_id': df_test.iloc[idx]['p_id2'],
                'year': int(test_year),
                'y_true_days': float(y_test_days.iloc[idx]),
                'y_pred_days': float(y_pred_days[idx]),
                'injury_true_flag': int(y_test_flag.iloc[idx]),
                'injury_prob': float(y_prob_injury[idx]),
                'injury_pred_flag': int(y_pred_flag[idx])
            })

    return pd.DataFrame(eval_rows)


def summarize_metrics(eval_df: pd.DataFrame) -> Dict:
    if eval_df.empty:
        raise ValueError("Rolling evaluation produced no rows. Need at least two seasons.")

    abs_residuals = (eval_df['y_true_days'] - eval_df['y_pred_days']).abs()
    regression_metrics = {
        'mse': float(mean_squared_error(eval_df['y_true_days'], eval_df['y_pred_days'])),
        'r2': float(r2_score(eval_df['y_true_days'], eval_df['y_pred_days'])),
        'mae': float(abs_residuals.mean())
    }
    classification_metrics = {
        'precision': float(precision_score(eval_df['injury_true_flag'],
                                           eval_df['injury_pred_flag'],
                                           zero_division=0)),
        'recall': float(recall_score(eval_df['injury_true_flag'],
                                     eval_df['injury_pred_flag'],
                                     zero_division=0)),
        'f1': float(f1_score(eval_df['injury_true_flag'],
                             eval_df['injury_pred_flag'],
                             zero_division=0)),
        'roc_auc': float(roc_auc_score(eval_df['injury_true_flag'],
                                       eval_df['injury_prob'])),
        'average_precision': float(average_precision_score(eval_df['injury_true_flag'],
                                                           eval_df['injury_prob']))
    }
    quantiles = {
        'q50': float(abs_residuals.quantile(0.50)),
        'q80': float(abs_residuals.quantile(0.80)),
        'q90': float(abs_residuals.quantile(0.90)),
        'q95': float(abs_residuals.quantile(0.95))
    }
    return {
        'regression_metrics': regression_metrics,
        'classification_metrics': classification_metrics,
        'abs_residual_quantiles': quantiles
    }


def save_decision_curves(eval_df: pd.DataFrame):
    rows = []
    total = len(eval_df)
    for threshold in DECISION_THRESHOLDS_DAYS:
        flagged = eval_df[eval_df['y_pred_days'] >= threshold]
        rate_flagged = len(flagged) / total
        avg_true = float(flagged['y_true_days'].mean()) if not flagged.empty else np.nan
        rows.append({
            'mode': 'regression_days',
            'threshold': threshold,
            'share_flagged': rate_flagged,
            'avg_true_days_flagged': avg_true
        })
    for threshold in DECISION_THRESHOLDS_PROB:
        flagged = eval_df[eval_df['injury_prob'] >= threshold]
        rate_flagged = len(flagged) / total
        avg_true = float(flagged['y_true_days'].mean()) if not flagged.empty else np.nan
        rows.append({
            'mode': 'classification_prob',
            'threshold': threshold,
            'share_flagged': rate_flagged,
            'avg_true_days_flagged': avg_true
        })
    pd.DataFrame(rows).to_csv(DECISION_CURVE_CSV, index=False)
    print(f"Saved decision curve data to {DECISION_CURVE_CSV}")


def main():
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    if 'start_year' not in df.columns:
        raise ValueError("Column 'start_year' is required for longitudinal training.")

    os.makedirs(MODELS_DIR, exist_ok=True)

    feature_cols = [
        col for col in df.columns
        if col not in ['p_id2', 'dob', 'season_days_injured']
    ]

    # Ensure boolean columns are cast to integers for downstream imputers
    bool_cols = df[feature_cols].select_dtypes(include=['bool']).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    preprocessor = build_preprocessor(df, feature_cols)

    reg_model = HistGradientBoostingRegressor(
        max_depth=None,
        learning_rate=0.05,
        max_iter=500,
        l2_regularization=0.1,
        random_state=RANDOM_STATE
    )
    clf_model = HistGradientBoostingClassifier(
        max_depth=None,
        learning_rate=0.05,
        max_iter=500,
        l2_regularization=0.1,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )

    reg_pipeline = train_pipeline(preprocessor, reg_model)
    clf_pipeline = train_pipeline(preprocessor, clf_model)

    print("Running rolling-origin evaluation (season-wise)...")
    eval_df = rolling_origin_evaluation(df, reg_pipeline, clf_pipeline, feature_cols)
    eval_df.to_csv(BACKTEST_CSV, index=False)
    print(f"Saved rolling predictions to {BACKTEST_CSV}")

    metrics_summary = summarize_metrics(eval_df)
    save_decision_curves(eval_df)

    X_full = df[feature_cols]
    y_days = df['season_days_injured']
    y_flag = (y_days >= CLASSIFICATION_THRESHOLD_DAYS).astype(int)

    print("Training final gradient-boosting models on full dataset...")
    reg_pipeline.fit(X_full, y_days)
    clf_pipeline.fit(X_full, y_flag)

    joblib.dump(reg_pipeline, REG_MODEL_PATH)
    joblib.dump(clf_pipeline, CLF_MODEL_PATH)
    print(f"Saved regressor to {REG_MODEL_PATH}")
    print(f"Saved classifier to {CLF_MODEL_PATH}")

    background = X_full.sample(n=min(256, len(X_full)), random_state=RANDOM_STATE)
    joblib.dump(background, SHAP_BACKGROUND_PATH)
    print(f"Saved SHAP background sample to {SHAP_BACKGROUND_PATH}")

    feature_ranges = {}
    for col in feature_cols:
        series = pd.to_numeric(X_full[col], errors='coerce')
        if series.notna().any():
            feature_ranges[col] = {
                'min': float(series.min()),
                'max': float(series.max())
            }

    metadata = {
        'feature_names': feature_cols,
        'classification_threshold_days': CLASSIFICATION_THRESHOLD_DAYS,
        'regression_metrics': metrics_summary['regression_metrics'],
        'classification_metrics': metrics_summary['classification_metrics'],
        'calibration': {
            'abs_residual_quantiles': metrics_summary['abs_residual_quantiles'],
            'note': 'Quantiles derived from season-wise rolling evaluation'
        },
        'feature_ranges': feature_ranges,
        'decision_thresholds_days': DECISION_THRESHOLDS_DAYS,
        'decision_thresholds_prob': DECISION_THRESHOLDS_PROB,
        'references': [
            'Martins et al. (2025) longitudinal gradient-boosted injury prediction',
            'Calderón-Díaz et al. (2024) explainable ML for muscle injuries',
            'Majumdar et al. (2024) class-imbalance aware multi-season models'
        ]
    }

    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == '__main__':
    main()

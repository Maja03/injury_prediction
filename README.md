# AI-Based Athlete Injury Prediction System Using Open Football Data

## 🏥 Overview

This thesis project develops an AI-based system capable of predicting injury risk and severity among professional football players, leveraging openly available football datasets. The system applies supervised machine learning algorithms to model complex relationships between player attributes and injury outcomes.

## 🎯 Key Features

### 🖥️ Interactive Web Application
- **Dynamic Player Search**: Search and select any player from the database
- **Real-time Risk Analysis**: Instant injury risk predictions with detailed explanations
- **Interactive Charts**: Feature importance charts and risk distribution visualizations
- **Player-Specific Reports**: Comprehensive individual player analysis pages
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### 🤖 AI Prediction Engine
- **Dual Gradient-Boosted Pipelines**: Regression estimates injury days while classification estimates the probability of missing ≥30 days
- **Season-Aware Validation**: Rolling-origin splits by `start_year` reproduce longitudinal studies (Martins et al., 2025)
- **Explainable AI**: SHAP explanations expose biomechanical factors per player (Calderón-Díaz et al., 2024)
- **Comprehensive Model Evaluation**: Regression + classification metrics, decision curves, and conformal-style intervals
- **Open Data Integration**: Built on publicly accessible football datasets with four-season coverage
- **Decision Support Hooks**: Configurable thresholds for medical and performance staff

## 📊 Model Performance

- **Algorithms**: HistGradientBoosting Regressor (injury days) + HistGradientBoosting Classifier (≥30-day injury flag)
- **Validation**: Rolling-origin (train on seasons 1..n-1, test on season n) to respect temporal drift
- **Stored Metrics**: `models/metadata.json` captures R², MSE, MAE, ROC-AUC, average precision, recall, and residual quantiles after training
- **Decision Analytics**: `models/decision_curve_data.csv` summarises bench/play trade-offs for multiple probability & day thresholds
- **Explainability**: Player-level SHAP attributions surface biomechanical/workload drivers directly in the UI

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Note: the pre-trained models in `models/*.joblib` were saved with **scikit-learn 1.7.2**.  
If you see `InconsistentVersionWarning` or errors like `SimpleImputer ... _fill_dtype`, reinstall deps so you’re on the pinned version in `requirements.txt`.

### 1. Data Preparation

Convert the Excel dataset to CSV format:

```bash
python convert_excel_to_csv.py
```

### 2. Model Training

Train the longitudinal gradient-boosted pipelines (regression + classification):

```bash
python -u scripts/train_and_evaluate.py
```

### 3. Run the Web Application

Launch the interactive web application:

```bash
python run_app.py
```

Then open your browser and go to: **http://localhost:5000**

#### Troubleshooting: `python app.py` / player pages not working

If the app opens but clicking a player fails (e.g. `/api/player-analysis/<id>` returns 400) and you see warnings like:

- `InconsistentVersionWarning: Trying to unpickle estimator ... from version 1.7.2 when using version 1.8.0`
- `SimpleImputer ... has no attribute _fill_dtype`

it means your environment’s **scikit-learn version doesn’t match** the version used to train the saved models in `models/*.joblib`.

Fix (recommended): install the pinned scikit-learn version from `requirements.txt`:

```bash
python -m pip uninstall -y scikit-learn
python -m pip install -r requirements.txt
python -c "import sklearn; print(sklearn.__version__)"
python app.py
```

Alternative: retrain the models in your current environment (then re-run the app):

```bash
python -u scripts/train_and_evaluate.py
python app.py
```

### 4. Alternative: Command Line Interface

For command-line usage:

```bash
python simple_app.py
```

## 📁 Project Structure

```
project/
├── app.py                            # Flask web application
├── run_app.py                        # Application startup script
├── simple_app.py                     # Command-line interface
├── convert_excel_to_csv.py           # Data conversion utility
├── requirements.txt                  # Python dependencies
├── templates/                        # Web application templates
│   ├── base.html                     # Base template
│   ├── index.html                    # Main dashboard
│   └── player_detail.html            # Player analysis page
├── scripts/
│   └── train_and_evaluate.py         # Model training script
├── data/
│   ├── processed_injury_dataset.xls  # Original dataset
│   └── processed_injury_dataset.csv  # Processed CSV data
├── models/
│   ├── injury_days_reg.joblib        # Gradient-boosting regression pipeline
│   ├── injury_flag_clf.joblib        # Gradient-boosting classification pipeline
│   ├── shap_background.joblib        # Background sample for SHAP
│   └── metadata.json                 # Calibration + metrics summary
└── README.md                         # This file
```

## 🔬 Methodology

### Data Sources
- **Player Statistics**: Performance metrics, playing time, games played
- **Historical Injury Records**: Previous injury patterns and severity
- **Physical Metrics**: Age, height, weight, BMI, FIFA ratings
- **Positional Data**: Player position, work rate, nationality

### Machine Learning Approach
- **Algorithms**: HistGradientBoosting regression for `season_days_injured` + HistGradientBoosting classification for ≥30-day injury indicator
- **Preprocessing**: ColumnTransformer with median-imputed numeric features and one-hot encoded categoricals
- **Feature Set**: 99 attributes spanning history, workload, biomechanics, and demographics (see `metadata.json`)
- **Validation**: Rolling-origin evaluation across seasons, aligned with Majumdar et al. (2024) and Martins et al. (2025)
- **Calibration**: Absolute residual quantiles provide conformal-style intervals; class weights mitigate imbalance

### Key Variables
- Previous injury history (`cumulative_days_injured`, `avg_days_injured_prev_seasons`)
- Training load and availability (`season_minutes_played`, `matches_played`)
- Physical/biomechanical markers (`age`, `height_cm`, `weight_kg`, `bmi`)
- Position & role descriptors (`position_numeric`, `work_rate_numeric`)
- Performance indicators (`fifa_rating`, `pace`, `physic`)

## 📈 Evaluation Metrics

After running the training script, consult `models/metadata.json` for:

- **Regression**: MSE, MAE, R², residual quantiles (q50/80/90/95)
- **Classification**: Precision, recall, F1, ROC-AUC, average precision
- **Decision Curves**: Share of flagged players vs. mean true injury days per threshold

## 🎨 Visualizations

The system generates comprehensive visualizations:

1. **Actual vs. Predicted Scatter Plot**: Model performance visualization
2. **Residuals Distribution**: Error pattern analysis
3. **Feature Importance Rankings**: Most influential factors
4. **Prediction Error Analysis**: Systematic bias detection

## 🔍 Explainability

Calderón-Díaz et al. (2024) emphasise transparent biomechanical signals, so the app surfaces:

- **Per-Player SHAP Contributions**: Top 10 drivers for every prediction
- **Team-Level Risk Mix**: Aggregates both predicted days and injury probabilities
- **Manual Overrides**: Document adjustments for exogenous events while logging reasons

## 🏆 Research Contributions

### Academic Impact
- Longitudinal gradient-boosted approach extends Martins et al. (2025) with multi-season open data
- Explainable biomechanical analysis mirrors Calderón-Díaz et al. (2024)
- Imbalance-aware evaluation follows Majumdar et al. (2022, 2024) guidance

### Practical Applications
- **Medical Staff**: Evidence-based injury risk assessment
- **Coaches**: Informed workload management decisions
- **Management**: Strategic player acquisition and retention
- **Players**: Personalized injury prevention strategies

## 📚 Research Alignment
- **Martins et al. (2025, Journal of Clinical Medicine)** – Four-season gradient-boosted study informs the rolling-origin split and tree-based architecture.
- **Calderón-Díaz et al. (2024, Sensors)** – Emphasises explainable ML for muscle injuries, driving SHAP-based reporting and biomechanical narratives.
- **Majumdar et al. (2024, Journal of Sports Analytics)** – Demonstrates multi-season class-imbalance strategies adopted in our injury-flag classifier.
- **Majumdar et al. (2022, Sports Medicine – Open)** – Identifies data leakage pitfalls and validation standards mirrored in this pipeline.
- **Pillitteri et al. (2023, FootballScience.net summary)** – Synthesises internal/external load indicators motivating the chosen feature hierarchy and thresholds.

## 🔮 Future Enhancements

- **Real-time Data Integration**: Live performance and health monitoring
- **Advanced Algorithms**: Deep learning and ensemble methods
- **Multi-sport Support**: Extending to other athletic disciplines
- **Mobile Application**: User-friendly interface for field deployment
- **Predictive Analytics**: Long-term career injury trajectory modeling

## 📚 Technical Details

### Model Architecture
```python
preprocess = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ]
)

regression_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('model', LGBMRegressor(n_estimators=500, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9,
                            random_state=42))
])

classification_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('model', LGBMClassifier(n_estimators=500, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9,
                             class_weight='balanced',
                             random_state=42))
])
```

### Data Processing
- **Missing Value Handling**: Median (numeric) / most frequent (categorical)
- **Categorical Encoding**: One-hot via ColumnTransformer
- **Targets**: `season_days_injured` (continuous) + binary flag (`>=30` days)

## 🎓 Thesis Context

This system exemplifies the integration of open football data with longitudinal, research-backed ML workflows. It demonstrates how publicly available multi-season records can be transformed into actionable insights for athlete health and workload management.

Performance varies by dataset composition; after each training run consult `models/metadata.json` for contemporary regression and classification metrics. Consistent with prior literature, moderate R² / ROC-AUC values reflect the inherent complexity of injury forecasting, which is why explainability, calibration, and decision curves accompany every prediction.

## 📞 Usage Examples

### Individual Player Prediction
```python
from injury_prediction_app import InjuryPredictionSystem

# Initialize system
system = InjuryPredictionSystem()

# Predict for specific player
player_data = {
    'age': 25,
    'height_cm': 180,
    'weight_kg': 75,
    'cumulative_days_injured': 30,
    # ... other features
}

prediction = system.predict_injury_risk(player_data)
print(f"Predicted injury days: {prediction['predicted_injury_days']}")
print(f"Risk level: {prediction['risk_level']}")
```

### System Evaluation
```python
# Generate comprehensive report
report = system.generate_system_report()

# Create visualizations
system.create_evaluation_visualizations()

# Get feature importance
importance = system.get_feature_importance(top_n=10)
```

## 🔄 Development Methodology

This project follows a **Scrum/Kanban hybrid methodology** with time-boxed sprints (1-2 weeks) and visual workflow management.

### Agile Practices

- **Time-boxed Sprints**: 1-2 week focused development cycles
- **Product Backlog**: Prioritized list of features and tasks (see `PRODUCT_BACKLOG.md`)
- **Kanban Board**: Visual workflow with columns: Backlog → In Progress → In Review → Testing → Done
- **Sprint Planning**: Regular planning sessions to select and commit to work items
- **Daily Standups**: Daily progress tracking and blocker identification
- **Sprint Reviews**: End-of-sprint demonstrations and feedback
- **Retrospectives**: Continuous improvement through reflection

### Sprint History

- **Sprint 1** (Week 1-2): Foundation - Core ML models, web framework, Docker (16 points ✅)
- **Sprint 2** (Week 3-4): Quality - CI/CD, testing, documentation (13 points 🔄)
- **Sprint 3** (Week 5-6): Enhancement - Error handling, performance, UX (Planned)

This methodology supports iterative development, continuous improvement, and provides structured documentation for the thesis paper.

## 📄 License

This project is developed as part of academic research. Please cite appropriately if used in research or commercial applications.

## 👥 Acknowledgments

- Open football databases for providing comprehensive player data
- Scikit-learn community for robust machine learning tools
- Sports medicine research for validation of feature importance insights

---

**Thesis**: AI-Based Athlete Injury Prediction System Using Open Football Data  
**Author**: Maja Czekała 
**Institution**: 
**Year**: 2025

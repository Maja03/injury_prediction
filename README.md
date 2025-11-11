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
- **Individual Injury Risk Prediction**: Predicts the number of days an athlete is expected to be injured during a competitive season
- **Risk Level Classification**: Categorizes players into Low/Medium/High injury risk levels
- **Feature Importance Analysis**: Identifies the most influential factors affecting injury risk
- **Comprehensive Model Evaluation**: Rigorous validation with multiple metrics and visualizations
- **Open Data Integration**: Built on publicly accessible football databases
- **Interpretable AI**: Random Forest model provides transparent decision-making insights

## 📊 Model Performance

- **Algorithm**: Random Forest Regression
- **R² Score**: 0.46 (explains 46% of variance in injury days)
- **Cross-Validation**: 5-fold CV with consistent performance
- **Features**: 99 player attributes including injury history, playing time, age, physical metrics, and positional data

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### 1. Data Preparation

Convert the Excel dataset to CSV format:

```bash
python convert_excel_to_csv.py
```

### 2. Model Training

Train the Random Forest model:

```bash
python -u scripts/train_and_evaluate.py
```

### 3. Run the Web Application

Launch the interactive web application:

```bash
python run_app.py
```

Then open your browser and go to: **http://localhost:5000**

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
│   └── trained_model.joblib          # Trained Random Forest model
└── README.md                         # This file
```

## 🔬 Methodology

### Data Sources
- **Player Statistics**: Performance metrics, playing time, games played
- **Historical Injury Records**: Previous injury patterns and severity
- **Physical Metrics**: Age, height, weight, BMI, FIFA ratings
- **Positional Data**: Player position, work rate, nationality

### Machine Learning Approach
- **Algorithm**: Random Forest Regression
- **Preprocessing**: Median imputation for missing values
- **Feature Engineering**: 99 engineered features from raw data
- **Validation**: 80/20 train-test split with cross-validation

### Key Variables
- Previous injury history (`cumulative_days_injured`, `avg_days_injured_prev_seasons`)
- Playing time metrics (`season_minutes_played`, `total_minutes_played`)
- Physical attributes (`age`, `height_cm`, `weight_kg`, `bmi`)
- Positional data (`position_numeric`, `work_rate_numeric`)
- Performance indicators (`fifa_rating`, `pace`, `physic`)

## 📈 Evaluation Metrics

The system provides comprehensive evaluation through:

- **Mean Squared Error (MSE)**: Overall prediction accuracy
- **Root Mean Squared Error (RMSE)**: Interpretable error measure
- **Mean Absolute Error (MAE)**: Average prediction error
- **R-squared (R²)**: Proportion of variance explained
- **Cross-Validation Scores**: Robustness assessment

## 🎨 Visualizations

The system generates comprehensive visualizations:

1. **Actual vs. Predicted Scatter Plot**: Model performance visualization
2. **Residuals Distribution**: Error pattern analysis
3. **Feature Importance Rankings**: Most influential factors
4. **Prediction Error Analysis**: Systematic bias detection

## 🔍 Feature Importance

Top influential features for injury prediction:

1. **Previous Injury History**: Strongest predictor of future injuries
2. **Playing Time**: Cumulative minutes and games played
3. **Age**: Physical wear and recovery capacity
4. **Physical Metrics**: BMI, height, weight relationships
5. **Position**: Different injury patterns by playing position

## 🏆 Research Contributions

### Academic Impact
- Novel application of AI to sports injury prediction using open data
- Comprehensive evaluation framework for injury prediction models
- Feature importance analysis aligning with sports medicine insights

### Practical Applications
- **Medical Staff**: Evidence-based injury risk assessment
- **Coaches**: Informed workload management decisions
- **Management**: Strategic player acquisition and retention
- **Players**: Personalized injury prevention strategies

## 🔮 Future Enhancements

- **Real-time Data Integration**: Live performance and health monitoring
- **Advanced Algorithms**: Deep learning and ensemble methods
- **Multi-sport Support**: Extending to other athletic disciplines
- **Mobile Application**: User-friendly interface for field deployment
- **Predictive Analytics**: Long-term career injury trajectory modeling

## 📚 Technical Details

### Model Architecture
```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10
    ))
])
```

### Data Processing
- **Missing Value Handling**: Median imputation
- **Feature Scaling**: Not required for Random Forest
- **Categorical Encoding**: One-hot encoding for categorical variables
- **Target Variable**: `season_days_injured` (continuous)

## 🎓 Thesis Context

This system exemplifies the integration of open football data with advanced AI methodologies, creating a novel decision support tool for sports professionals. It demonstrates how publicly available data can be transformed into actionable insights for athlete health and performance optimization.

The moderate predictive power (R² = 0.46) aligns with the complexity of injury prediction in professional sports, where multiple factors interact in non-linear ways. The system's interpretability through feature importance rankings provides valuable insights for sports medicine professionals.

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

## 📄 License

This project is developed as part of academic research. Please cite appropriately if used in research or commercial applications.

## 👥 Acknowledgments

- Open football databases for providing comprehensive player data
- Scikit-learn community for robust machine learning tools
- Sports medicine research for validation of feature importance insights

---

**Thesis**: AI-Based Athlete Injury Prediction System Using Open Football Data  
**Author**: [Your Name]  
**Institution**: [Your University]  
**Year**: 2025

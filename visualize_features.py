"""
Visual Feature Importance Analysis for AI-Based Athlete Injury Prediction System

This script creates visual representations of the model's feature importance
to enhance understanding and presentation of the injury prediction system.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

def create_feature_importance_visualization():
    """Create a comprehensive feature importance visualization"""
    
    # Load model and data
    model = joblib.load('models/trained_model.joblib')
    data = pd.read_csv('data/processed_injury_dataset.csv')
    
    # Get feature names
    feature_names = [col for col in data.columns 
                    if col not in ['p_id2', 'dob', 'season_days_injured']]
    
    # Get feature importance
    rf_model = model.named_steps['randomforestregressor']
    importances = rf_model.feature_importances_
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top 15 features horizontal bar chart
    top_features = importance_df.head(15)
    y_pos = np.arange(len(top_features))
    
    bars = ax1.barh(y_pos, top_features['importance'], color='skyblue', alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_features['feature'], fontsize=10)
    ax1.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax1.set_title('Top 15 Most Important Features\nfor Injury Prediction', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
        ax1.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left', va='center', fontsize=9)
    
    # Feature importance distribution
    ax2.hist(importance_df['importance'], bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Feature Importance Value', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Feature Importance Values', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mean_importance = importance_df['importance'].mean()
    ax2.axvline(mean_importance, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_importance:.3f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    print("Feature importance visualization saved as 'feature_importance_analysis.png'")
    plt.show()
    
    return importance_df

def create_prediction_visualization():
    """Create actual vs predicted visualization"""
    
    # Load model and data
    model = joblib.load('models/trained_model.joblib')
    data = pd.read_csv('data/processed_injury_dataset.csv')
    
    # Prepare data
    X = data.drop(columns=['p_id2', 'dob', 'season_days_injured'])
    y = data['season_days_injured']
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot: Actual vs Predicted
    ax1.scatter(y, y_pred, alpha=0.6, s=50, color='blue')
    
    # Perfect prediction line
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Injury Days', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Injury Days', fontsize=12, fontweight='bold')
    ax1.set_title('Actual vs. Predicted Injury Days', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(y, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=12, fontweight='bold')
    
    # Residuals plot
    residuals = y - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, s=50, color='green')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Predicted Injury Days', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax2.set_title('Residuals Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("Model performance visualization saved as 'model_performance_analysis.png'")
    plt.show()

def create_risk_distribution_visualization():
    """Create risk level distribution visualization"""
    
    # Load model and data
    model = joblib.load('models/trained_model.joblib')
    data = pd.read_csv('data/processed_injury_dataset.csv')
    
    # Prepare data and make predictions
    X = data.drop(columns=['p_id2', 'dob', 'season_days_injured'])
    y_pred = model.predict(X)
    
    # Categorize predictions into risk levels
    risk_levels = []
    for pred in y_pred:
        if pred < 30:
            risk_levels.append('Low')
        elif pred < 60:
            risk_levels.append('Medium')
        else:
            risk_levels.append('High')
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Risk level distribution
    risk_counts = pd.Series(risk_levels).value_counts()
    colors = ['lightgreen', 'orange', 'red']
    
    wedges, texts, autotexts = ax1.pie(risk_counts.values, labels=risk_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Distribution of Predicted Risk Levels\nAcross All Players', 
                  fontsize=14, fontweight='bold')
    
    # Predicted injury days distribution
    ax2.hist(y_pred, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.axvline(30, color='green', linestyle='--', linewidth=2, label='Low Risk Threshold')
    ax2.axvline(60, color='orange', linestyle='--', linewidth=2, label='Medium Risk Threshold')
    ax2.set_xlabel('Predicted Injury Days', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Players', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Predicted Injury Days', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('risk_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print("Risk distribution visualization saved as 'risk_distribution_analysis.png'")
    plt.show()

def main():
    """Main function to generate all visualizations"""
    print("Creating Visual Analysis for AI-Based Athlete Injury Prediction System")
    print("=" * 70)
    
    try:
        print("\nGenerating Feature Importance Analysis...")
        importance_df = create_feature_importance_visualization()
        
        print("\nGenerating Model Performance Analysis...")
        create_prediction_visualization()
        
        print("\nGenerating Risk Distribution Analysis...")
        create_risk_distribution_visualization()
        
        print("\nAll visualizations created successfully!")
        print("\nGenerated Files:")
        print("   • feature_importance_analysis.png")
        print("   • model_performance_analysis.png") 
        print("   • risk_distribution_analysis.png")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    main()

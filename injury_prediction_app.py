"""
AI-Based Athlete Injury Prediction System Using Open Football Data

This application provides a comprehensive injury prediction system for professional football players,
leveraging machine learning to predict injury risk and severity based on player statistics,
historical injury records, and performance metrics.

Author: [Your Name]
Thesis: AI-Based Athlete Injury Prediction System Using Open Football Data
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class InjuryPredictionSystem:
    """
    AI-Based Athlete Injury Prediction System
    
    This class encapsulates the complete injury prediction system including
    model loading, prediction, evaluation, and visualization capabilities.
    """
    
    def __init__(self, model_path='models/trained_model.joblib', data_path='data/processed_injury_dataset.csv'):
        """
        Initialize the injury prediction system
        
        Args:
            model_path (str): Path to the trained model
            data_path (str): Path to the dataset
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.data = None
        self.feature_names = None
        self.load_model()
        self.load_data()
    
    def load_model(self):
        """Load the trained Random Forest model"""
        try:
            self.model = joblib.load(self.model_path)
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model file not found. Please train the model first using scripts/train_and_evaluate.py")
            return False
        return True
    
    def load_data(self):
        """Load the processed injury dataset"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully! Shape: {self.data.shape}")
            
            # Get feature names (excluding target and irrelevant columns)
            self.feature_names = [col for col in self.data.columns 
                                if col not in ['p_id2', 'dob', 'season_days_injured']]
            print(f"Features identified: {len(self.feature_names)} features")
            
        except FileNotFoundError:
            print("Dataset file not found. Please run convert_excel_to_csv.py first")
            return False
        return True
    
    def predict_injury_risk(self, player_data):
        """
        Predict injury risk for a specific player
        
        Args:
            player_data (dict): Dictionary containing player features
            
        Returns:
            dict: Prediction results including risk level and confidence
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Convert player data to DataFrame
            player_df = pd.DataFrame([player_data])
            
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(player_df.columns)
            if missing_features:
                return {"error": f"Missing features: {missing_features}"}
            
            # Reorder columns to match training data
            player_df = player_df[self.feature_names]
            
            # Make prediction
            predicted_days = self.model.predict(player_df)[0]
            
            # Determine risk level
            if predicted_days < 30:
                risk_level = "Low"
            elif predicted_days < 60:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            return {
                "predicted_injury_days": round(predicted_days, 2),
                "risk_level": risk_level,
                "confidence": "Based on Random Forest model with R² = 0.46"
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def evaluate_model_performance(self):
        """
        Comprehensive model evaluation with multiple metrics
        
        Returns:
            dict: Evaluation metrics and performance analysis
        """
        if self.model is None or self.data is None:
            return {"error": "Model or data not loaded"}
        
        # Prepare data
        X = self.data.drop(columns=['p_id2', 'dob', 'season_days_injured'])
        y = self.data['season_days_injured']
        
        # Make predictions on full dataset
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        
        evaluation = {
            "mean_squared_error": round(mse, 2),
            "root_mean_squared_error": round(rmse, 2),
            "mean_absolute_error": round(mae, 2),
            "r_squared": round(r2, 4),
            "cross_validation_r2_mean": round(cv_scores.mean(), 4),
            "cross_validation_r2_std": round(cv_scores.std(), 4),
            "model_interpretability": "Random Forest provides feature importance rankings",
            "prediction_accuracy": f"Explains {r2*100:.1f}% of variance in injury days"
        }
        
        return evaluation
    
    def get_feature_importance(self, top_n=15):
        """
        Get feature importance rankings from the Random Forest model
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            dict: Feature importance analysis
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        # Get feature importance from the Random Forest
        rf_model = self.model.named_steps['randomforestregressor']
        importances = rf_model.feature_importances_
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(top_n)
        
        return {
            "top_features": top_features.to_dict('records'),
            "total_features": len(self.feature_names),
            "interpretation": "Higher importance values indicate stronger influence on injury prediction"
        }
    
    def create_evaluation_visualizations(self, save_plots=True):
        """
        Create comprehensive evaluation visualizations
        
        Args:
            save_plots (bool): Whether to save plots to files
            
        Returns:
            dict: Paths to saved visualization files
        """
        if self.model is None or self.data is None:
            return {"error": "Model or data not loaded"}
        
        # Prepare data
        X = self.data.drop(columns=['p_id2', 'dob', 'season_days_injured'])
        y = self.data['season_days_injured']
        y_pred = self.model.predict(X)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AI-Based Athlete Injury Prediction System - Model Evaluation', 
                     fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted Scatter Plot
        axes[0, 0].scatter(y, y_pred, alpha=0.6, s=50)
        axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Injury Days')
        axes[0, 0].set_ylabel('Predicted Injury Days')
        axes[0, 0].set_title('Actual vs. Predicted Injury Days')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² score to the plot
        r2 = r2_score(y, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Residuals Distribution
        residuals = y - y_pred
        axes[0, 1].hist(residuals, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Residual (Actual - Predicted)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature Importance (Top 10)
        rf_model = self.model.named_steps['randomforestregressor']
        importances = rf_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(10)
        
        axes[1, 0].barh(range(len(feature_importance)), feature_importance['importance'])
        axes[1, 0].set_yticks(range(len(feature_importance)))
        axes[1, 0].set_yticklabels(feature_importance['feature'], fontsize=8)
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_title('Top 10 Most Important Features')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction Error Analysis
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6, s=50)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Predicted Injury Days')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Prediction Error Analysis')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = 'model_evaluation_visualizations.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Visualizations saved to: {plot_path}")
        
        plt.show()
        
        return {"visualization_path": plot_path if save_plots else None}
    
    def generate_system_report(self):
        """
        Generate a comprehensive system report
        
        Returns:
            dict: Complete system analysis and performance report
        """
        print("=" * 60)
        print("AI-BASED ATHLETE INJURY PREDICTION SYSTEM")
        print("=" * 60)
        print()
        
        # Model Performance
        print("MODEL PERFORMANCE EVALUATION")
        print("-" * 40)
        evaluation = self.evaluate_model_performance()
        for metric, value in evaluation.items():
            if metric != "error":
                print(f"{metric.replace('_', ' ').title()}: {value}")
        print()
        
        # Feature Importance
        print("FEATURE IMPORTANCE ANALYSIS")
        print("-" * 40)
        importance = self.get_feature_importance()
        if "error" not in importance:
            print("Top 10 Most Influential Features:")
            for i, feature in enumerate(importance["top_features"][:10], 1):
                print(f"{i:2d}. {feature['feature']}: {feature['importance']:.4f}")
        print()
        
        # Dataset Information
        print("DATASET INFORMATION")
        print("-" * 40)
        print(f"Total Players: {len(self.data)}")
        print(f"Total Features: {len(self.feature_names)}")
        print(f"Target Variable: season_days_injured")
        print(f"Average Injury Days: {self.data['season_days_injured'].mean():.1f}")
        print(f"Max Injury Days: {self.data['season_days_injured'].max()}")
        print()
        
        # System Capabilities
        print("YSTEM CAPABILITIES")
        print("-" * 40)
        print("Individual player injury risk prediction")
        print("Risk level classification (Low/Medium/High)")
        print("Feature importance analysis")
        print("Comprehensive model evaluation")
        print("Visualization and reporting")
        print("Open data integration")
        print()
        
        print("=" * 60)
        print("SYSTEM READY FOR DEPLOYMENT")
        print("=" * 60)
        
        return {
            "evaluation": evaluation,
            "feature_importance": importance,
            "dataset_info": {
                "total_players": len(self.data),
                "total_features": len(self.feature_names),
                "avg_injury_days": self.data['season_days_injured'].mean()
            }
        }

def main():
    """
    Main function to demonstrate the injury prediction system
    """
    print("Initializing AI-Based Athlete Injury Prediction System...")
    print()
    
    # Initialize the system
    system = InjuryPredictionSystem()
    
    if system.model is None or system.data is None:
        print("System initialization failed. Please check model and data files.")
        return
    
    # Generate comprehensive report
    report = system.generate_system_report()
    
    # Create visualizations
    print("\nGenerating evaluation visualizations...")
    system.create_evaluation_visualizations()
    
    # Example prediction (using first player in dataset)
    print("\nEXAMPLE PREDICTION")
    print("-" * 40)
    example_player = system.data.iloc[0].to_dict()
    prediction = system.predict_injury_risk(example_player)
    
    if "error" not in prediction:
        print(f"Player: {example_player.get('p_id2', 'Unknown')}")
        print(f"Predicted Injury Days: {prediction['predicted_injury_days']}")
        print(f"Risk Level: {prediction['risk_level']}")
        print(f"Actual Injury Days: {example_player.get('season_days_injured', 'N/A')}")
    
    print("\nSystem demonstration completed!")

if __name__ == "__main__":
    main()

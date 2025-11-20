"""
Simplified AI-Based Athlete Injury Prediction System

This is a simplified version of the injury prediction system that focuses
on core functionality without complex visualizations.
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class SimpleInjuryPredictionSystem:
    """Simplified injury prediction system"""
    
    def __init__(self, model_path='models/injury_days_reg.joblib', data_path='data/processed_injury_dataset.csv'):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.data = None
        self.feature_names = None
        self.metadata = self._load_metadata()
        self.load_model()
        self.load_data()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = joblib.load(self.model_path)
            print("Model loaded successfully!")
            return True
        except FileNotFoundError:
            print("Model file not found. Please train the model first.")
            return False
    
    def load_data(self):
        """Load the dataset"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully! Shape: {self.data.shape}")
            
            # Get feature names
            self.feature_names = [col for col in self.data.columns 
                                if col not in ['p_id2', 'dob', 'season_days_injured']]
            print(f"Features identified: {len(self.feature_names)} features")
            return True
        except FileNotFoundError:
            print("Dataset file not found. Please run convert_excel_to_csv.py first")
            return False

    def _load_metadata(self):
        try:
            with open('models/metadata.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def predict_injury_risk(self, player_data):
        """Predict injury risk for a player"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Convert to DataFrame
            player_df = pd.DataFrame([player_data])
            
            # Check for missing features
            missing_features = set(self.feature_names) - set(player_df.columns)
            if missing_features:
                return {"error": f"Missing features: {list(missing_features)[:5]}..."}
            
            # Reorder columns
            player_df = player_df[self.feature_names]
            
            # Make prediction
            predicted_days = self.model.predict(player_df)[0]
            
            thresholds = self.metadata.get('decision_thresholds_days', [30, 60])
            low_high = thresholds[0] if thresholds else 30
            med_high = thresholds[1] if len(thresholds) > 1 else 60

            # Determine risk level with detailed interpretation
            if predicted_days < low_high:
                risk_level = "Low"
                risk_interpretation = "Low risk of injury this season"
                recommendation = "Normal training load and monitoring"
            elif predicted_days < med_high:
                risk_level = "Medium"
                risk_interpretation = "Moderate risk, monitoring recommended"
                recommendation = "Consider workload management and enhanced recovery"
            else:
                risk_level = "High"
                risk_interpretation = "High risk, proactive measures necessary"
                recommendation = "Reduce training intensity and implement injury prevention protocols"
            
            reg_metrics = self.metadata.get('regression_metrics') or {}
            r2_metric = reg_metrics.get('r2')
            confidence_note = f"Based on gradient-boosting regressor (rolling R² = {r2_metric:.2f})" if r2_metric is not None else "Based on gradient-boosting regressor"

            return {
                "predicted_injury_days": round(predicted_days, 2),
                "risk_level": risk_level,
                "risk_interpretation": risk_interpretation,
                "recommendation": recommendation,
                "confidence": confidence_note,
                "thresholds": {
                    "low": f"< {int(low_high)} days",
                    "medium": f"{int(low_high)}-{int(med_high)} days", 
                    "high": f"> {int(med_high)} days"
                }
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def evaluate_model(self):
        """Evaluate model performance"""
        if self.model is None or self.data is None:
            return {"error": "Model or data not loaded"}
        
        # Prepare data
        X = self.data.drop(columns=['p_id2', 'dob', 'season_days_injured'])
        y = self.data['season_days_injured']
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        return {
            "mse": round(mse, 2),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "r2": round(r2, 4),
            "explained_variance": f"{r2*100:.1f}%"
        }
    
    def get_feature_importance(self, top_n=10):
        """Get top feature importance"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        booster = self.model.named_steps.get('model')
        preprocessor = self.model.named_steps.get('preprocess')
        if booster is None:
            return {"error": "Model step missing"}
        importances = booster.feature_importances_
        try:
            feature_labels = preprocessor.get_feature_names_out()
        except Exception:
            feature_labels = [f'feature_{idx}' for idx in range(len(importances))]
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_labels,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n).to_dict('records')
    
    def generate_report(self):
        """Generate comprehensive system report"""
        print("=" * 70)
        print("AI-BASED ATHLETE INJURY PREDICTION SYSTEM")
        print("=" * 70)
        print("Using Open Football Data & Machine Learning")
        print()
        
        # Model Performance
        print("MODEL PERFORMANCE EVALUATION")
        print("-" * 40)
        evaluation = self.evaluate_model()
        if "error" not in evaluation:
            print(f"R-squared Score: {evaluation['r2']}")
            print(f"Explained Variance: {evaluation['explained_variance']}")
            print(f"Mean Absolute Error: {evaluation['mae']} days")
            print(f"Root Mean Square Error: {evaluation['rmse']} days")
            print()
            print("Performance Interpretation:")
            print(f"   • The model explains {evaluation['explained_variance']} of injury variance")
            print("   • Moderate predictive power suitable for injury risk assessment")
            print("   • Aligns with complexity of sports injury prediction")
        print()
        
        # Feature Importance
        print("TOP 10 MOST INFLUENTIAL FACTORS")
        print("-" * 40)
        importance = self.get_feature_importance()
        if "error" not in importance:
            for i, feature in enumerate(importance, 1):
                print(f"{i:2d}. {feature['feature']}: {feature['importance']:.4f}")
            print()
            print("Feature Interpretation:")
            print("   • Higher values indicate stronger influence on injury prediction")
            print("   • Previous injury history typically ranks highest")
            print("   • Physical and performance metrics provide additional insights")
        print()
        
        # Risk Level Interpretation
        print("RISK LEVEL INTERPRETATION")
        print("-" * 40)
        print("Low Risk (< 30 days predicted):")
        print("   • Low risk of significant injury this season")
        print("   • Normal training load and monitoring recommended")
        print()
        print("Medium Risk (30-60 days predicted):")
        print("   • Moderate risk, enhanced monitoring recommended")
        print("   • Consider workload management and recovery optimization")
        print()
        print("High Risk (> 60 days predicted):")
        print("   • High risk, proactive injury prevention measures necessary")
        print("   • Reduce training intensity and implement prevention protocols")
        print()
        
        # Dataset Info
        print("DATASET INFORMATION")
        print("-" * 40)
        print(f"Total Professional Players: {len(self.data)}")
        print(f"Total Predictive Features: {len(self.feature_names)}")
        print(f"Average Injury Days per Season: {self.data['season_days_injured'].mean():.1f}")
        print(f"Maximum Injury Days Recorded: {self.data['season_days_injured'].max()}")
        print(f"Data Sources: Open football databases and public records")
        print()
        
        # Next Steps
        print("NEXT STEPS & APPLICATION INTEGRATION")
        print("-" * 40)
        print("The system is designed for integration into interactive applications:")
        print()
        print("Interactive Web/Mobile App:")
        print("   • User-friendly interface for coaches and medical staff")
        print("   • Real-time player data input and prediction")
        print("   • Visual dashboards with risk monitoring")
        print()
        print("Sports Medicine Integration:")
        print("   • Electronic health record (EHR) system integration")
        print("   • Automated risk alerts and recommendations")
        print("   • Workload management decision support")
        print()
        print("Advanced Analytics:")
        print("   • Team-wide injury risk assessment")
        print("   • Seasonal trend analysis and planning")
        print("   • Player acquisition and retention insights")
        print()
        print("Research & Development:")
        print("   • Continuous model improvement with new data")
        print("   • Multi-sport expansion and validation")
        print("   • Integration with wearable device data")
        print()
        
        print("SYSTEM READY FOR DEPLOYMENT")
        print("=" * 70)

def main():
    """Main function"""
    print("Initializing AI-Based Athlete Injury Prediction System...")
    print()
    
    # Initialize system
    system = SimpleInjuryPredictionSystem()
    
    if system.model is None or system.data is None:
        print("System initialization failed.")
        return
    
    # Generate report
    system.generate_report()
    
    # Example prediction
    print("\n🔮 EXAMPLE PREDICTION DEMONSTRATION")
    print("-" * 50)
    example_player = system.data.iloc[0].to_dict()
    prediction = system.predict_injury_risk(example_player)
    
    if "error" not in prediction:
        print(f"Player: {example_player.get('p_id2', 'Unknown')}")
        print(f"Age: {example_player.get('age', 'N/A')} years")
        print(f"Position: {example_player.get('position_numeric', 'N/A')}")
        print()
        print("PREDICTION RESULTS:")
        print(f"   Predicted Injury Days: {prediction['predicted_injury_days']} days")
        print(f"   Risk Level: {prediction['risk_level']}")
        print(f"   Interpretation: {prediction['risk_interpretation']}")
        print(f"   Recommendation: {prediction['recommendation']}")
        print(f"   Model Confidence: {prediction['confidence']}")
        print()
        print("RISK THRESHOLDS:")
        thresholds = prediction['thresholds']
        print(f"   Low Risk: {thresholds['low']}")
        print(f"   Medium Risk: {thresholds['medium']}")
        print(f"   High Risk: {thresholds['high']}")
        print()
        print(f"ACTUAL INJURY DAYS: {example_player.get('season_days_injured', 'N/A')} days")
        print("   (For model validation and comparison)")
    else:
        print(f"Prediction failed: {prediction['error']}")
    
    print("\nSystem demonstration completed!")

if __name__ == "__main__":
    main()

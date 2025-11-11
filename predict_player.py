"""
Individual Player Injury Risk Prediction Interface

This script provides a simple interface for predicting injury risk
for individual players using the trained AI model.
"""

import pandas as pd
import numpy as np
from injury_prediction_app import InjuryPredictionSystem

def get_player_input():
    """
    Interactive function to get player data from user
    
    Returns:
        dict: Player data dictionary
    """
    print("AI-Based Athlete Injury Risk Prediction")
    print("=" * 50)
    print("Please provide the following player information:")
    print()
    
    player_data = {}
    
    # Basic information
    player_data['p_id2'] = input("Player ID: ").strip()
    player_data['start_year'] = int(input("Season start year: "))
    player_data['age'] = float(input("Age: "))
    player_data['height_cm'] = float(input("Height (cm): "))
    player_data['weight_kg'] = float(input("Weight (kg): "))
    
    # Calculate BMI
    player_data['bmi'] = player_data['weight_kg'] / ((player_data['height_cm'] / 100) ** 2)
    
    # Physical attributes
    player_data['pace'] = float(input("Pace rating (1-100): "))
    player_data['physic'] = float(input("Physical rating (1-100): "))
    player_data['fifa_rating'] = float(input("FIFA rating (1-100): "))
    
    # Playing time (current season)
    player_data['season_minutes_played'] = float(input("Season minutes played: "))
    player_data['season_games_played'] = float(input("Season games played: "))
    player_data['season_matches_in_squad'] = float(input("Season matches in squad: "))
    
    # Historical data
    player_data['total_minutes_played'] = float(input("Total career minutes played: "))
    player_data['total_games_played'] = float(input("Total career games played: "))
    player_data['cumulative_minutes_played'] = float(input("Cumulative minutes played: "))
    player_data['cumulative_games_played'] = float(input("Cumulative games played: "))
    
    # Injury history
    player_data['total_days_injured'] = float(input("Total career days injured: "))
    player_data['cumulative_days_injured'] = float(input("Cumulative days injured: "))
    player_data['avg_days_injured_prev_seasons'] = float(input("Average days injured (previous seasons): "))
    player_data['season_days_injured_prev_season'] = float(input("Days injured in previous season: "))
    player_data['significant_injury_prev_season'] = int(input("Significant injury in previous season (0/1): "))
    
    # Performance metrics
    player_data['minutes_per_game_prev_seasons'] = float(input("Minutes per game (previous seasons): "))
    player_data['avg_games_per_season_prev_seasons'] = float(input("Average games per season (previous): "))
    player_data['minutes_per_game'] = player_data['season_minutes_played'] / max(player_data['season_games_played'], 1)
    
    # Position and work rate (simplified)
    print("\nPosition (1=Goalkeeper, 2=Defender, 3=Midfielder, 4=Forward):")
    position = int(input("Position: "))
    player_data['position_numeric'] = position
    
    # Set position flags
    player_data['position_Goalkeeper'] = 1 if position == 1 else 0
    player_data['position_Defender'] = 1 if position == 2 else 0
    player_data['position_Midfielder'] = 1 if position == 3 else 0
    player_data['position_Forward'] = 1 if position == 4 else 0
    
    print("\nWork Rate (1=Low, 2=Medium, 3=High):")
    work_rate = int(input("Work Rate: "))
    player_data['work_rate_numeric'] = work_rate
    
    # Set work rate flags (simplified)
    player_data['work_rate_Low/Low'] = 1 if work_rate == 1 else 0
    player_data['work_rate_Medium/Medium'] = 1 if work_rate == 2 else 0
    player_data['work_rate_High/High'] = 1 if work_rate == 3 else 0
    
    # Set other work rate combinations to 0
    for rate in ['High/Low', 'High/Medium', 'Low/High', 'Low/Medium', 'Medium/High', 'Medium/Low']:
        player_data[f'work_rate_{rate}'] = 0
    
    # Nationality (simplified - using most common)
    print("\nNationality (simplified - using England as default):")
    player_data['nationality_England'] = 1
    
    # Set other nationalities to 0
    nationalities = ['Algeria', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Belgium', 'Bermuda',
                    'Bosnia Herzegovina', 'Cameroon', 'Canada', 'Chile', 'Colombia', 'Costa Rica',
                    'Croatia', 'Curacao', 'Czech Republic', 'DR Congo', 'Denmark', 'Ecuador', 'Egypt',
                    'Estonia', 'Finland', 'France', 'Gabon', 'Germany', 'Ghana', 'Greece', 'Guinea',
                    'Iceland', 'Iran', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Kenya',
                    'Mali', 'Morocco', 'Netherlands', 'New Zealand', 'Nigeria', 'Northern Ireland',
                    'Norway', 'Poland', 'Portugal', 'Republic of Ireland', 'Romania', 'Scotland',
                    'Senegal', 'Serbia', 'Slovenia', 'South Africa', 'Spain', 'Sweden', 'Switzerland',
                    'Turkey', 'Ukraine', 'United States', 'Uruguay', 'Wales']
    
    for nationality in nationalities:
        player_data[f'nationality_{nationality}'] = 0
    
    # Injury risk (this will be predicted)
    player_data['injury_risk'] = 0  # Placeholder
    
    # Date of birth (placeholder)
    player_data['dob'] = '1990-01-01'
    
    return player_data

def main():
    """
    Main function for individual player prediction
    """
    try:
        # Initialize the injury prediction system
        print("Loading AI-Based Athlete Injury Prediction System...")
        system = InjuryPredictionSystem()
        
        if system.model is None or system.data is None:
            print("System initialization failed. Please ensure model and data files exist.")
            return
        
        print("System loaded successfully!")
        print()
        
        # Get player data
        player_data = get_player_input()
        
        print("\nProcessing prediction...")
        
        # Make prediction
        prediction = system.predict_injury_risk(player_data)
        
        if "error" in prediction:
            print(f"Prediction failed: {prediction['error']}")
            return
        
        # Display results
        print("\n" + "=" * 60)
        print("INJURY RISK PREDICTION RESULTS")
        print("=" * 60)
        print(f"Player ID: {player_data['p_id2']}")
        print(f"Age: {player_data['age']} years")
        print(f"Position: {['Goalkeeper', 'Defender', 'Midfielder', 'Forward'][player_data['position_numeric']-1]}")
        print(f"BMI: {player_data['bmi']:.1f}")
        print()
        print("PREDICTION RESULTS:")
        print(f"   Predicted Injury Days: {prediction['predicted_injury_days']} days")
        print(f"   Risk Level: {prediction['risk_level']}")
        print(f"   Confidence: {prediction['confidence']}")
        print()
        
        # Enhanced risk interpretation
        print("DETAILED RISK ANALYSIS:")
        print(f"   Risk Level: {prediction['risk_level']}")
        print(f"   Interpretation: {prediction['risk_interpretation']}")
        print(f"   Recommendation: {prediction['recommendation']}")
        print()
        
        # Risk thresholds explanation
        print("RISK LEVEL THRESHOLDS:")
        thresholds = prediction['thresholds']
        print(f"   🟢 Low Risk ({thresholds['low']}): Low risk of significant injury this season")
        print(f"   🟡 Medium Risk ({thresholds['medium']}): Moderate risk, monitoring recommended")
        print(f"   🔴 High Risk ({thresholds['high']}): High risk, proactive measures necessary")
        print()
        
        # Show feature importance for context
        print("KEY FACTORS INFLUENCING PREDICTION:")
        importance = system.get_feature_importance(top_n=5)
        if "error" not in importance:
            for i, feature in enumerate(importance["top_features"][:5], 1):
                print(f"   {i}. {feature['feature']}: {feature['importance']:.3f}")
        
        print("\n" + "=" * 60)
        print("Prediction completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nPrediction cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please check your input data and try again.")

if __name__ == "__main__":
    main()

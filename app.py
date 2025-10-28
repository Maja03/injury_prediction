from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
import plotly.utils
import json

app = Flask(__name__)

class InjuryPredictionWebApp:
    """Web application class for injury prediction system"""
    
    def __init__(self):
        self.model = None
        self.data = None
        self.players = None
        self.feature_names = None
        self.load_system()
    
    def load_system(self):
        """Load model and data"""
        try:
            # Load model
            self.model = joblib.load('models/trained_model.joblib')
            
            # Load data
            self.data = pd.read_csv('data/processed_injury_dataset.csv')
            
            # Define features to match training (exclude identifiers and target only)
            self.feature_names = [
                col for col in self.data.columns
                if col not in ['p_id2', 'dob', 'season_days_injured']
            ]
            
            # Create players dataframe with one record per player — latest year record
            self.players = (
                self.data.sort_values('start_year', ascending=False)
                         .groupby('p_id2')
                         .first()
                         .reset_index()
            )
            
            print(f"System loaded: {len(self.data)} total records, {len(self.players)} unique players, {len(self.feature_names)} features")
        
        except Exception as e:
            print(f"Error loading system: {e}")
    
    def predict_player_injury(self, player_id):
        """Predict injury risk for a specific player based on latest record"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Get latest record for this player
            player_records = self.data[self.data['p_id2'] == player_id]
            if player_records.empty:
                return {"error": "Player not found"}
            player_data = player_records.sort_values('start_year', ascending=False).iloc[0]
            
            # Prepare features
            X = pd.DataFrame([player_data[self.feature_names]])
            
            # Make prediction
            predicted_days = float(self.model.predict(X)[0])
            
            # Determine risk level
            if predicted_days < 30:
                risk_level = "Low"
                risk_color = "success"
                risk_interpretation = "Low risk of significant injury this season"
                recommendation = "Normal training load and monitoring"
            elif predicted_days < 60:
                risk_level = "Medium"
                risk_color = "warning"
                risk_interpretation = "Moderate risk, monitoring recommended"
                recommendation = "Consider workload management and enhanced recovery"
            else:
                risk_level = "High"
                risk_color = "danger"
                risk_interpretation = "High risk, proactive measures necessary"
                recommendation = "Reduce training intensity and implement prevention protocols"
            
            # Safe extraction helpers
            def safe_float(value, default=None):
                try:
                    if pd.isna(value):
                        return default
                    return float(value)
                except Exception:
                    return default

            def safe_int(value, default=None):
                try:
                    if pd.isna(value):
                        return default
                    return int(value)
                except Exception:
                    return default

            position_numeric = safe_int(player_data.get('position_numeric'), None)
            position_name = self.get_position_name(position_numeric) if position_numeric is not None else "Unknown"
            # Prefer total_days_injured; fallback to season_days_injured if missing
            actual_days = safe_float(player_data.get('total_days_injured'), None)
            if actual_days is None:
                actual_days = safe_float(player_data.get('season_days_injured'), None)
            age_val = safe_float(player_data.get('age'), None)

            return {
                "player_id": player_id,
                "predicted_injury_days": round(predicted_days, 2),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "risk_interpretation": risk_interpretation,
                "recommendation": recommendation,
                "actual_injury_days": actual_days,
                "age": age_val,
                "position": position_name,
                "confidence": "Based on Random Forest model with R² = 0.46"
            }
        
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def get_position_name(self, position_num):
        """Convert position number to name"""
        positions = {0: "Goalkeeper", 1: "Defender", 2: "Midfielder", 3: "Forward"}
        return positions.get(position_num, "Unknown")
    
    def get_player_comparison_data(self, player_id):
        """Get data for player comparison charts"""
        if self.model is None:
            return None
        
        try:
            # Get latest record for this player
            player_records = self.data[self.data['p_id2'] == player_id]
            if player_records.empty:
                return None
            player_data = player_records.sort_values('start_year', ascending=False).iloc[0]
            
            # Feature importances from RandomForestRegressor step of pipeline
            rf_model = self.model.named_steps['randomforestregressor']
            importances = rf_model.feature_importances_
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(10)
            
            # Ensure JSON-serializable python floats
            player_values = []
            for feature in feature_importance['feature']:
                value = player_data.get(feature)
                if pd.isna(value):
                    player_values.append(None)
                else:
                    try:
                        player_values.append(float(value))
                    except Exception:
                        player_values.append(None)
            
            return {
                'feature_names': feature_importance['feature'].tolist(),
                'feature_importance': [float(v) for v in feature_importance['importance'].tolist()],
                'player_values': player_values
            }
        
        except Exception:
            return None
    
    def get_team_statistics(self):
        """Get overall team statistics using unique/latest player records"""
        if self.players is None or self.model is None:
            return None
        
        try:
            # Prepare features for predictions on latest records only
            X = self.players[self.feature_names]
            y_pred = self.model.predict(X)
            
            risk_levels = []
            for pred in y_pred:
                if pred < 30:
                    risk_levels.append('Low')
                elif pred < 60:
                    risk_levels.append('Medium')
                else:
                    risk_levels.append('High')
            
            risk_counts = pd.Series(risk_levels).value_counts()
            
            return {
                'total_players': len(self.players),
                'avg_predicted_days': round(np.mean(y_pred), 2),
                'risk_distribution': risk_counts.to_dict(),
                'high_risk_players': int(risk_counts.get('High', 0)),
                'medium_risk_players': int(risk_counts.get('Medium', 0)),
                'low_risk_players': int(risk_counts.get('Low', 0))
            }
        
        except Exception as e:
            print(f"Error calculating team statistics: {e}")
            return None

# Initialize the web app
web_app = InjuryPredictionWebApp()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/players')
def get_players():
    if web_app.players is None:
        return jsonify({"error": "No players available"})
    
    players_list = []
    for _, player in web_app.players.iterrows():
        players_list.append({
            'id': player['p_id2'],
            'name': player['p_id2'],
            'age': int(player['age']),
            'position': web_app.get_position_name(player['position_numeric']),
            'latest_year': int(player['start_year'])
        })
    
    return jsonify(players_list)

@app.route('/api/predict/<player_id>')
def predict_player(player_id):
    result = web_app.predict_player_injury(player_id)
    return jsonify(result)

@app.route('/api/player-analysis/<player_id>')
def get_player_analysis(player_id):
    try:
        prediction = web_app.predict_player_injury(player_id)
        # Check if prediction contains an error
        if "error" in prediction:
            return jsonify({"error": prediction["error"]}), 400
        
        comparison_data = web_app.get_player_comparison_data(player_id)
        # Return prediction even if comparison data is unavailable
        return jsonify({
            'prediction': prediction,
            'comparison_data': comparison_data
        })
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/team-stats')
def get_team_stats():
    stats = web_app.get_team_statistics()
    if stats is None:
        return jsonify({"error": "No data available"})
    return jsonify(stats)

@app.route('/api/charts/feature-importance/<player_id>')
def get_feature_importance_chart(player_id):
    data = web_app.get_player_comparison_data(player_id)
    if data is None:
        return jsonify({"error": "No data available"})
    
    fig = go.Figure(data=[
        go.Bar(
            y=data['feature_names'],
            x=data['feature_importance'],
            orientation='h',
            marker_color='skyblue',
            text=[f"{imp:.3f}" for imp in data['feature_importance']],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f"Feature Importance for {player_id}",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=500,
        margin=dict(l=200)
    )
    
    fig_json = json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    return jsonify(fig_json)

@app.route('/api/charts/risk-distribution')
def get_risk_distribution_chart():
    stats = web_app.get_team_statistics()
    if stats is None:
        return jsonify({"error": "No data available"})
    
    fig = go.Figure(data=[
        go.Pie(
            labels=list(stats['risk_distribution'].keys()),
            values=list(stats['risk_distribution'].values()),
            marker_colors=['lightgreen', 'orange', 'red'],
            textinfo='label+percent',
            textposition='outside'
        )
    ])
    
    fig.update_layout(title="Team Risk Level Distribution", height=400)
    
    fig_json = json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    return jsonify(fig_json)

@app.route('/player/<player_id>')
def player_detail(player_id):
    return render_template('player_detail.html', player_id=player_id)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

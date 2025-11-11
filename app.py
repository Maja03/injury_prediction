from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
import plotly.utils
import json
import os

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

            # Load calibration/metadata
            self.metadata = None
            try:
                with open(os.path.join('models', 'metadata.json'), 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = None

            # Load settings (costs and thresholds)
            self.settings = self._load_settings()

            # Fallback runtime calibration quantiles (if metadata missing): compute from dataset
            self.runtime_calibration_quantiles = None
            try:
                X_all = self.data[self.feature_names]
                y_all = self.data['season_days_injured']
                y_hat_all = self.model.predict(X_all)
                abs_res = (y_all - y_hat_all).abs()
                self.runtime_calibration_quantiles = {
                    'q50': float(abs_res.quantile(0.50)),
                    'q80': float(abs_res.quantile(0.80)),
                    'q90': float(abs_res.quantile(0.90)),
                    'q95': float(abs_res.quantile(0.95))
                }
            except Exception:
                self.runtime_calibration_quantiles = None
        
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

            # Apply manual override if present
            override_info = self._get_override(player_id)
            override_applied = False
            if override_info and isinstance(override_info.get('override_days'), (int, float)):
                predicted_days = float(override_info['override_days'])
                override_applied = True
            
            # Determine risk level
            thresholds = self.settings.get('risk_thresholds', {"low_high": 30, "med_high": 60})
            low_high = float(thresholds.get('low_high', 30))
            med_high = float(thresholds.get('med_high', 60))
            if predicted_days < low_high:
                risk_level = "Low"
                risk_color = "success"
                risk_interpretation = "Low risk of significant injury this season"
                recommendation = "Normal training load and monitoring"
            elif predicted_days < med_high:
                risk_level = "Medium"
                risk_color = "warning"
                risk_interpretation = "Moderate risk, monitoring recommended"
                recommendation = "Consider workload management and enhanced recovery"
            else:
                risk_level = "High"
                risk_color = "danger"
                risk_interpretation = "High risk, proactive measures necessary"
                recommendation = "Reduce training intensity and implement prevention protocols"

            # Conformal-like prediction interval using absolute residual quantiles
            interval = self._compute_prediction_interval(predicted_days)

            # Simple OOD/uncertainty flags
            uncertainty_flag, uncertainty_reason = self._assess_uncertainty(X, interval)

            # Cost-sensitive decision suggestion (bench if predicted days exceed decision threshold)
            decision_threshold_days = float(self.settings.get('decision_threshold_days', med_high))
            decision_suggestion = "Bench" if predicted_days >= decision_threshold_days else "Play"
            
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
                "confidence": "Calibrated with residual quantiles; interval provided",
                "prediction_interval_low": round(interval[0], 2) if interval else None,
                "prediction_interval_high": round(interval[1], 2) if interval else None,
                "uncertainty_flag": uncertainty_flag,
                "uncertainty_reason": uncertainty_reason,
                "decision_threshold_days": decision_threshold_days,
                "decision_suggestion": decision_suggestion,
                "override_applied": override_applied,
                "override": override_info
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

    # ----- Settings, metadata, and overrides helpers -----
    def _load_settings(self):
        default = {
            'risk_thresholds': { 'low_high': 30, 'med_high': 60 },
            'decision_threshold_days': 60,
            'interval_quantile': 'q90'
        }
        try:
            os.makedirs('config', exist_ok=True)
            path = os.path.join('config', 'settings.json')
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # shallow merge
                default.update(data)
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(default, f, indent=2)
            return default
        except Exception:
            return default

    def _save_settings(self, settings):
        try:
            os.makedirs('config', exist_ok=True)
            with open(os.path.join('config', 'settings.json'), 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)
            self.settings = settings
            return True
        except Exception:
            return False

    def _compute_prediction_interval(self, pred_value: float):
        try:
            qkey = self.settings.get('interval_quantile', 'q90')
            q = 0.0
            if self.metadata:
                qmap = self.metadata.get('calibration', {}).get('abs_residual_quantiles', {})
                q = float(qmap.get(qkey, qmap.get('q90', 0.0)))
            # Fallback to runtime quantiles if metadata missing or zero
            if (not self.metadata) or (q == 0.0):
                if self.runtime_calibration_quantiles:
                    q = float(self.runtime_calibration_quantiles.get(qkey, self.runtime_calibration_quantiles.get('q90', 0.0)))
            # If still zero, return symmetric zero-width interval as last resort
            return max(0.0, pred_value - q), max(0.0, pred_value + q)
        except Exception:
            return None

    def _assess_uncertainty(self, X: pd.DataFrame, interval):
        # Simple heuristics: interval width, missing values, or features out of training range
        try:
            reasons = []
            if interval:
                width = interval[1] - interval[0]
                if width >= 30:  # wide interval in days
                    reasons.append(f"Wide interval ({width:.1f} days)")
            if X.isna().any().any():
                reasons.append("Missing values imputed")
            fr = (self.metadata or {}).get('feature_ranges') or {}
            if fr:
                row = X.iloc[0]
                for feat, bounds in fr.items():
                    if feat in row.index:
                        val = row[feat]
                        try:
                            v = float(val)
                            if v < bounds['min'] or v > bounds['max']:
                                reasons.append(f"Out-of-range: {feat}")
                        except Exception:
                            continue
            return (len(reasons) > 0, "; ".join(reasons) if reasons else None)
        except Exception:
            return (False, None)

    def _get_override(self, player_id: str):
        try:
            os.makedirs('config', exist_ok=True)
            path = os.path.join('config', 'overrides.json')
            if not os.path.exists(path):
                return None
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get(player_id)
        except Exception:
            return None

    def _set_override(self, player_id: str, payload: dict):
        try:
            os.makedirs('config', exist_ok=True)
            path = os.path.join('config', 'overrides.json')
            data = {}
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f) or {}
            data[player_id] = payload
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False

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

# ----- Export evaluation CSV -----
@app.route('/api/export/evaluation')
def export_evaluation():
    try:
        # Use latest unique player records
        if web_app.players is None:
            return jsonify({"error": "No data available"}), 400
        X = web_app.players[web_app.feature_names]
        preds = web_app.model.predict(X)
        rows = []
        for i, row in web_app.players.iterrows():
            player_id = row['p_id2']
            pred_days = float(preds[i])
            # intervals/uncertainty and suggestion via shared helpers
            Xrow = web_app.players.iloc[[i]][web_app.feature_names]
            interval = web_app._compute_prediction_interval(pred_days)
            uncertainty_flag, uncertainty_reason = web_app._assess_uncertainty(Xrow, interval)
            thresholds = web_app.settings.get('risk_thresholds', {"low_high": 30, "med_high": 60})
            low_high = float(thresholds.get('low_high', 30))
            med_high = float(thresholds.get('med_high', 60))
            risk_level = 'Low' if pred_days < low_high else ('Medium' if pred_days < med_high else 'High')
            decision_threshold_days = float(web_app.settings.get('decision_threshold_days', med_high))
            decision_suggestion = "Bench" if pred_days >= decision_threshold_days else "Play"
            override = web_app._get_override(player_id)
            rows.append({
                'player_id': player_id,
                'age': row.get('age'),
                'position': web_app.get_position_name(row.get('position_numeric')),
                'predicted_days': round(pred_days, 2),
                'pi_low': round(interval[0], 2) if interval else None,
                'pi_high': round(interval[1], 2) if interval else None,
                'uncertainty_flag': bool(uncertainty_flag),
                'uncertainty_reason': uncertainty_reason,
                'risk_level': risk_level,
                'decision_threshold_days': decision_threshold_days,
                'suggestion': decision_suggestion,
                'override_days': (override or {}).get('override_days'),
                'override_reason': (override or {}).get('reason')
            })
        df = pd.DataFrame(rows)
        csv_text = df.to_csv(index=False)
        return app.response_class(csv_text, mimetype='text/csv')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

# ----- Settings API -----
@app.route('/api/settings', methods=['GET', 'POST'])
def settings_api():
    if request.method == 'GET':
        return jsonify(web_app.settings)
    try:
        body = request.get_json(force=True, silent=True) or {}
        new_settings = web_app.settings.copy()
        # Shallow merge allowed keys
        if 'risk_thresholds' in body and isinstance(body['risk_thresholds'], dict):
            new_settings['risk_thresholds'].update(body['risk_thresholds'])
        if 'decision_threshold_days' in body:
            new_settings['decision_threshold_days'] = float(body['decision_threshold_days'])
        if 'interval_quantile' in body:
            new_settings['interval_quantile'] = str(body['interval_quantile'])
        ok = web_app._save_settings(new_settings)
        if not ok:
            return jsonify({'error': 'Failed to persist settings'}), 500
        return jsonify(new_settings)
    except Exception as e:
        return jsonify({'error': f'Invalid settings: {e}'}), 400

# ----- Manual override API -----
@app.route('/api/override/<player_id>', methods=['POST'])
def set_override(player_id):
    try:
        payload = request.get_json(force=True, silent=True) or {}
        # expected: { override_days: number, reason: str }
        override_days = payload.get('override_days')
        reason = payload.get('reason', 'manual')
        if override_days is None:
            return jsonify({'error': 'override_days required'}), 400
        ok = web_app._set_override(player_id, {
            'override_days': float(override_days),
            'reason': reason
        })
        if not ok:
            return jsonify({'error': 'Failed to save override'}), 500
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/player/<player_id>')
def player_detail(player_id):
    return render_template('player_detail.html', player_id=player_id)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

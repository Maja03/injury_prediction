from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
import plotly.utils
import json
import os
import shap
from sklearn.inspection import permutation_importance

app = Flask(__name__)

class InjuryPredictionWebApp:
    """Web application class for injury prediction system"""
    
    def __init__(self):
        self.reg_model = None
        self.clf_model = None
        self.data = None
        self.players = None
        self.feature_names = None
        self.metadata = None
        self.settings = None
        self.classification_threshold_days = 30
        self.shap_background = None
        self.shap_explainer = None
        self.global_importance = None
        self.load_system()
    
    def load_system(self):
        """Load models, metadata, and data as described in the literature-backed pipeline."""
        try:
            self.reg_model = joblib.load('models/injury_days_reg.joblib')
            self.clf_model = joblib.load('models/injury_flag_clf.joblib')
        except Exception as exc:
            raise RuntimeError("Failed to load trained gradient-boosting pipelines. "
                               "Run scripts/train_and_evaluate.py first.") from exc

        # Load data
        self.data = pd.read_csv('data/processed_injury_dataset.csv')

        # Load metadata (feature list, calibration, thresholds)
        self.metadata = {}
        try:
            with open(os.path.join('models', 'metadata.json'), 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        except Exception:
            self.metadata = {}

        self.feature_names = self.metadata.get('feature_names') or [
            col for col in self.data.columns
            if col not in ['p_id2', 'dob', 'season_days_injured']
        ]
        self.classification_threshold_days = self.metadata.get(
            'classification_threshold_days', 30
        )

        # Latest-record snapshot per player
        self.players = (
            self.data.sort_values('start_year', ascending=False)
                     .groupby('p_id2')
                     .first()
                     .reset_index()
        )

        print(
            f"System loaded: {len(self.data)} total records, "
            f"{len(self.players)} unique players, {len(self.feature_names)} features"
        )

        # Load SHAP background used to build explainer
        try:
            self.shap_background = joblib.load('models/shap_background.joblib')
        except Exception:
            self.shap_background = None

        self.settings = self._load_settings()
        self._init_shap_explainer()
        self._compute_global_importance()

        # Fallback runtime quantiles if metadata missing
        self.runtime_calibration_quantiles = None
        try:
            X_all = self.data[self.feature_names]
            y_all = self.data['season_days_injured']
            y_hat_all = self.reg_model.predict(X_all)
            abs_res = (y_all - y_hat_all).abs()
            self.runtime_calibration_quantiles = {
                'q50': float(abs_res.quantile(0.50)),
                'q80': float(abs_res.quantile(0.80)),
                'q90': float(abs_res.quantile(0.90)),
                'q95': float(abs_res.quantile(0.95))
            }
        except Exception:
            self.runtime_calibration_quantiles = None
    
    def predict_player_injury(self, player_id):
        """Predict injury risk for a specific player based on latest record"""
        if self.reg_model is None or self.clf_model is None:
            return {"error": "Models not loaded"}
        
        try:
            # Get latest record for this player
            player_records = self.data[self.data['p_id2'] == player_id]
            if player_records.empty:
                return {"error": "Player not found"}
            player_data = player_records.sort_values('start_year', ascending=False).iloc[0]
            
            # Prepare features
            X = pd.DataFrame([player_data[self.feature_names]])
            
            # Make prediction
            predicted_days = float(self.reg_model.predict(X)[0])
            prob_injury = float(self.clf_model.predict_proba(X)[0][1])

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

            # Cost-sensitive decision suggestion (bench if either signal exceeds thresholds)
            decision_threshold_days = float(self.settings.get('decision_threshold_days', med_high))
            decision_threshold_prob = float(self.settings.get('decision_threshold_prob', 0.5))
            decision_suggestion = "Bench" if (
                predicted_days >= decision_threshold_days or prob_injury >= decision_threshold_prob
            ) else "Play"
            
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

            shap_contributions = self._compute_shap_breakdown(X)

            return {
                "player_id": player_id,
                "predicted_injury_days": round(predicted_days, 2),
                "injury_probability": round(prob_injury, 3),
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
                "decision_threshold_prob": decision_threshold_prob,
                "decision_suggestion": decision_suggestion,
                "override_applied": override_applied,
                "override": override_info,
                "feature_contributions": shap_contributions
            }
        
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def get_position_name(self, position_num):
        """Convert position number to name"""
        positions = {0: "Goalkeeper", 1: "Defender", 2: "Midfielder", 3: "Forward"}
        return positions.get(position_num, "Unknown")
    
    def get_player_comparison_data(self, player_id):
        """Get data for player comparison charts"""
        if self.reg_model is None or self.clf_model is None:
            return None
        
        try:
            # Get latest record for this player
            player_records = self.data[self.data['p_id2'] == player_id]
            if player_records.empty:
                return None
            player_data = player_records.sort_values('start_year', ascending=False).iloc[0]
            
            X = pd.DataFrame([player_data[self.feature_names]])
            shap_contribs = self._compute_shap_breakdown(X)
            if shap_contribs:
                feature_names = [item['feature'] for item in shap_contribs]
                feature_importance = [abs(float(item['contribution'])) for item in shap_contribs]
                shap_contribution = [float(item['contribution']) for item in shap_contribs]
                player_values = []
                for feat in feature_names:
                    val = player_data.get(feat)
                    if pd.isna(val):
                        player_values.append(None)
                    else:
                        try:
                            player_values.append(float(val))
                        except Exception:
                            player_values.append(None)

                return {
                    'feature_names': feature_names,
                    'feature_importance': feature_importance,
                    'shap_contribution': shap_contribution,
                    'player_values': player_values
                }

            # Fallback to global permutation importance
            return self._fallback_global_importance(player_data)
        
        except Exception:
            return None
    
    def get_team_statistics(self):
        """Get overall team statistics using unique/latest player records"""
        if self.players is None or self.reg_model is None or self.clf_model is None:
            return None
        
        try:
            # Prepare features for predictions on latest records only
            X = self.players[self.feature_names]
            y_pred_days = self.reg_model.predict(X)
            y_prob = self.clf_model.predict_proba(X)[:, 1]
            thresholds = self.settings.get('risk_thresholds', {"low_high": 30, "med_high": 60})
            low_high = float(thresholds.get('low_high', 30))
            med_high = float(thresholds.get('med_high', 60))
            prob_threshold = float(self.settings.get('decision_threshold_prob', 0.5))
            
            risk_levels = []
            for days, prob in zip(y_pred_days, y_prob):
                if days >= med_high or prob >= prob_threshold:
                    risk_levels.append('High')
                elif days >= low_high or prob >= (prob_threshold * 0.6):
                    risk_levels.append('Medium')
                else:
                    risk_levels.append('Low')
            
            risk_counts = pd.Series(risk_levels).value_counts()
            
            return {
                'total_players': len(self.players),
                'avg_predicted_days': round(np.mean(y_pred_days), 2),
                'avg_injury_probability': round(np.mean(y_prob), 3),
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
            'decision_threshold_prob': 0.5,
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

    def _init_shap_explainer(self):
        if self.reg_model is None or self.shap_background is None:
            self.shap_explainer = None
            return
        try:
            background_df = pd.DataFrame(self.shap_background)[self.feature_names]
            masker = shap.maskers.Independent(background_df)

            def model_predict(data):
                df = pd.DataFrame(data, columns=self.feature_names)
                return self.reg_model.predict(df)

            self.shap_explainer = shap.Explainer(
                model_predict,
                masker=masker,
                feature_names=self.feature_names
            )
        except Exception as exc:
            self.shap_explainer = None
            print(f"Warning: SHAP explainer disabled ({exc})")

    def _compute_global_importance(self):
        if self.reg_model is None or self.data is None:
            self.global_importance = None
            return
        try:
            X = self.data[self.feature_names]
            y = self.data['season_days_injured']
            result = permutation_importance(
                self.reg_model,
                X,
                y,
                n_repeats=5,
                random_state=42,
                n_jobs=-1
            )
            importance = []
            for idx, feat in enumerate(self.feature_names):
                importance.append({
                    'feature': feat,
                    'importance': float(result.importances_mean[idx])
                })
            importance.sort(key=lambda item: abs(item['importance']), reverse=True)
            self.global_importance = importance[:10]
        except Exception as exc:
            self.global_importance = None
            print(f"Warning: failed to compute permutation importance ({exc})")

    def _fallback_global_importance(self, player_data):
        if not self.global_importance:
            return None
        feature_names = [item['feature'] for item in self.global_importance]
        feature_importance = [abs(float(item['importance'])) for item in self.global_importance]
        shap_contribution = [float(item['importance']) for item in self.global_importance]
        player_values = []
        for feat in feature_names:
            val = player_data.get(feat)
            if pd.isna(val):
                player_values.append(None)
            else:
                try:
                    player_values.append(float(val))
                except Exception:
                    player_values.append(None)
        return {
            'feature_names': feature_names,
            'feature_importance': feature_importance,
            'shap_contribution': shap_contribution,
            'player_values': player_values
        }

    def _compute_shap_breakdown(self, X: pd.DataFrame):
        if self.shap_explainer is None:
            return None
        try:
            explanation = self.shap_explainer(X)
            values = explanation.values[0]
            features = explanation.feature_names
            row = X.iloc[0]
            contributions = []
            for feat, contrib in zip(features, values):
                val = row.get(feat)
                try:
                    val = float(val)
                except Exception:
                    val = None
                contributions.append({
                    'feature': feat,
                    'value': val,
                    'contribution': float(contrib)
                })
            contributions.sort(key=lambda item: abs(item['contribution']), reverse=True)
            return contributions[:10]
        except Exception as exc:
            print(f"Warning: SHAP computation failed ({exc})")
            return None

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
        if web_app.players is None or web_app.reg_model is None or web_app.clf_model is None:
            return jsonify({"error": "No data available"}), 400
        X = web_app.players[web_app.feature_names]
        preds = web_app.reg_model.predict(X)
        probs = web_app.clf_model.predict_proba(X)[:, 1]
        rows = []
        for i, row in web_app.players.iterrows():
            player_id = row['p_id2']
            pred_days = float(preds[i])
            prob_injury = float(probs[i])
            # intervals/uncertainty and suggestion via shared helpers
            Xrow = web_app.players.iloc[[i]][web_app.feature_names]
            interval = web_app._compute_prediction_interval(pred_days)
            uncertainty_flag, uncertainty_reason = web_app._assess_uncertainty(Xrow, interval)
            thresholds = web_app.settings.get('risk_thresholds', {"low_high": 30, "med_high": 60})
            low_high = float(thresholds.get('low_high', 30))
            med_high = float(thresholds.get('med_high', 60))
            risk_level = 'Low'
            if pred_days >= med_high or prob_injury >= web_app.settings.get('decision_threshold_prob', 0.5):
                risk_level = 'High'
            elif pred_days >= low_high:
                risk_level = 'Medium'
            decision_threshold_days = float(web_app.settings.get('decision_threshold_days', med_high))
            decision_threshold_prob = float(web_app.settings.get('decision_threshold_prob', 0.5))
            decision_suggestion = "Bench" if (
                pred_days >= decision_threshold_days or prob_injury >= decision_threshold_prob
            ) else "Play"
            override = web_app._get_override(player_id)
            rows.append({
                'player_id': player_id,
                'age': row.get('age'),
                'position': web_app.get_position_name(row.get('position_numeric')),
                'predicted_days': round(pred_days, 2),
                'injury_probability': round(prob_injury, 3),
                'pi_low': round(interval[0], 2) if interval else None,
                'pi_high': round(interval[1], 2) if interval else None,
                'uncertainty_flag': bool(uncertainty_flag),
                'uncertainty_reason': uncertainty_reason,
                'risk_level': risk_level,
                'decision_threshold_days': decision_threshold_days,
                'decision_threshold_prob': decision_threshold_prob,
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

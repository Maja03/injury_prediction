"""
Test suite for the Injury Prediction API endpoints.

This module contains tests for the Flask web application API endpoints,
ensuring proper functionality of the injury prediction system.
"""

import pytest
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Add parent directory to path to import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, InjuryPredictionWebApp


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_web_app():
    """Create a mock web app instance for testing."""
    with patch('app.web_app') as mock_app:
        # Mock players data - ensure it's a real DataFrame, not a mock
        # Use native Python types to avoid JSON serialization issues
        mock_players_df = pd.DataFrame({
            'p_id2': ['player1', 'player2', 'player3'],
            'age': [25.0, 28.0, 30.0],
            'position_numeric': [1, 2, 3],
            'start_year': [2023, 2023, 2023]
        })
        mock_app.players = mock_players_df
        
        # Mock get_position_name method - assign function directly
        def get_position_name(position_num):
            positions = {0: "Goalkeeper", 1: "Defender", 2: "Midfielder", 3: "Forward"}
            return positions.get(int(position_num) if position_num is not None else None, "Unknown")
        
        # Assign the function directly to avoid MagicMock issues
        mock_app.get_position_name = get_position_name
        
        # Mock prediction result
        mock_app.predict_player_injury.return_value = {
            'player_id': 'player1',
            'predicted_injury_days': 15.5,
            'injury_probability': 0.35,
            'risk_level': 'Low',
            'risk_color': 'success',
            'risk_interpretation': 'Low risk of significant injury this season',
            'recommendation': 'Normal training load and monitoring',
            'actual_injury_days': 12.0,
            'age': 25,
            'position': 'Defender',
            'confidence': 'Calibrated with residual quantiles',
            'prediction_interval_low': 10.0,
            'prediction_interval_high': 20.0,
            'uncertainty_flag': False,
            'uncertainty_reason': None,
            'decision_suggestion': 'Play'
        }
        
        # Mock team statistics
        mock_app.get_team_statistics.return_value = {
            'total_players': 3,
            'avg_predicted_days': 20.5,
            'avg_injury_probability': 0.4,
            'risk_distribution': {'Low': 2, 'Medium': 1, 'High': 0},
            'high_risk_players': 0,
            'medium_risk_players': 1,
            'low_risk_players': 2
        }
        
        # Mock player comparison data
        mock_app.get_player_comparison_data.return_value = {
            'feature_names': ['age', 'position_numeric', 'total_minutes'],
            'feature_importance': [0.5, 0.3, 0.2],
            'shap_contribution': [0.1, -0.05, 0.02],
            'player_values': [25.0, 1.0, 2000.0]
        }
        
        # Mock settings
        mock_app.settings = {
            'risk_thresholds': {'low_high': 30, 'med_high': 60},
            'decision_threshold_days': 60,
            'decision_threshold_prob': 0.5,
            'interval_quantile': 'q90'
        }
        
        yield mock_app


class TestIndexRoute:
    """Test cases for the index route."""
    
    def test_index_route(self, client):
        """Test that the index route returns 200 status."""
        response = client.get('/')
        assert response.status_code == 200


class TestPlayersAPI:
    """Test cases for the players API endpoint."""
    
    def test_get_players_success(self, client, mock_web_app):
        """Test successful retrieval of players list."""
        response = client.get('/api/players')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert isinstance(data, list)
        assert len(data) == 3
        assert 'id' in data[0]
        assert 'name' in data[0]
        assert 'age' in data[0]
        assert 'position' in data[0]
    
    def test_get_players_no_data(self, client):
        """Test players endpoint when no players are available."""
        with patch('app.web_app.players', None):
            response = client.get('/api/players')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'error' in data
            assert data['error'] == "No players available"


class TestPredictAPI:
    """Test cases for the prediction API endpoint."""
    
    def test_predict_player_success(self, client, mock_web_app):
        """Test successful player injury prediction."""
        response = client.get('/api/predict/player1')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'player_id' in data
        assert 'predicted_injury_days' in data
        assert 'risk_level' in data
        assert data['player_id'] == 'player1'
        assert data['risk_level'] == 'Low'
        assert isinstance(data['predicted_injury_days'], (int, float))
    
    def test_predict_player_not_found(self, client, mock_web_app):
        """Test prediction when player is not found."""
        mock_web_app.predict_player_injury.return_value = {"error": "Player not found"}
        
        response = client.get('/api/predict/nonexistent')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == "Player not found"
    
    def test_predict_player_model_error(self, client, mock_web_app):
        """Test prediction when model fails to load."""
        mock_web_app.predict_player_injury.return_value = {"error": "Models not loaded"}
        
        response = client.get('/api/predict/player1')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'error' in data


class TestPlayerAnalysisAPI:
    """Test cases for the player analysis API endpoint."""
    
    def test_get_player_analysis_success(self, client, mock_web_app):
        """Test successful retrieval of player analysis."""
        response = client.get('/api/player-analysis/player1')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'prediction' in data
        assert 'comparison_data' in data
        assert data['prediction']['player_id'] == 'player1'
    
    def test_get_player_analysis_error(self, client, mock_web_app):
        """Test player analysis when prediction fails."""
        mock_web_app.predict_player_injury.return_value = {"error": "Player not found"}
        
        response = client.get('/api/player-analysis/nonexistent')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data


class TestTeamStatsAPI:
    """Test cases for the team statistics API endpoint."""
    
    def test_get_team_stats_success(self, client, mock_web_app):
        """Test successful retrieval of team statistics."""
        response = client.get('/api/team-stats')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'total_players' in data
        assert 'avg_predicted_days' in data
        assert 'risk_distribution' in data
        assert data['total_players'] == 3
    
    def test_get_team_stats_no_data(self, client):
        """Test team stats when no data is available."""
        with patch('app.web_app.get_team_statistics', return_value=None):
            response = client.get('/api/team-stats')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'error' in data


class TestSettingsAPI:
    """Test cases for the settings API endpoint."""
    
    def test_get_settings(self, client, mock_web_app):
        """Test retrieval of current settings."""
        response = client.get('/api/settings')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'risk_thresholds' in data
        assert 'decision_threshold_days' in data
    
    def test_update_settings_success(self, client, mock_web_app):
        """Test successful update of settings."""
        mock_web_app._save_settings = Mock(return_value=True)
        
        new_settings = {
            'risk_thresholds': {'low_high': 25, 'med_high': 55},
            'decision_threshold_days': 55
        }
        
        response = client.post(
            '/api/settings',
            data=json.dumps(new_settings),
            content_type='application/json'
        )
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['decision_threshold_days'] == 55
    
    def test_update_settings_invalid(self, client, mock_web_app):
        """Test update settings with invalid data."""
        invalid_settings = {'invalid_key': 'invalid_value'}
        
        response = client.post(
            '/api/settings',
            data=json.dumps(invalid_settings),
            content_type='application/json'
        )
        # Should still return 200 but may not update invalid keys
        assert response.status_code in [200, 400]


class TestOverrideAPI:
    """Test cases for the manual override API endpoint."""
    
    def test_set_override_success(self, client, mock_web_app):
        """Test successful setting of player override."""
        mock_web_app._set_override = Mock(return_value=True)
        
        override_data = {
            'override_days': 45.0,
            'reason': 'Test override'
        }
        
        response = client.post(
            '/api/override/player1',
            data=json.dumps(override_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'ok'
    
    def test_set_override_missing_days(self, client, mock_web_app):
        """Test override with missing override_days."""
        response = client.post(
            '/api/override/player1',
            data=json.dumps({'reason': 'Test'}),
            content_type='application/json'
        )
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data


class TestExportAPI:
    """Test cases for the export API endpoint."""
    
    def test_export_evaluation_success(self, client, mock_web_app):
        """Test successful export of evaluation CSV."""
        # Mock players and models
        mock_web_app.players = pd.DataFrame({
            'p_id2': ['player1', 'player2'],
            'age': [25, 28],
            'position_numeric': [1, 2],
            'start_year': [2023, 2023]
        })
        
        # Mock model predictions
        mock_reg_model = Mock()
        mock_reg_model.predict = Mock(return_value=np.array([15.5, 20.3]))
        mock_web_app.reg_model = mock_reg_model
        
        mock_clf_model = Mock()
        mock_clf_model.predict_proba = Mock(return_value=np.array([[0.65, 0.35], [0.6, 0.4]]))
        mock_web_app.clf_model = mock_clf_model
        
        mock_web_app.feature_names = ['age', 'position_numeric']
        mock_web_app._compute_prediction_interval = Mock(return_value=(10.0, 20.0))
        mock_web_app._assess_uncertainty = Mock(return_value=(False, None))
        mock_web_app._get_override = Mock(return_value=None)
        
        response = client.get('/api/export/evaluation')
        assert response.status_code == 200
        assert response.content_type == 'text/csv; charset=utf-8'
        
        # Check CSV content
        csv_content = response.data.decode('utf-8')
        assert 'player_id' in csv_content
        assert 'predicted_days' in csv_content


class TestChartAPI:
    """Test cases for the chart API endpoints."""
    
    def test_feature_importance_chart(self, client, mock_web_app):
        """Test feature importance chart endpoint."""
        response = client.get('/api/charts/feature-importance/player1')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'data' in data or 'layout' in data  # Plotly JSON structure
    
    def test_risk_distribution_chart(self, client, mock_web_app):
        """Test risk distribution chart endpoint."""
        response = client.get('/api/charts/risk-distribution')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'data' in data or 'layout' in data  # Plotly JSON structure


class TestInjuryPredictionSystem:
    """Test cases for the InjuryPredictionSystem class."""
    
    @patch('injury_prediction_app.joblib.load')
    @patch('injury_prediction_app.pd.read_csv')
    def test_system_initialization(self, mock_read_csv, mock_joblib_load):
        """Test initialization of InjuryPredictionSystem."""
        # Mock model and data loading
        mock_model = Mock()
        mock_joblib_load.return_value = mock_model
        
        mock_data = pd.DataFrame({
            'p_id2': ['p1', 'p2'],
            'dob': ['1990-01-01', '1991-01-01'],
            'season_days_injured': [10, 20],
            'age': [25, 26],
            'position_numeric': [1, 2]
        })
        mock_read_csv.return_value = mock_data
        
        # Mock metadata loading
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = '{}'
            
            from injury_prediction_app import InjuryPredictionSystem
            system = InjuryPredictionSystem()
            
            assert system.model is not None
            assert system.data is not None
    
    def test_predict_injury_risk_success(self):
        """Test successful injury risk prediction."""
        with patch('injury_prediction_app.InjuryPredictionSystem.load_model') as mock_load_model, \
             patch('injury_prediction_app.InjuryPredictionSystem.load_data') as mock_load_data, \
             patch('injury_prediction_app.InjuryPredictionSystem._load_metadata') as mock_metadata:
            
            mock_metadata.return_value = {
                'decision_thresholds_days': [30, 60],
                'regression_metrics': {'r2': 0.85}
            }
            
            from injury_prediction_app import InjuryPredictionSystem
            
            system = InjuryPredictionSystem.__new__(InjuryPredictionSystem)
            system.model = Mock()
            system.model.predict = Mock(return_value=np.array([25.5]))
            system.feature_names = ['age', 'position_numeric']
            system.metadata = mock_metadata.return_value
            
            player_data = {'age': 25, 'position_numeric': 1}
            result = system.predict_injury_risk(player_data)
            
            assert 'predicted_injury_days' in result
            assert 'risk_level' in result
            assert result['risk_level'] in ['Low', 'Medium', 'High']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

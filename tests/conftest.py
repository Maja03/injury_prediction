"""
Pytest configuration and shared fixtures for the test suite.
"""

import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory."""
    return os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope="session")
def sample_player_data():
    """Sample player data for testing."""
    return {
        'p_id2': 'test_player_1',
        'age': 25.0,
        'position_numeric': 1,
        'total_minutes': 2000.0,
        'start_year': 2023
    }

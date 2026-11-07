"""Shared test data."""

import pytest


@pytest.fixture
def valid_order():
    return {
        "distance_km": 5.5,
        "items": 3,
        "hour": 19,
        "weather": "rain",
        "day_type": "weekend",
    }

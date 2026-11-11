"""Tests for validated late-delivery inference."""

import importlib.util
import math
from pathlib import Path

import joblib
import pytest

import predict


def test_valid_probabilities(valid_order):
    result = predict.predict_order(valid_order)

    assert 0 <= result["probability_late"] <= 1
    assert 0 <= result["probability_on_time"] <= 1
    assert (
        result["probability_late"] + result["probability_on_time"]
    ) == pytest.approx(1.0)


def test_harder_order_has_higher_risk():
    easy = {
        "distance_km": 0.5,
        "items": 1,
        "hour": 10,
        "weather": "clear",
        "day_type": "weekday",
    }

    hard = {
        "distance_km": 20,
        "items": 10,
        "hour": 19,
        "weather": "snow",
        "day_type": "weekend",
    }

    easy_risk = predict.predict_order(easy)["probability_late"]
    hard_risk = predict.predict_order(hard)["probability_late"]

    assert hard_risk > easy_risk


@pytest.mark.parametrize(
    "offset,expected",
    [
        (-0.000001, False),
        (0, True),
        (0.000001, True),
    ],
)
def test_threshold_boundary(offset, expected):
    probability = predict.THRESHOLD + offset

    assert predict.voucher_decision(probability) is expected


def test_threshold_comes_from_metadata():
    saved = float(predict.MODEL_INFO["threshold"])

    assert predict.THRESHOLD == saved
    assert predict.voucher_decision(saved) is True


def test_missing_prep_minutes_accepted(valid_order):
    order = {**valid_order, "prep_minutes": None}

    result = predict.predict_order(order)

    assert 0 <= result["probability_late"] <= 1
    assert any("prep_minutes" in message for message in result["warnings"])


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("distance_km", 0),
        ("distance_km", 31),
        ("items", 0),
        ("items", 21),
        ("items", 2.5),
        ("hour", 9),
        ("hour", 23),
        ("hour", 19.5),
    ],
)
def test_invalid_numbers(valid_order, field, bad_value):
    order = {**valid_order, field: bad_value}

    with pytest.raises(ValueError, match=field):
        predict.predict_order(order)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("weather", "Snow"),
        ("weather", "hail"),
        ("day_type", "holiday"),
    ],
)
def test_unknown_categories(valid_order, field, bad_value):
    order = {**valid_order, field: bad_value}

    with pytest.raises(ValueError, match=field):
        predict.predict_order(order)


@pytest.mark.parametrize(
    "field",
    predict.FEATURES,
)
def test_missing_required_field(valid_order, field):
    order = valid_order.copy()
    order.pop(field)

    with pytest.raises(ValueError, match=field):
        predict.predict_order(order)


def test_unexpected_field_rejected(valid_order):
    order = {**valid_order, "vehicle": "bike"}

    with pytest.raises(ValueError, match="vehicle"):
        predict.predict_order(order)


def test_nonfinite_number_rejected(valid_order):
    order = {**valid_order, "distance_km": math.inf}

    with pytest.raises(ValueError, match="distance_km"):
        predict.predict_order(order)


def test_model_loaded_once(monkeypatch, valid_order):
    """Count loading calls in a fresh module instance."""

    real_load = joblib.load
    calls = []

    def counting_load(*args, **kwargs):
        calls.append(args[0])
        return real_load(*args, **kwargs)

    monkeypatch.setattr(joblib, "load", counting_load)

    spec = importlib.util.spec_from_file_location(
        "_predict_load_test",
        Path(predict.__file__),
    )

    assert spec is not None
    assert spec.loader is not None

    fresh_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh_module)

    assert len(calls) == 1

    fresh_module.predict_order(valid_order)
    fresh_module.predict_order(valid_order)

    assert len(calls) == 1


@pytest.mark.parametrize("bad_value", [0, 0.5, 61])
def test_invalid_prep_minutes(valid_order, bad_value):
    order = {**valid_order, "prep_minutes": bad_value}

    with pytest.raises(ValueError, match="prep_minutes"):
        predict.predict_order(order)

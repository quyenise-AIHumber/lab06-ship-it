"""Validated inference for QuickBite late deliveries."""

import json
import math
from numbers import Real
from pathlib import Path

import joblib
import pandas as pd

# ----------------------------------
# 1. Load model and metadata ONCE
# ----------------------------------
ROOT = Path(__file__).resolve().parent

MODEL_INFO = json.loads(
    (ROOT / "models" / "model_info.json").read_text(encoding="utf-8")
)

MODEL = joblib.load(ROOT / "models" / "late_orders.joblib")

NUMERIC_FEATURES = MODEL_INFO["numeric_features"]
CATEGORICAL_FEATURES = MODEL_INFO["categorical_features"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

THRESHOLD = float(MODEL_INFO["threshold"])

# ----------------------------------
# 2. Input validation rules
# ----------------------------------
# Each numeric rule is (min, max, whole_number).
NUMERIC_RULES = {
    "distance_km": (0.1, 30, False),
    "items": (1, 20, True),
    "hour": (10, 22, True),
    "prep_minutes": (1, 60, False),
    "courier_months": (0, 120, False),
}

CATEGORY_RULES = {
    "weather": {"clear", "rain", "snow"},
    "day_type": {"weekday", "weekend"},
    "vehicle": {"bike", "scooter", "car"},
}


def is_missing(value):
    """Check for missing or blank input."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Real):
        return math.isnan(float(value))
    return False


def check_value(field, value, warnings):
    """Validate one field against its allowed values."""

    if is_missing(value):
        if field == "prep_minutes":
            warnings.append(
                "prep_minutes is missing; the model can "
                "handle this when it is a selected feature."
            )
            return float("nan")

        raise ValueError(f"{field}: this field is required.")

    if field in NUMERIC_RULES:
        minimum, maximum, whole_number = NUMERIC_RULES[field]

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{field}: enter a finite number.")

        number = float(value)

        if not minimum <= number <= maximum:
            raise ValueError(f"{field}: must be between {minimum} and {maximum}.")

        if whole_number and not number.is_integer():
            raise ValueError(f"{field}: must be a whole number.")

        return int(number) if whole_number else number

    if field in CATEGORY_RULES:
        if value not in CATEGORY_RULES[field]:
            choices = ", ".join(sorted(CATEGORY_RULES[field]))
            raise ValueError(f"{field}: choose one of {choices}.")

        return value

    raise ValueError(f"{field}: no validation rule exists.")


def validate_order(order):
    """Validate an order and build its model input."""

    if not isinstance(order, dict):
        raise ValueError("order: expected a dictionary.")

    allowed = set(FEATURES) | {"prep_minutes"}
    extra = set(order) - allowed

    if extra:
        field = sorted(extra)[0]
        raise ValueError(f"{field}: unknown input field.")

    cleaned = {}
    warnings = []

    # Validate the features required by the saved model.
    for field in FEATURES:
        cleaned[field] = check_value(field, order.get(field), warnings)

    # Support the optional preparation-time input,
    # even when the saved model does not use it.
    if "prep_minutes" in order and "prep_minutes" not in FEATURES:
        check_value(
            "prep_minutes",
            order["prep_minutes"],
            warnings,
        )
        warnings.append("prep_minutes is not used by this saved model.")

    frame = pd.DataFrame([cleaned], columns=FEATURES)

    return frame, warnings


def voucher_decision(probability):
    """Apply the threshold saved with the model."""

    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability: must be between 0 and 1.")

    return bool(probability >= THRESHOLD)


def predict_order(order):
    """Validate, score, and decide for one order."""

    frame, warnings = validate_order(order)

    probability = float(MODEL.predict_proba(frame)[0, 1])

    return {
        "probability_late": probability,
        "probability_on_time": 1 - probability,
        "offer_voucher": voucher_decision(probability),
        "threshold": THRESHOLD,
        "warnings": warnings,
    }

"""Manual smoke checks for the inference module."""

from predict import predict_order

valid_order = {
    "distance_km": 5.5,
    "items": 3,
    "hour": 19,
    "weather": "rain",
    "day_type": "weekend",
}

print("TEST 1: Valid order")
print(predict_order(valid_order))

print("\nTEST 2: Missing distance")
try:
    predict_order({**valid_order, "distance_km": None})
except ValueError as error:
    print("EXPECTED ERROR:", error)

print("\nTEST 3: Invalid items")
try:
    predict_order({**valid_order, "items": 25})
except ValueError as error:
    print("EXPECTED ERROR:", error)

print("\nTEST 4: Unknown weather")
try:
    predict_order({**valid_order, "weather": "Snow"})
except ValueError as error:
    print("EXPECTED ERROR:", error)

print("\nTEST 5: Optional missing preparation time")
print(predict_order({**valid_order, "prep_minutes": None}))

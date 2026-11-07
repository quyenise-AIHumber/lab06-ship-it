---
title: QuickBite Late Order Check
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: 3.12
app_file: app.py
pinned: false
short_description: Will this order arrive late?
---

# QuickBite Late Order Check

## What it does

This Gradio application estimates the probability that a
QuickBite order will arrive late when the order is placed.

It uses the saved Logistic Regression pipeline from Lab 05
and recommends a $4 voucher when late-delivery probability
meets or exceeds the saved decision threshold of 0.25.

Space URL: To be added after deployment.

## How to run it locally

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate lab06-ship-it
```

Run the application:

```bash
gradio app.py
```

Open the local address displayed in the terminal.

## Decisions

The interface reads its required features from model_info.json.

| Field | Allowed values | Reason |
|---|---|---|
| distance_km | 0.1 to 30 | Reject implausible input distances |
| items | Whole numbers 1 to 20 | Ensure valid order quantities |
| hour | Whole numbers 10 to 22 | Match supported ordering hours |
| weather | clear, rain, snow | Prevent unknown categories |
| day_type | weekday, weekend | Use recognized categories |

The following fields were deliberately excluded from
the saved Lab 05 model:

- order_id: identifier, not a legitimate predictor.
- prep_minutes: actual preparation time is unavailable
  when an order is first placed.
- customer_rating: collected after delivery and would
  create data leakage.
- vehicle: courier assignment timing is not confirmed.
- courier_months: courier identity may be unknown.

The inference module can accept an optional missing
prep_minutes input and issue a warning, but this field
is not used by the saved model and is not shown in the
five-feature dispatcher form.

The input ranges follow the lab requirements. Some accepted
values are outside the most common values in the synthetic
training data, so extreme inputs may be less reliable.

The decision threshold is read from model_info.json.
It is never independently hardcoded into the application.

The input validator rejects unknown categories, including
"Snow" when the allowed label is "snow". This prevents the
one-hot encoder from silently treating an unexpected
category as an unrecognized value.

## Result

The model was developed using 3,000 simulated orders.

Selected model: Logistic Regression
Decision threshold: 0.25

Held-out test set: 750 orders

Confusion matrix:
- True negatives: 567
- False positives: 66
- False negatives: 64
- True positives: 53

Precision: 0.4454
Recall: 0.4530
F1 score: 0.4492
ROC-AUC: 0.7646

Business costs on the held-out test set:

| Policy | Total cost |
|---|---:|
| Flag nobody | $1,755 |
| Flag everybody | $3,000 |
| Model with threshold 0.25 | $1,436 |

Savings versus flag nobody: $319, or 18.18%.

These results come from the simulated Lab 05 test set.
They are not validated real-world savings.

## Tests

Automated tests cover:

- Valid prediction probabilities.
- Harder orders having higher predicted risk.
- Voucher decisions at the threshold boundary.
- Threshold loading from saved metadata.
- Optional missing preparation time.
- Invalid numeric inputs and missing fields.
- Unknown categorical values.
- Loading the saved model once.
- Consistency of dependencies and deployment settings.

Run:

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Limitations

The model was trained on synthetic food delivery orders,
not on real QuickBite operational records.

It cannot observe road closures, unexpected restaurant
disruptions, live courier availability, or other operational
conditions absent from its selected features.

It should not be used to guarantee arrival times or make
automated customer compensation decisions without
human review and further real-world validation.

Model performance and voucher costs should be monitored
before any production deployment.

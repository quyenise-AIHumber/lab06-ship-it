"""QuickBite dispatcher interface built with Gradio Blocks."""

import gradio as gr

from predict import (
    CATEGORY_RULES,
    FEATURES,
    NUMERIC_RULES,
    predict_order,
)

# Compatibility placeholder for free ZeroGPU Spaces.
# It does not perform inference or request GPU resources.
try:
    import spaces
except ImportError:
    spaces = None

if spaces is not None:

    @spaces.GPU
    def _gpu_placeholder():
        return None


def check_delivery(*values):
    """Send dispatcher inputs to the prediction engine."""

    order = dict(zip(FEATURES, values, strict=True))

    try:
        result = predict_order(order)
    except ValueError as error:
        # Clear any previous results when input is invalid.
        return (
            "",
            "",
            f"INVALID INPUT: {error}",
            "",
            "",
        )

    probability_late = result["probability_late"]
    probability_on_time = result["probability_on_time"]

    if result["offer_voucher"]:
        recommendation = "Offer the customer a $4 voucher."
    else:
        recommendation = "No voucher needed."

    warning = "\n".join(result["warnings"])

    return (
        f"{probability_late:.1%}",
        f"{probability_on_time:.1%}",
        recommendation,
        f"{result['threshold']:.2f}",
        warning or "No warnings.",
    )


with gr.Blocks(title="QuickBite Late Order Check") as demo:
    gr.Markdown(
        "# QuickBite Late Order Check\n"
        "Predict the risk of late delivery when an order "
        "is placed."
    )

    gr.Markdown("### Order information")

    inputs = []

    # Create exactly the fields expected by the model.
    # FEATURES comes from model_info.json via predict.py.
    with gr.Row():
        with gr.Column():
            for field in FEATURES:
                label = field.replace("_", " ").title()

                if field in NUMERIC_RULES:
                    minimum, maximum, _ = NUMERIC_RULES[field]

                    component = gr.Number(
                        label=(f"{label} ({minimum:g} to {maximum:g})"),
                        value=None,
                    )

                elif field in CATEGORY_RULES:
                    component = gr.Dropdown(
                        label=label,
                        choices=sorted(CATEGORY_RULES[field]),
                        value=None,
                        allow_custom_value=False,
                    )

                else:
                    raise ValueError(f"No UI rule for {field}")

                inputs.append(component)

        with gr.Column():
            gr.Markdown("### Prediction results")

            late_output = gr.Textbox(
                label="Chance of late delivery",
                interactive=False,
            )
            on_time_output = gr.Textbox(
                label="Chance of on-time delivery",
                interactive=False,
            )
            action_output = gr.Textbox(
                label="Recommended action",
                interactive=False,
                lines=2,
            )
            threshold_output = gr.Textbox(
                label="Decision threshold",
                interactive=False,
            )
            warning_output = gr.Textbox(
                label="Warnings and validation",
                interactive=False,
                lines=2,
            )

    predict_button = gr.Button(
        "Check Delivery Risk",
        variant="primary",
    )

    predict_button.click(
        fn=check_delivery,
        inputs=inputs,
        outputs=[
            late_output,
            on_time_output,
            action_output,
            threshold_output,
            warning_output,
        ],
    )

    gr.Markdown(
        "This is a demonstration based on simulated "
        "delivery data. Predictions are not guarantees."
    )


if __name__ == "__main__":
    demo.launch()

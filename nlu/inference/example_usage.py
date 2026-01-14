"""Example: how to import and use NLUModel from other modules.

This file demonstrates the intended usage for other code in the project.
"""
import os
from inference.predictor import NLUModel


def example():
    base_dir = os.environ.get("NLU_BASE_DIR", "outputs/snips_joint")
    try:
        model = NLUModel(base_dir=base_dir)
        model.load(device=os.environ.get("NLU_DEVICE", "cpu"))
    except FileNotFoundError as e:
        print(f"Cannot run example because checkpoint not found: {e}")
        return

    text = "what is the weather like in munich tomorrow"
    res = model.predict(text)
    print("Intent:", res["intent"])
    print("Word-level slots:", res["word_slots"])


if __name__ == "__main__":
    example()
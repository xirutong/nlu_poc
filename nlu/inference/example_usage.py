"""Example: how to import and use NLUModel from other modules.

This file demonstrates the intended usage for other code in the project.
"""
from nlu.inference.predictor import NLUModel

def example():
    try:
        model = NLUModel()
        model.load(device="cpu")
    except FileNotFoundError as e:
        print(f"Cannot run example because checkpoint not found: {e}")
        return

    text = "I want to book a table for two at an Italian restaurant tomorrow night"
    res = model.predict(text)
    #print("Intent:", res["intent"])
    #print("Word-level slots:", res["word_slots"])


if __name__ == "__main__":
    example()
import os
import json
import pandas as pd

def log_to_csv(results: list, filename: str):
    """
    Save list of dictionaries to a CSV file under log/
    """
    df = pd.DataFrame(results)
    filepath = os.path.join("logs", filename)
    df.to_csv(filepath, index=False)
    print(f"Saved results to {filepath}")

def log_to_json(results: list, filename: str):
    """
    Save list of dictionaries to a JSON file under log/
    """
    filepath = os.path.join("logs", filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {filepath}")

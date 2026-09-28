import pandas as pd
from pathlib import Path


def load_raw_data():
    current_dir = Path(__file__).resolve().parent
    file_path = current_dir.parent.parent / "data" / "raw" / "customer_churn_raw_data.csv"

    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: failed fetching customer_churn_raw_data file!")
        return None
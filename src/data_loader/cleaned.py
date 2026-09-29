import pandas as pd
from pathlib import Path


def load_clean_data():
    current_dir = Path(__file__).resolve().parent
    file_path = current_dir.parent.parent / "data" / "processed" / "cleaned.csv"

    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: failed fetching cleaned file!")
        return None
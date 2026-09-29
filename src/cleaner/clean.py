from pathlib import Path

import pandas as pd

def clean(df: pd.DataFrame):
    # remove duplicates
    df = df.drop_duplicates()

    # convert TotalCharges Column and handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df[df['tenure'] > 0].copy()
    df['TotalCharges'] = df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'])

    # drop customerID column
    df = df.drop(columns=['customerID'])

    # standardize columns to simple values
    replace_cols = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df[replace_cols] = df[replace_cols].replace({
        'No internet service': 'No',
        'No phone service': 'No'
    })

    # redefine automatic payments
    df['PaymentMethod'] = df['PaymentMethod'].replace({
        'Bank transfer (automatic)': 'automatic',
        'Credit card (automatic)': 'automatic'
    })

    # save the Dataframe
    root_path = Path(__file__).resolve().parents[2]
    df.to_csv(root_path / "data" / "processed" / "cleaned.csv", index=False)

    return df
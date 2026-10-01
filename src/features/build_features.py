from pathlib import Path
import pandas as pd


def build(df: pd.DataFrame):

    # sum the number of active services
    services_col = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies']
    df['active_services'] = (df[services_col] == 'Yes').sum(axis=1)

    # save the Dataframe
    root_path = Path(__file__).resolve().parents[2]
    df.to_csv(root_path / "data" / "processed" / "featured.csv", index=False)

    return df

#FILE RESPONSIBLE FOR READING DATA
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "Mental_Health_and_Social_Media_Balance_Dataset.csv"

def load_data(path=DATA_PATH):
    print("\n--- Loading Dataset ---")
    df = pd.read_csv(path)

    print("\n--- Dataset Shape ---")
    print(df.shape)

    print("\n--- Feature Names ---")
    print(df.columns.tolist())

    # Automatically correcting the Happiness Index column
    if "Happiness_Index" not in df.columns:
        for col in df.columns:
            if "happiness" in col.lower():
                print(f"\n'{col}' -> renamed to 'Happiness_Index'.")
                df = df.rename(columns={col: "Happiness_Index"})
                break

    return df
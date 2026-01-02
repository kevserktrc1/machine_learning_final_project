

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    print("\n--- Preprocessing Started ---")


    if "User_ID" in df.columns:
        df = df.drop(columns=["User_ID"])
        print("The User_ID column has been removed.")

    #categorical columns were identified.
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print("Categorical variables:", categorical_cols)

    # One-hot encoding
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print("One-hot encoding completed.")


    y = df["Happiness_Index"]
    X = df.drop(columns=["Happiness_Index"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Train-test split completed.")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("StandardScaler was applied.")

    print("--- Preprocessing is over ---")

    return X_train_scaled, X_test_scaled, y_train, y_test

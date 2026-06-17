import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical columns
    encoder = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:
        df[col] = encoder.fit_transform(df[col])

    return df
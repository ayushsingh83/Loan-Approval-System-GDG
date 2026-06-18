import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder


def preprocess_data(filepath):

    df = pd.read_csv(filepath)

    # Fill missing values
    for col in df.columns:

        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mode()[0])

        else:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical columns
    encoders = {}

    for col in df.columns:

        if pd.api.types.is_string_dtype(df[col]):

            encoder = LabelEncoder()

            df[col] = encoder.fit_transform(df[col])

            encoders[col] = encoder

    # Save encoders
    with open("models/encoder.pkl", "wb") as file:
        pickle.dump(encoders, file)

    return df
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            try:
                df[col] = df[col].fillna(df[col].median())
            except:
                pass

    # Encode categorical columns
    encoder = LabelEncoder()

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in categorical_cols:
    	df[col] = encoder.fit_transform(df[col].astype(str))

    return df
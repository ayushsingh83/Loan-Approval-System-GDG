"""
preprocessing.py
-----------------
Shared data-cleaning and feature-engineering logic used by BOTH
train_model.py (training time) and app.py (inference time).

Keeping this logic in one place guarantees that the exact same
transformations are applied to training data and to live user
input from the Flask form.
"""

import pandas as pd
import numpy as np


# --------------------------------------------------------------------------
# Column name constants
# --------------------------------------------------------------------------
NUMERIC_COLS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

CATEGORICAL_COLS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

TARGET_COL = "Loan_Status"


# --------------------------------------------------------------------------
# Step 1: Raw value cleaning
# --------------------------------------------------------------------------
def clean_raw_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix inconsistent / noisy raw values found in the dataset
    (typos, alternate spellings, invalid codes) BEFORE any
    encoding or imputation happens.
    """
    df = df.copy()

    # ---- Gender: fix typos -------------------------------------------------
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({
            "Mle": "Male",
            "Fmale": "Female",
        })

    # ---- Married: standardise Yes/No --------------------------------------
    if "Married" in df.columns:
        df["Married"] = df["Married"].replace({
            "Y": "Yes",
            "N": "No",
        })

    # ---- Dependents: standardise to string codes ---------------------------
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace({
            "four": "4",
            "3+": "3",
        })

    # ---- Self_Employed: fix invalid category -------------------------------
    if "Self_Employed" in df.columns:
        df["Self_Employed"] = df["Self_Employed"].replace({
            "Self": "Yes",
        })

    # ---- Property_Area: standardise spelling --------------------------------
    if "Property_Area" in df.columns:
        df["Property_Area"] = df["Property_Area"].replace({
            "semi-urban": "Semiurban",
            "Metro": "Urban",
        })

    # ---- Education: standardise spelling -------------------------------------
    if "Education" in df.columns:
        df["Education"] = df["Education"].replace({
            "Grad": "Graduate",
        })

    # ---- LoanAmount: remove impossible / outlier values --------------------
    if "LoanAmount" in df.columns:
        # Negative loan amounts are invalid -> treat as missing
        df.loc[df["LoanAmount"] < 0, "LoanAmount"] = np.nan
        # 9999 is a clear placeholder/outlier sentinel -> treat as missing
        df.loc[df["LoanAmount"] >= 9999, "LoanAmount"] = np.nan

    # ---- Loan_Amount_Term: remove sentinel outlier --------------------------
    if "Loan_Amount_Term" in df.columns:
        df.loc[df["Loan_Amount_Term"] >= 999, "Loan_Amount_Term"] = np.nan

    # ---- Credit_History: only 0 / 1 are valid --------------------------------
    if "Credit_History" in df.columns:
        df.loc[~df["Credit_History"].isin([0, 1]), "Credit_History"] = np.nan

    return df


# --------------------------------------------------------------------------
# Step 2: Missing value imputation
# --------------------------------------------------------------------------
def impute_missing(df: pd.DataFrame, medians: dict = None, modes: dict = None):
    """
    Fill missing values.
      - Numerical columns -> median
      - Categorical columns -> mode

    If `medians` / `modes` dicts are supplied (computed at training time),
    they are used directly (needed for consistent inference on single rows).
    Otherwise they are computed from `df` itself (used during training).

    Returns: (df_filled, medians_used, modes_used)
    """
    df = df.copy()

    medians_used = {}
    for col in NUMERIC_COLS:
        if col in df.columns:
            fill_value = medians[col] if medians and col in medians else df[col].median()
            medians_used[col] = fill_value
            df[col] = df[col].fillna(fill_value)

    modes_used = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            fill_value = modes[col] if modes and col in modes else df[col].mode()[0]
            modes_used[col] = fill_value
            df[col] = df[col].fillna(fill_value)

    return df, medians_used, modes_used


# --------------------------------------------------------------------------
# Step 3: Feature engineering
# --------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that help the models capture
    repayment-capacity signals beyond the raw columns.
    """
    df = df.copy()

    # Total household income
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    # Log-transform skewed monetary columns (reduces impact of outliers)
    df["LoanAmount_log"] = np.log1p(df["LoanAmount"])
    df["TotalIncome_log"] = np.log1p(df["TotalIncome"])

    # EMI-style ratio: how big is the loan relative to total income
    # (add 1 to avoid division by zero)
    df["Loan_Income_Ratio"] = df["LoanAmount"] / (df["TotalIncome"] + 1)

    # Approximate monthly EMI assuming equal principal repayment over term
    df["EMI"] = df["LoanAmount"] / (df["Loan_Amount_Term"].replace(0, np.nan))
    df["EMI"] = df["EMI"].fillna(0)

    # Balance income left after EMI
    df["Balance_Income"] = (df["ApplicantIncome"] / 12) - df["EMI"]

    return df


# --------------------------------------------------------------------------
# Step 4: Encoding
# --------------------------------------------------------------------------
def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Label-encode categorical columns.

    If `fit` is True, fits new LabelEncoders and returns them.
    If `fit` is False, uses the provided `encoders` dict to transform
    (used at inference time). Unseen categories fall back to the
    most frequent class seen during training.
    """
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    encoders = encoders or {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue

        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen categories gracefully by mapping to the first class
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    return df, encoders


# --------------------------------------------------------------------------
# Full pipeline for a single inference-time record (used by Flask app)
# --------------------------------------------------------------------------
FEATURE_COLUMNS_BEFORE_SCALING = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area",
    "TotalIncome", "LoanAmount_log", "TotalIncome_log",
    "Loan_Income_Ratio", "EMI", "Balance_Income",
]

SCALED_NUMERIC_COLS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "TotalIncome", "LoanAmount_log", "TotalIncome_log",
    "Loan_Income_Ratio", "EMI", "Balance_Income",
]


def preprocess_single_record(raw_dict: dict, encoders: dict, scaler, medians: dict, modes: dict) -> pd.DataFrame:
    """
    Take a raw dict of form-input values (as strings/numbers from the
    Flask form), run it through the SAME cleaning -> imputation ->
    feature engineering -> encoding -> scaling pipeline used at
    training time, and return a single-row DataFrame ready for
    model.predict().
    """
    df = pd.DataFrame([raw_dict])

    # Step 1: clean raw values (fix typos etc.)
    df = clean_raw_values(df)

    # Step 2: impute any missing values using training-time medians/modes
    df, _, _ = impute_missing(df, medians=medians, modes=modes)

    # Step 3: feature engineering
    df = engineer_features(df)

    # Step 4: encode categoricals using saved encoders
    df, _ = encode_categoricals(df, encoders=encoders, fit=False)

    # Reorder columns to match training feature order
    df = df[FEATURE_COLUMNS_BEFORE_SCALING]

    # Step 5: scale numeric columns using saved scaler
    df[SCALED_NUMERIC_COLS] = scaler.transform(df[SCALED_NUMERIC_COLS])

    return df

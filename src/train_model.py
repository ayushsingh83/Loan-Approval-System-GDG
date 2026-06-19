import sys
import os
import pickle

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from preprocessing import (
    clean_raw_values,
    impute_missing,
    engineer_features,
    encode_categoricals,
    FEATURE_COLUMNS_BEFORE_SCALING,
    SCALED_NUMERIC_COLS,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset/Loan_approval.csv")

print("Dataset Shape:", df.shape)

# Step 1: Cleaning
df = clean_raw_values(df)

# Step 2: Missing values
df, medians, modes = impute_missing(df)

# Step 3: Feature Engineering
df = engineer_features(df)

# Step 4: Encode categoricals
df, encoders = encode_categoricals(df, fit=True)

# Encode target column
target_encoder = None
if "Loan_Status" in df.columns:
    from sklearn.preprocessing import LabelEncoder

    target_encoder = LabelEncoder()
    df["Loan_Status"] = target_encoder.fit_transform(
        df["Loan_Status"].astype(str)
    )

# Features and target
X = df[FEATURE_COLUMNS_BEFORE_SCALING]
y = df["Loan_Status"]

# Feature Scaling
scaler = StandardScaler()
X[SCALED_NUMERIC_COLS] = scaler.fit_transform(X[SCALED_NUMERIC_COLS])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

print("\nLogistic Regression Accuracy:", lr_accuracy)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Random Forest Accuracy:", rf_accuracy)

# Select best model
best_model = rf_model if rf_accuracy >= lr_accuracy else lr_model

# Save model
os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as file:
    pickle.dump(best_model, file)

with open("models/scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

with open("models/encoder.pkl", "wb") as file:
    pickle.dump(encoders, file)

with open("models/imputation_values.pkl", "wb") as file:
    pickle.dump(
        {
            "medians": medians,
            "modes": modes,
        },
        file,
    )

print("\nModel saved successfully!")
print("Scaler saved successfully!")
print("Encoders saved successfully!")
print("Imputation values saved successfully!")

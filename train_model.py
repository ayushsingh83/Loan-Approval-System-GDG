"""
train_model.py
---------------
End-to-end training pipeline for the Loan Approval Prediction System.

Steps performed:
    1. Load dataset
    2. Exploratory Data Analysis (EDA) - prints summaries & saves plots
    3. Data cleaning & missing-value imputation
    4. Feature engineering
    5. Encoding categorical variables
    6. Train/test split & scaling
    7. Train Logistic Regression, Decision Tree, Random Forest
    8. Evaluate & compare models (Accuracy, Precision, Recall, F1, Confusion Matrix)
    9. Select the best model and persist model.pkl, encoder.pkl, scaler.pkl

Run:
    python train_model.py
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for saving plots without a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from preprocessing import (
    clean_raw_values,
    impute_missing,
    engineer_features,
    encode_categoricals,
    FEATURE_COLUMNS_BEFORE_SCALING,
    SCALED_NUMERIC_COLS,
    TARGET_COL,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Loan_approval.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "images")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def main():
    # ----------------------------------------------------------------------
    # 1. Load dataset
    # ----------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: LOADING DATASET")
    print("=" * 70)
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        raise SystemExit(
            f"Dataset not found at {DATA_PATH}. "
            f"Place 'Loan_approval.csv' inside the 'dataset/' folder."
        )

    print(f"Dataset loaded successfully. Shape: {df.shape}")

    # Drop Loan_ID if present (not predictive)
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])
        print("Dropped 'Loan_ID' column.")

    # ----------------------------------------------------------------------
    # 2. Exploratory Data Analysis
    # ----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nTarget distribution (Loan_Status):")
    print(df[TARGET_COL].value_counts())
    print(df[TARGET_COL].value_counts(normalize=True).round(3))

    print("\nNumerical summary statistics:")
    print(df.describe())

    # ---- Visualizations -----------------------------------------------------
    sns.set_style("whitegrid")

    # Target distribution plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x=TARGET_COL, data=df, palette="viridis")
    plt.title("Loan Status Distribution")
    plt.savefig(os.path.join(PLOTS_DIR, "target_distribution.png"), bbox_inches="tight")
    plt.close()

    # Numerical feature distributions
    numeric_cols_raw = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flatten(), numeric_cols_raw):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "numerical_distributions.png"), bbox_inches="tight")
    plt.close()

    # Categorical feature counts vs target
    cat_cols_raw = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, col in zip(axes.flatten(), cat_cols_raw):
        sns.countplot(x=col, hue=TARGET_COL, data=df, ax=ax, palette="Set2")
        ax.set_title(f"{col} vs Loan Status")
        ax.tick_params(axis="x", rotation=30)
    fig.delaxes(axes.flatten()[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "categorical_vs_target.png"), bbox_inches="tight")
    plt.close()

    # Correlation heatmap (numerical columns only, before encoding)
    plt.figure(figsize=(8, 6))
    corr = df[numeric_cols_raw + ["Credit_History"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (Numerical Features)")
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), bbox_inches="tight")
    plt.close()

    print(f"\nEDA plots saved to: {PLOTS_DIR}")

    # ----------------------------------------------------------------------
    # 3. Data Cleaning & Missing Value Imputation
    # ----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: DATA CLEANING & MISSING VALUE IMPUTATION")
    print("=" * 70)

    df = clean_raw_values(df)
    print("Cleaned inconsistent raw values (typos, invalid codes, outliers).")

    df, medians, modes = impute_missing(df)
    print(f"Numerical columns imputed with median: {medians}")
    print(f"Categorical columns imputed with mode: {modes}")
    print("\nMissing values after imputation:")
    print(df.isnull().sum())

    # ----------------------------------------------------------------------
    # 4. Feature Engineering
    # ----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 70)

    df = engineer_features(df)
    print("Created derived features: TotalIncome, LoanAmount_log, "
          "TotalIncome_log, Loan_Income_Ratio, EMI, Balance_Income")

    # Correlation of new numeric features with target (after temp encoding)
    temp_target = df[TARGET_COL].map({"Y": 1, "N": 0})
    new_feats = ["TotalIncome", "LoanAmount_log", "TotalIncome_log",
                  "Loan_Income_Ratio", "EMI", "Balance_Income", "Credit_History"]
    corr_with_target = df[new_feats].corrwith(temp_target).sort_values(ascending=False)
    print("\nCorrelation of engineered features with target:")
    print(corr_with_target)

    # ----------------------------------------------------------------------
    # 5. Encoding
    # ----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: ENCODING CATEGORICAL FEATURES")
    print("=" * 70)

    df, encoders = encode_categoricals(df, fit=True)
    print(f"Label-encoded columns: {list(encoders.keys())}")
    for col, le in encoders.items():
        print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Encode target
    df[TARGET_COL] = df[TARGET_COL].map({"Y": 1, "N": 0})

    # ----------------------------------------------------------------------
    # 6. Train/Test Split & Scaling
    # ----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: TRAIN/TEST SPLIT & SCALING")
    print("=" * 70)

    X = df[FEATURE_COLUMNS_BEFORE_SCALING]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[SCALED_NUMERIC_COLS] = scaler.fit_transform(X_train[SCALED_NUMERIC_COLS])
    X_test[SCALED_NUMERIC_COLS] = scaler.transform(X_test[SCALED_NUMERIC_COLS])
    print(f"Scaled numerical columns: {SCALED_NUMERIC_COLS}")

    # ----------------------------------------------------------------------
    # 7. Model Training
    # ----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7: MODEL TRAINING")
    print("=" * 70)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    }

    results = []
    trained_models = {}
    confusion_matrices = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
        })
        trained_models[name] = model
        confusion_matrices[name] = cm

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

    # ----------------------------------------------------------------------
    # 8. Model Comparison
    # ----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 8: MODEL COMPARISON")
    print("=" * 70)

    results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
    print("\n" + results_df.to_string(index=False))

    # Plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Rejected", "Approved"],
                    yticklabels=["Rejected", "Approved"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"), bbox_inches="tight")
    plt.close()

    # Plot model comparison bar chart
    plt.figure(figsize=(10, 5))
    results_melt = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(x="Model", y="Score", hue="Metric", data=results_melt, palette="muted")
    plt.title("Model Comparison")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------------------------
    # 9. Select Best Model & Persist
    # ----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 9: SELECTING BEST MODEL & SAVING ARTIFACTS")
    print("=" * 70)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\nBest model selected: {best_model_name}")
    print(f"  F1 Score : {results_df.iloc[0]['F1 Score']}")
    print(f"  Accuracy : {results_df.iloc[0]['Accuracy']}")

    # Save model
    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    # Save encoders + medians + modes (everything needed for preprocessing)
    preprocessing_objects = {
        "encoders": encoders,
        "medians": medians,
        "modes": modes,
        "feature_columns": FEATURE_COLUMNS_BEFORE_SCALING,
        "best_model_name": best_model_name,
    }
    with open(os.path.join(MODEL_DIR, "encoder.pkl"), "wb") as f:
        pickle.dump(preprocessing_objects, f)

    # Save scaler
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved model.pkl, encoder.pkl, scaler.pkl to: {MODEL_DIR}")
    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()

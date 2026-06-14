import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Loan_approval.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())
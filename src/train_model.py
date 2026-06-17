import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from preprocessing import preprocess_data
# Load and preprocess dataset
df = preprocess_data("dataset/Loan_approval.csv")

print("Dataset Shape:", df.shape)

print("\nProcessed Dataset:")
print(df.head())
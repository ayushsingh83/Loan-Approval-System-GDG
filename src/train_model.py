import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load and preprocess dataset
df = preprocess_data("dataset/Loan_approval.csv")

print("Dataset Shape:", df.shape)

# Features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel trained successfully!")
from flask import Flask, render_template, request
import pickle

from preprocessing import (
preprocess_single_record
)

app = Flask(__name__)

# Load model

with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler

with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Load encoders

with open("models/encoder.pkl", "rb") as file:
    encoders = pickle.load(file)

# Load imputation values

with open("models/imputation_values.pkl", "rb") as file:
    imputation_data = pickle.load(file)

    medians = imputation_data["medians"]
    modes = imputation_data["modes"]

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        raw_input = {
            "Gender": request.form["Gender"],
            "Married": request.form["Married"],
            "Dependents": request.form["Dependents"],
            "Education": request.form["Education"],
            "Self_Employed": request.form["Self_Employed"],
            "ApplicantIncome": float(request.form["ApplicantIncome"]),
            "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
            "LoanAmount": float(request.form["LoanAmount"]),
            "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
            "Credit_History": float(request.form["Credit_History"]),
            "Property_Area": request.form["Property_Area"],
        }

        processed_input = preprocess_single_record(
            raw_input,
            encoders,
            scaler,
            medians,
            modes
        )

        prediction = model.predict(processed_input)

        if prediction[0] == 1:
            prediction = "Loan Approved"
        else:
            prediction = "Loan Rejected"

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)

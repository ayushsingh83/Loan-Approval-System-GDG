"""
app.py
------
Flask web application for the Loan Approval Prediction System.

Loads the trained model, encoders, and scaler (produced by train_model.py)
and serves a form where users can enter applicant details and receive
a loan approval / rejection prediction with probability.
"""

import os
import pickle

from flask import Flask, render_template, request

from preprocessing import preprocess_single_record

# --------------------------------------------------------------------------
# App initialisation
# --------------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# --------------------------------------------------------------------------
# Load model artifacts at startup
# --------------------------------------------------------------------------
model = None
scaler = None
encoders = None
medians = None
modes = None
best_model_name = "Model"
ARTIFACTS_LOADED = False
LOAD_ERROR = None

try:
    with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb") as f:
        preprocessing_objects = pickle.load(f)
        encoders = preprocessing_objects["encoders"]
        medians = preprocessing_objects["medians"]
        modes = preprocessing_objects["modes"]
        best_model_name = preprocessing_objects.get("best_model_name", "Model")

    ARTIFACTS_LOADED = True
except Exception as exc:  # pragma: no cover - defensive loading
    LOAD_ERROR = str(exc)
    print(f"[WARNING] Could not load model artifacts: {exc}")
    print("Run 'python train_model.py' first to generate model/*.pkl files.")


# --------------------------------------------------------------------------
# Form field definitions (used for validation)
# --------------------------------------------------------------------------
GENDER_OPTIONS = ["Male", "Female"]
MARRIED_OPTIONS = ["Yes", "No"]
DEPENDENTS_OPTIONS = ["0", "1", "2", "3"]
EDUCATION_OPTIONS = ["Graduate", "Not Graduate"]
SELF_EMPLOYED_OPTIONS = ["Yes", "No"]
PROPERTY_AREA_OPTIONS = ["Urban", "Semiurban", "Rural"]


def validate_form(form) -> dict:
    """
    Validate and parse the incoming form data.

    Returns a dict of:
        {"data": {...cleaned values...}, "errors": [...]}
    """
    errors = []
    data = {}

    # ---- Categorical fields ------------------------------------------------
    gender = form.get("Gender", "").strip()
    if gender not in GENDER_OPTIONS:
        errors.append("Please select a valid Gender.")
    data["Gender"] = gender

    married = form.get("Married", "").strip()
    if married not in MARRIED_OPTIONS:
        errors.append("Please select a valid Marital status.")
    data["Married"] = married

    dependents = form.get("Dependents", "").strip()
    if dependents not in DEPENDENTS_OPTIONS:
        errors.append("Please select a valid number of Dependents.")
    data["Dependents"] = dependents

    education = form.get("Education", "").strip()
    if education not in EDUCATION_OPTIONS:
        errors.append("Please select a valid Education level.")
    data["Education"] = education

    self_employed = form.get("Self_Employed", "").strip()
    if self_employed not in SELF_EMPLOYED_OPTIONS:
        errors.append("Please select a valid Self-Employed status.")
    data["Self_Employed"] = self_employed

    property_area = form.get("Property_Area", "").strip()
    if property_area not in PROPERTY_AREA_OPTIONS:
        errors.append("Please select a valid Property Area.")
    data["Property_Area"] = property_area

    # ---- Numerical fields ----------------------------------------------------
    def parse_float(field_name, label, min_value=0, allow_zero=True):
        raw_value = form.get(field_name, "").strip()
        try:
            value = float(raw_value)
            if value < min_value or (not allow_zero and value == 0):
                errors.append(f"{label} must be greater than {min_value}.")
            return value
        except (ValueError, TypeError):
            errors.append(f"{label} must be a valid number.")
            return None

    data["ApplicantIncome"] = parse_float("ApplicantIncome", "Applicant Income", min_value=0, allow_zero=False)
    data["CoapplicantIncome"] = parse_float("CoapplicantIncome", "Co-applicant Income", min_value=0, allow_zero=True)
    data["LoanAmount"] = parse_float("LoanAmount", "Loan Amount", min_value=0, allow_zero=False)
    data["Loan_Amount_Term"] = parse_float("Loan_Amount_Term", "Loan Amount Term", min_value=0, allow_zero=False)

    # Credit history: must be 0 or 1
    credit_history_raw = form.get("Credit_History", "").strip()
    if credit_history_raw not in ("0", "1"):
        errors.append("Please select a valid Credit History (0 or 1).")
        data["Credit_History"] = None
    else:
        data["Credit_History"] = float(credit_history_raw)

    return {"data": data, "errors": errors}


@app.route("/", methods=["GET", "POST"])
def index():
    """
    GET  -> render the empty form
    POST -> validate input, run prediction pipeline, render result
    """
    prediction = None
    probability = None
    error_message = None
    form_values = {}

    if not ARTIFACTS_LOADED:
        error_message = (
            "Model artifacts are not available. "
            "Please run 'python train_model.py' to generate model/*.pkl files."
        )

    if request.method == "POST" and ARTIFACTS_LOADED:
        form_values = request.form.to_dict()
        validation = validate_form(request.form)
        errors = validation["errors"]
        cleaned_data = validation["data"]

        if errors:
            error_message = " ".join(errors)
        else:
            try:
                # Run the SAME preprocessing pipeline used during training
                processed_df = preprocess_single_record(
                    cleaned_data, encoders=encoders, scaler=scaler,
                    medians=medians, modes=modes,
                )

                pred = model.predict(processed_df)[0]

                # Probability of "Approved" class (label 1)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(processed_df)[0]
                    probability = round(float(proba[1]) * 100, 2)
                else:
                    probability = None

                prediction = "Approved" if pred == 1 else "Rejected"

            except Exception as exc:
                error_message = f"An error occurred while making the prediction: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error_message=error_message,
        form_values=form_values,
        gender_options=GENDER_OPTIONS,
        married_options=MARRIED_OPTIONS,
        dependents_options=DEPENDENTS_OPTIONS,
        education_options=EDUCATION_OPTIONS,
        self_employed_options=SELF_EMPLOYED_OPTIONS,
        property_area_options=PROPERTY_AREA_OPTIONS,
        best_model_name=best_model_name,
        artifacts_loaded=ARTIFACTS_LOADED,
    )


@app.route("/health")
def health():
    """Simple health-check endpoint (useful for deployment platforms)."""
    return {"status": "ok", "model_loaded": ARTIFACTS_LOADED}


if __name__ == "__main__":
    # Debug mode should be turned off in production
    app.run(debug=True, host="0.0.0.0", port=5000)

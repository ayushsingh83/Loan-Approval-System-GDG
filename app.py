from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        gender = request.form["Gender"]
        married = request.form["Married"]
        dependents = int(request.form["Dependents"])
        education = request.form["Education"]
        self_employed = request.form["Self_Employed"]

        applicant_income = float(request.form["ApplicantIncome"])
        coapplicant_income = float(request.form["CoapplicantIncome"])
        loan_amount = float(request.form["LoanAmount"])
        loan_term = float(request.form["Loan_Amount_Term"])
        credit_history = float(request.form["Credit_History"])

        property_area = request.form["Property_Area"]

        print("Form data received successfully")

        # Basic encoding
        gender = 1 if gender.lower() == "male" else 0
        married = 1 if married.lower() == "yes" else 0
        education = 1 if education.lower() == "graduate" else 0
        self_employed = 1 if self_employed.lower() == "yes" else 0

        property_map = {
            "rural": 0,
            "semiurban": 1,
            "urban": 2
        }

        property_area = property_map.get(
            property_area.lower(),
            0
        )

        input_data = [[
            0,                      # Loan_ID placeholder
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            property_area
        ]]

        print(input_data)

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)
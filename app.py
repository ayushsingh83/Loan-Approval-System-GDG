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
        income = float(request.form["ApplicantIncome"])

        # Convert inputs to numbers
        if gender.lower() == "male":
            gender = 1
        else:
            gender = 0

        if married.lower() == "yes":
            married = 1
        else:
            married = 0

        # Temporary feature vector
        input_data = [gender, married, income]

        # First attempt at model prediction
        prediction = model.predict([input_data])[0]

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)
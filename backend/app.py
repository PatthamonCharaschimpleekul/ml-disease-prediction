from flask import Flask, render_template, request
from predict import predict_disease
import pickle

app = Flask(__name__)

symptom_list = pickle.load(open("symptom_list.pkl", "rb"))

display_symptoms = [s.replace("_"," ") for s in symptom_list]
checkbox_symptoms = request.form.getlist("symptom_checkbox")

@app.route("/", methods=["GET", "POST"])
def home():

    predictions = None
    symptoms_input = ""

    if request.method == "POST":

        symptoms_input = request.form["symptoms"]
        symptoms = [s.strip() for s in symptoms_input.split(",")]

        symptoms.extend(checkbox_symptoms)

        predictions = predict_disease(symptoms)

    return render_template(
        "index.html",
        predictions=predictions,
        symptoms_input=symptoms_input,
        symptom_list=symptom_list,
        display_symptoms=display_symptoms
    )

if __name__ == "__main__":
    app.run(debug=True)
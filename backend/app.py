from flask import Flask, render_template, request
from predict import predict_disease
import pickle

app = Flask(__name__)

symptom_list = pickle.load(open("symptom_list.pkl", "rb"))
display_symptoms = [s.replace("_"," ") for s in symptom_list]

@app.route("/", methods=["GET", "POST"])
def home():
    predictions = None
    symptoms_input = ""

    if request.method == "POST":
        # รับค่าจาก Textbox
        s_input = request.form.get("symptoms", "")
        symptoms_text = [s.strip() for s in s_input.split(",") if s.strip()]

        # รับค่าจาก Checkbox
        checkbox_symptoms = request.form.getlist("symptom_checkbox")

        # รวมค่าและกำจัดตัวซ้ำ (Set)
        final_symptoms = list(set(symptoms_text + checkbox_symptoms))

        if final_symptoms:
            predictions = predict_disease(final_symptoms)
            symptoms_input = ", ".join(final_symptoms)

    return render_template(
        "index.html",
        predictions=predictions,
        symptoms_input=symptoms_input,
        symptom_list=symptom_list,
        display_symptoms=display_symptoms
    )

if __name__ == "__main__":
    app.run(debug=True)
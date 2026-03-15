from flask import Flask, request, render_template, jsonify
from predict import predict_disease

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def home():
    predictions = None
    symptoms_input = ""

    if request.method == "POST":
        symptoms_input = request.form["symptoms"]
        symptoms = [s.strip() for s in symptoms_input.split(",")]
        predictions = predict_disease(symptoms)
    return render_template(
        "index.html", predictions=predictions, symptoms_input=symptoms_input
    )

if __name__ == "__main__":
    app.run(debug=True)

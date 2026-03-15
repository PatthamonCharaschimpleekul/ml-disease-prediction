from flask import Flask, request, render_template, jsonify
from predict import predict_disease

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("frontend/index.html")

@app.route("/predict", methods=["POST"])

def predict():
    symptoms = request.json["symtoms"]
    predictions = predict_disease(symptoms)
    return (jsonify(predictions))

if __name__ == "__main__":
    app.run(debug=True)

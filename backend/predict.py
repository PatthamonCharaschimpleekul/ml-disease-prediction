from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd

# load model
model = pickle.load(open("model.pkl", "rb"))

# load symptom mapping
symptom_index = pickle.load(open("symptom_index.pkl", "rb"))

# ⭐ load symptom list (fix sklearn warning)
symptom_list = pickle.load(open("symptom_list.pkl", "rb"))
num_symptoms = len(symptom_index)

# load description and precaution
description_df = pd.read_csv("data/description.csv")
precaution_df = pd.read_csv("data/precaution.csv")

# create description dictionary
description_dict = dict(
    zip(description_df["disease"], description_df["description"])
)

# create precaution dictionary
precaution_dict = {}

for _, row in precaution_df.iterrows():
    precaution_dict[row["disease"]] = [
        row["precaution1"],
        row["precaution2"],
        row["precaution3"],
        row["precaution4"]
    ]

def predict_disease(input_symptoms):
    if not input_symptoms:
        return []

    # create vector
    input_vector = np.zeros(num_symptoms)
    for symptom in input_symptoms:
        symptom = symptom.strip().lower().replace(" ", "_")
        if symptom in symptom_index:
            idx = symptom_index[symptom]
            input_vector[idx] = 1
        else:
            print("Warning unknown symptom:", symptom)

    # convert to dataframe (แก้ sklearn warning)
    input_df = pd.DataFrame([input_vector], columns=symptom_list)

    # prediction
    probs = model.predict_proba(input_df)[0]
    classes = model.classes_
    results = list(zip(classes, probs))
    results.sort(key=lambda x: x[1], reverse=True)
    top3 = results[:3]
    output = []

    for disease, prob in top3:
        description = description_dict.get(disease, "No description available")
        precautions = precaution_dict.get(disease, [])
        output.append({
            "disease": disease,
            "confidence": float(prob),
            "description": description,
            "precautions": precautions
        })
    return output

# test
if __name__ == "__main__":
    symptoms = [
        "itching",
        "skin_rash"
    ]

    predictions = predict_disease(symptoms)
    if not predictions:
        print("No prediction returned")

    for p in predictions:
        print("\nDisease:", p["disease"])
        print("Confidence:", round(p["confidence"], 3))
        print("Description:", p["description"])
        print("Precautions:")

        for pr in p["precautions"]:
            print("-", pr)
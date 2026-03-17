import pickle
import numpy as np
import pandas as pd

# load assets
model = pickle.load(open("model.pkl", "rb"))
symptom_index = pickle.load(open("symptom_index.pkl", "rb"))
symptom_list = pickle.load(open("symptom_list.pkl", "rb"))
num_symptoms = len(symptom_index)

description_df = pd.read_csv("data/description.csv")
precaution_df = pd.read_csv("data/precaution.csv")

description_dict = dict(zip(description_df["disease"], description_df["description"]))
precaution_dict = {row["disease"]: [row["precaution1"], row["precaution2"], row["precaution3"], row["precaution4"]] 
                  for _, row in precaution_df.iterrows()}

def predict_disease(input_symptoms):
    if not input_symptoms:
        return []

    input_vector = np.zeros(num_symptoms)
    for symptom in input_symptoms:
        # แปลงชื่ออาการให้ตรงกับใน database
        symptom = symptom.strip().lower().replace(" ", "_")
        if symptom in symptom_index:
            idx = symptom_index[symptom]
            input_vector[idx] = 1

    # แปลงเป็น DataFrame พร้อมชื่อคอลัมน์ เพื่อป้องกัน Warning
    input_df = pd.DataFrame([input_vector], columns=symptom_list)

    probs = model.predict_proba(input_df)[0]
    classes = model.classes_
    results = sorted(list(zip(classes, probs)), key=lambda x: x[1], reverse=True)
    
    top3 = results[:3]
    output = []

    for disease, prob in top3:
        output.append({
            "disease": disease,
            "confidence": float(prob),
            "description": description_dict.get(disease, "No description available"),
            "precautions": precaution_dict.get(disease, [])
        })
    return output
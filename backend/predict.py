import pickle
import numpy as np

#download model
model = pickle.load(open("model.pkl", "rb"))

#download symptom list
symptom_index = pickle.load(open("symptom_index.pkl","rb"))

#number of symptom 
num_symptoms = len(symptom_index)

#create predict function
def predict_disease(input_symptoms):
    #create vectior 0
    input_vector = np.zeros(num_symptoms)

    #map symptoms to vector
    for symptom in input_symptoms:
        #clean input
        symptom = symptom.strip().lower().replace(" ", "_")
        if symptom in symptom_index:
            idx = symptom_index[symptom]
            input_vector[idx] = 1
        else:
            print("Warning unknown symptom: ", symptom)
    #prediction
    probs = model.predict_proba([input_vector])[0]
    classes = model.classes_
    results = list(zip(classes, probs))
    results.sort(key=lambda x: x[1], reverse=True)
    top3 = results[:3]
    return top3

#test & run
if __name__ == "__main__":
    symptoms = [
        "itching",
        "skin_rash"
    ]
    predictions = predict_disease(symptoms)
    if not predictions:
        print("No prediction returned")
    print("\nTop Predictions:")
    for disease, prob in predictions:
        print(f"{disease} : {prob:.3f}")
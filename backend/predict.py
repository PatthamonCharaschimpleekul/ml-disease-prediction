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
        if symptom in symptom_index:
            idx = symptom_index[symptom]
            input_vector[idx] = 1
        else:
            print("Warning unknown symptom: ", symptom)
    #prediction
    prediction = model.predict([input_vector])[0]
    confidence = model.predict_proba([input_vector]).max()
    return prediction, confidence

#test & run
if __name__ == "__main__":
    symptoms = ["itching", "skin_rash"]
    desease, conf = predict_disease(symptoms)
    print("Prediction: ", desease)
    print("Confidence: ", conf)
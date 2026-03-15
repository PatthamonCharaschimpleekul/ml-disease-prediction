import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#download Dataset
data = pd.read_csv('data/Training.csv')

#data observes
'''print("\n--------DATASET INFO--------")
print(data.info)
print("dataset shape: ", data.shape)
print(data.head)

missing = data.isnull().sum()
print(missing[missing > 0])
data = data.dropna()

duplicate_rows = data.duplicated().sum()
print("Duplicate rows:", duplicate_rows)
data = data.drop_duplicates()

print("\n----- DATA TYPES -----")
print(data.dtypes)

print("\n------ CHECK INVALID VALUES -----")
symptom_columns = data.columns[:-1]
for col in symptom_columns:
    unique_vals = data[col].unique()
    if not set(unique_vals).issubset({0,1}):
        print(f"Column {col} has invalid values:", unique_vals)

print("\n----- UNIQUE PROGNOSIS NAMES ------")
print(data["prognosis"].unique())'''

#features&tagets
X = data.drop("prognosis", axis=1)
y = data["prognosis"]

symptom_list = list(X.columns)

#create mapping ddictionary
symptom_index = {symptom: idx for idx, symptom in enumerate(symptom_list)}

#split model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=69, stratify=y
)

#train nodel
model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=20,
    min_samples_split=5,
    random_state=69
)

model.fit(X_train, y_train)

#evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy = ", accuracy)

#save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(symptom_list, open("symptom_list.pkl", "wb"))
pickle.dump(symptom_index, open("symptom_index.pkl", "wb"))
print("\nModel and mapping saved successfully")
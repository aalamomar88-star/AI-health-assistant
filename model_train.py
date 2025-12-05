#model.train
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import itertools
import random
import json

#symptoms we'll support
SYMPTOMS=[
    "fever","cough","sore_throat","runny_nose","fatigue",
    "body_ache","headache","shortness_of_breath","loss_of_taste_smell",
    "chills","sweating","nausea","vomiting","diarrhea","abdominal_pain",
    "high_fever","chest_pain","bleeding"
]
#Diseases and typical symptom patterns (simple sunthetic rules)
DISEASES={
    "Common Cold":["runny_nose","sore_throat","cough","headache"],
    "Flu (Influenza)":["fever","body_ache","fatigue","cough","headache","chills"],
    "COVID-19":["fever","cough","loss_of_taste_smell","fatigue","shortness_of_breath"],
    "Malaria":["fever","chills","sweating","headache","high_fever"],
    "Gastroenteritis":["nausea","vomiting","diarrhea","abdominal_pain","fever"]
}
#generate synthetic examples
rows=[]
labels=[]
for disease,typical in DISEASES.items():
    for _ in range(300): #generate many samples per disease
        #start with typical symptoms
        present=set()
        for s in SYMPTOMS:
            if s in typical:
                #typical symptom present with high prob
                if random.random()<0.85:
                    present.add(s)
                else:
                    #maybe some noise
                    if random.random()<0.06:
                        present.add(s)
        #occasionally missing a typical symptom
        if len(typical)>0 and random.random()<0.05:
            present.discard(random.choice(typical))
        row=[1 if s in present else 0 for s in SYMPTOMS]
        rows.append(row)
        labels.append(disease)

#build DataFrame
df=pd.DataFrame(rows,columns=SYMPTOMS)
df['disease']=labels

#train/test split
X=df[SYMPTOMS].values
y=df['disease'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.12,random_state=42)

clf=RandomForestClassifier(n_estimators=150,random_state=42)
clf.fit(X_train,y_train)

print("Train score:",clf.score(X_train,y_train))
print("test score:",clf.score(X_test,y_test))

#save model and symptons list
dump(clf,"model.joblib")
with open("symptoms_list.json","w") as f:
    json.dump(SYMPTOMS,f)

print("Saved model.joblib and symptoms_list.json")    

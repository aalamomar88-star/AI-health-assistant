#app.py
from flask import Flask,render_templates,request,jsonify
from joblib import load
import json
import numpy as np
from disease_info import disease_info

app=Flask(__name__)

#load model and symptom list
clf=load("model.joblib")
with open("symptoms_list.json","r") as f:
    SYMPTOMS=json.load(f)

#helper:convert symptom names to vector
def symptoms_to_vector(symptom_list):
    vec=[1 if s in symptom_list else 0 for s in SYMPTOMS]
    return np.array(vec).reshape(1,-1)

@app.route("/")
def index():
    return render_template("index.html",symptoms=SYMPTOMS)

@app.route("/predict",methods=["POST"])
def predict():
    data=request.json
    #data expected: { "symptoms":["fever","cough"],"text":"..." (optional)}
    selected=set(data.get("symptoms",[]))

    #for robustness,also parse 'text_extracted' if present (client does basic NLP)
    text_extracted=data.get("text_extracted",[])
    for t in text_extracted:
        selected.add(t)

    #create vector
    if len(selected)==0:
        #if nothing selected,return empty guidence
        return jsonify({"error":"No symptoms provided.please select or type symptoms."}),400

    vec=symptoms_to_vector(selected)
    probs=clf.predict_proba(vec)[0]
    classes=clf.classes_

    #build top 3
    top_idx=np.argsort(probs)[::-1][:3]
    results=[]
    for idx in top_idx:
        disease=classes[idx]
        conf=float(probs[idx])
        info=disease_info.get(disease,{})
        results.append({
            "disease":disease,
            "confidence":round(conf,3),
            "description":info.get("description",""),
            "causes":info.get("causes",""),
            "when_to_see_doctor":info.get("when_to_see_doctor",""),
            "home_remedy":info.get("home_remedy","")

        })
    # risk level heuristic:high if top confidence>0.7 or disease in critical list
    critical=["Malaria","COVID-19"]
    top_disease=results[0]["disease"]
    top_conf=results[0]["confidence"]
    if top_conf>0.7 or top_disease in critical:
        risk="High"
    elif top_conf>0.4:
        risk="Medium"
    else:
        risk="Low"

    return jsonify({
        "predictions":results,
        "risk_level":risk,
        "selected_symptoms":list(selected)
    })
if __name__=="__main__":
    app.run(debug=True)
                        

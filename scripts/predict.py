import joblib
import pandas as pd
import numpy as np

model = joblib.load("../model/fixie_model.pkl")
labels = joblib.load("../model/labels.pkl")

data = pd.read_csv("../data/new_tickets.csv")

results = []
for desc in data["Customer_Description"]:
    prob = model.predict_proba([desc])[0]
    pred = model.predict([desc])[0]
    top10 = sorted(zip(labels, prob), key=lambda x: x[1], reverse=True)[:10]
    results.append({
        "description": desc,
        "predicted_fix": pred,
        "confidence": round(np.max(prob), 4),
        "top_10_fixes": top10
    })

pd.DataFrame(results).to_csv("../data/output.csv", index=False)

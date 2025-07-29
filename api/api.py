from flask import Flask, request, jsonify
import joblib
import traceback
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("../model/fixie_model.pkl")
labels = joblib.load("../model/labels.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        descriptions = data.get("descriptions", [])
        if not descriptions:
            return jsonify({"error": "Missing 'descriptions' key or value is empty"}), 400

        probs = model.predict_proba(descriptions)
        predictions = model.predict(descriptions)

        result = []
        for i, desc in enumerate(descriptions):
            conf = round(np.max(probs[i]), 4)
            ranked = sorted(zip(labels, probs[i]), key=lambda x: x[1], reverse=True)[:10]
            result.append({
                "description": desc,
                "predicted_fix": predictions[i],
                "confidence": conf,
                "top_10_fixes": [{"label": lbl, "score": round(score, 4)} for lbl, score in ranked],
                "predicted_module": "AutoDetectModule"
            })

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/")
def home():
    return "FixieBot API is running. Use POST /predict"

if __name__ == "__main__":
    app.run(debug=True)

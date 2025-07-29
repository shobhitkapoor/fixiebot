from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib
import numpy as np
import traceback

app = Flask(__name__)

model_path = "../model/bert_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

labels = joblib.load("../model/labels.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        descriptions = data.get("descriptions", [])
        if not descriptions:
            return jsonify({"error": "Missing 'descriptions' key or value is empty"}), 400

        result = []
        for desc in descriptions:
            inputs = tokenizer(desc, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy().flatten()
            pred_idx = int(np.argmax(probs))
            conf = round(probs[pred_idx], 4)
            ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)[:10]

            result.append({
                "description": desc,
                "predicted_fix": labels[pred_idx],
                "confidence": conf,
                "top_10_fixes": [{"label": lbl, "score": round(score, 4)} for lbl, score in ranked],
                "predicted_module": "AutoDetectModule"
            })

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(debug=True)

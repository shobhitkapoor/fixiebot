# 🤖 FixieBot – Predictive Fix Recommendation Engine

FixieBot is an AI-powered tool that predicts the right fix for customer support tickets based on historical ticket data using NLP (TF-IDF + Logistic Regression) and BERT models.

## 🔧 Features

- Predict Fixes from customer ticket descriptions
- Auto-detect likely affected modules
- Confidence scoring and Top-10 recommendations
- Feedback loop for model improvement
- Streamlit-based UI dashboard
- REST API (Flask) for prediction and feedback
- JIRA integration and cluster visualizations
- BERT-based advanced NLP model (optional)

## 🗂️ Folder Structure

```
FixieBot/
├── api/
│   ├── app.py               # REST API (Classic TF-IDF model)
│   └── app_bert.py          # REST API (BERT model)
├── scripts/
│   ├── train.py             # Train classic TF-IDF model
│   ├── train_bert_fix.py    # Train BERT model
│   ├── predict.py           # Predict from CSV (classic)
│   ├── feedback_loop.py     # Update model with new feedback
├── data/
│   ├── historical_tickets.csv
│   └── new_tickets.csv
├── output/
│   └── output_predictions.csv
├── integrations/
│   └── jira_webhook.py
├── reports/
│   └── cluster_visualization.py
├── streamlit_ui/
│   └── app.py               # Streamlit frontend UI
├── model/
│   └── fixie_model.pkl
├── Makefile
├── Dockerfile
└── README.md
```

## 🚀 Quickstart

### 1. 🔧 Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. 🔁 Train Model (Classic)

```bash
make train
```

### 3. 🧠 Train BERT Model (Optional)

```bash
make train_bert
```

### 4. 🌐 Run API (Classic)

```bash
make run
```

Visit [http://localhost:5000](http://localhost:5000)

### 5. 🎛 Streamlit Dashboard

```bash
make ui
```

## 📈 Prediction Output Example

- `output_predictions.csv` will contain:

| Ticket ID | Description | Predicted Fix | Confidence | Top 10 Fixes | Module |
|-----------|-------------|---------------|------------|--------------|--------|

## 🔁 Feedback Loop

To update the model with corrections:

```bash
make feedback
```

## 🧪 Sample API Request

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" \
  -d '{"descriptions": ["Customer reports issue with login delay."]}'
```

## 📦 Docker (Optional)

```bash
docker build -t fixiebot .
docker run -p 5000:5000 fixiebot
```

## 🧪 Requirements

- Python 3.8+
- Flask
- scikit-learn
- pandas, numpy
- joblib
- transformers (for BERT)
- Streamlit

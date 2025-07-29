import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load historical tickets dataset
df = pd.read_csv("../data/historical_tickets.csv")

X = df["Customer_Description"]
y = df["Fix_Applied"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=500))
])

pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "../model/fixie_model.pkl")
labels = sorted(y.unique().tolist())
joblib.dump(labels, "../model/labels.pkl")

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

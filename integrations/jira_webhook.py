from flask import Flask, request, jsonify
import pandas as pd
import uuid
import os

app = Flask(__name__)
FEEDBACK_PATH = "../data/feedback.csv"

@app.route("/jira_webhook", methods=["POST"])
def webhook():
    data = request.json
    ticket = {
        "Ticket_ID": str(uuid.uuid4())[:8],
        "Customer_Description": data.get("description", ""),
        "Product": data.get("product", "Unknown"),
        "Fix_Applied": data.get("fix", ""),
        "Resolution_Time": data.get("time", 0),
        "Tags": data.get("tags", "")
    }

    df = pd.DataFrame([ticket])
    if os.path.exists(FEEDBACK_PATH):
        df.to_csv(FEEDBACK_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(FEEDBACK_PATH, index=False)

    return jsonify({"status": "Feedback recorded"})

if __name__ == "__main__":
    app.run(debug=True, port=5005)

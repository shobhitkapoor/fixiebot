import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="FixieBot UI", layout="wide")
st.title("🤖 FixieBot - Predictive Fix Recommendation")

with st.sidebar:
    st.header("🛠 Ticket Input")
    description = st.text_area("Customer Issue Description", height=200)
    model_type = st.radio("Model Type", ["TF-IDF Model", "BERT Model"])
    predict_btn = st.button("🔍 Predict Fix")

if predict_btn and description:
    with st.spinner("Predicting..."):
        api_url = "http://localhost:5000/predict" if model_type == "TF-IDF Model" else "http://localhost:5000/bert"
        try:
            response = requests.post(api_url, json={"descriptions": [description]})
            if response.status_code == 200:
                data = response.json()[0]

                st.subheader("📍 Prediction Results")
                st.write(f"**Predicted Fix:** `{data['predicted_fix']}`")
                st.write(f"**Predicted Module:** `{data['predicted_module']}`")
                st.write(f"**Confidence Score:** `{data['confidence']}`")

                st.markdown("### 🎯 Top 10 Fix Recommendations")
                top_df = pd.DataFrame(data["top_10_fixes"])
                st.table(top_df)

                st.markdown("### 📥 Submit Feedback")
                fix_applied = st.selectbox("Was this prediction correct? If not, select the correct fix:", [data['predicted_fix']] + [f["label"] for f in data["top_10_fixes"]])
                submit_feedback = st.button("✅ Submit Feedback")
                if submit_feedback:
                    feedback = {
                        "Ticket_ID": "UISTREAM001",
                        "Customer_Description": description,
                        "Product": data["predicted_module"],
                        "Fix_Applied": fix_applied,
                        "Resolution_Time": 0,
                        "Tags": "auto_ui"
                    }
                    requests.post("http://localhost:5005/jira_webhook", json=feedback)
                    st.success("Feedback submitted and added to feedback.csv")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")
else:
    st.info("📝 Enter a description and click Predict Fix.")

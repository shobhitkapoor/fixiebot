import pandas as pd

# Load feedback data
feedback = pd.read_csv("../data/feedback.csv")
historical = pd.read_csv("../data/historical_tickets.csv")

# Append feedback to historical
df_combined = pd.concat([historical, feedback], ignore_index=True)
df_combined.to_csv("../data/historical_tickets.csv", index=False)

print("✅ Feedback added and ready for retraining")

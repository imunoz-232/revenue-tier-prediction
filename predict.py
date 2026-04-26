import pandas as pd
import joblib

# -----------------------------
# LOAD MODEL + COLUMNS
# -----------------------------
saved = joblib.load("model/revenue_model.pkl")

model = saved["model"]
training_columns = saved["columns"]

# -----------------------------
# NEW TRANSACTION
# -----------------------------
new_transaction = pd.DataFrame([{
    "product_category": "Electronics",
    "customer_region": "North",
    "discount_percent": 10,
    "payment_method": "Credit Card",
    "rating": 4.5,
    "review_count": 120,
    "month": 4
}])

# -----------------------------
# ENCODE
# -----------------------------
new_encoded = pd.get_dummies(new_transaction, drop_first=True)

# -----------------------------
# MATCH TRAINING COLUMNS
# -----------------------------
for col in training_columns:
    if col not in new_encoded.columns:
        new_encoded[col] = 0

new_encoded = new_encoded[training_columns]

# -----------------------------
# PREDICT
# -----------------------------
prediction = model.predict(new_encoded)

print("Predicted Revenue Tier:", prediction[0])
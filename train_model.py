import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("data/amazon_sales.csv")

print("Dataset loaded successfully")
print("Columns:", df.columns.tolist())

# -----------------------------
# 2. CLEAN DATA
# -----------------------------
df = df.drop_duplicates()
df = df.dropna()

# -----------------------------
# 3. DATE FEATURE
# -----------------------------
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df = df.dropna(subset=["order_date"])
df["month"] = df["order_date"].dt.month

# -----------------------------
# 4. CREATE TARGET
# -----------------------------
low = df["total_revenue"].quantile(0.33)
high = df["total_revenue"].quantile(0.66)

df["revenue_tier"] = pd.cut(
    df["total_revenue"],
    bins=[-float("inf"), low, high, float("inf")],
    labels=["Low", "Medium", "High"]
)

print("\nRevenue tier distribution:")
print(df["revenue_tier"].value_counts())

# -----------------------------
# 5. SELECT FEATURES (NO LEAKAGE)
# -----------------------------
features = [
    "product_category",
    "customer_region",
    "discount_percent",
    "payment_method",
    "rating",
    "review_count",
    "month"
]

X = df[features]
y = df["revenue_tier"]

# -----------------------------
# 6. ENCODE
# -----------------------------
X_encoded = pd.get_dummies(X, drop_first=True)

# -----------------------------
# 7. SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 8. MODEL
# -----------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ))
])

# -----------------------------
# 9. TRAIN
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 10. EVALUATE
# -----------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 11. SAVE MODEL + COLUMNS
# -----------------------------
joblib.dump({
    "model": model,
    "columns": X_encoded.columns.tolist()
}, "model/revenue_model.pkl")

print("\nModel trained and saved successfully.")
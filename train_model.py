"""
Revenue Tier Prediction - Model Training Script
Trains a Logistic Regression model for revenue tier classification.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

print("=" * 60)
print("Revenue Tier Prediction - Model Training")
print("=" * 60)

data_path = os.path.join("data", "clean_amazon_sales.csv")
print("\n[1/6] Loading dataset...")
df = pd.read_csv(data_path)
print(f"✓ Loaded {len(df)} records from {data_path}")

print("\n[2/6] Cleaning data...")
df = df.drop_duplicates()
df = df.dropna()
print(f"✓ Cleaned data: {len(df)} records remain")

print("\n[3/6] Preparing target variable...")
if "revenue_tier" in df.columns:
    df["revenue_tier"] = df["revenue_tier"].astype(str)
    print("✓ Using existing revenue_tier column")
elif "total_revenue" in df.columns:
    low = df["total_revenue"].quantile(0.33)
    high = df["total_revenue"].quantile(0.66)
    df["revenue_tier"] = pd.cut(
        df["total_revenue"],
        bins=[-float("inf"), low, high, float("inf")],
        labels=["Low", "Medium", "High"]
    )
    print("✓ Created revenue_tier from total_revenue percentiles")
else:
    raise ValueError("Dataset must include either 'revenue_tier' or 'total_revenue' column.")

print("✓ Revenue Tier Distribution:")
print(df["revenue_tier"].value_counts())

features = [
    "product_category",
    "customer_region",
    "discount_percent",
    "payment_method",
    "rating",
    "review_count",
    "month"
]

print("\n[4/6] Preparing features...")
missing_columns = [col for col in features if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

X = df[features].copy()
y = df["revenue_tier"].copy()

X_encoded = pd.get_dummies(X, drop_first=False)
print(f"✓ Features shape: {X_encoded.shape}")
print(f"✓ Feature columns: {len(X_encoded.columns)}")

print("\n[5/6] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"✓ Training records: {len(X_train)}")
print(f"✓ Test records: {len(X_test)}")

print("\n[6/6] Training Logistic Regression model...")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)
print("✓ Model trained successfully")

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
print(f"\n✓ Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"✓ Macro F1 Score: {macro_f1:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)
os.makedirs("model", exist_ok=True)
model_data = {
    "model": model,
    "columns": X_encoded.columns.tolist(),
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "features": features,
}
joblib.dump(model_data, os.path.join("model", "revenue_model.pkl"))
print("✓ Model saved to model/revenue_model.pkl")
print("\n" + "=" * 60)
print("✓ TRAINING COMPLETE!")
print("=" * 60)
print("\nNext step: run 'streamlit run app.py' to open the application.")

"""
Revenue Tier Prediction - Streamlit Application
Simple, beginner-friendly interface for revenue tier classification.
"""

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(
    page_title="Revenue Tier Prediction",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("model/revenue_model.pkl")

try:
    model_data = load_model()
    model = model_data["model"]
    feature_columns = model_data["columns"]
    accuracy = model_data.get("accuracy")
    macro_f1 = model_data.get("macro_f1")
except Exception:
    st.error("Unable to load the trained model. Please run 'python train_model.py' first.")
    st.stop()

st.title("🚀 Revenue Tier Prediction")
st.markdown("Predict whether a transaction belongs to Low, Medium, or High revenue tier using simple product and customer features.")
st.markdown("---")

page = st.sidebar.radio("Navigation", ["Dashboard", "Single Prediction", "Batch Analysis"])

required_cols = [
    "product_category",
    "customer_region",
    "discount_percent",
    "payment_method",
    "rating",
    "review_count",
    "month"
]


def prepare_features(df):
    df_encoded = pd.get_dummies(df)
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_columns]
    return df_encoded


if page == "Dashboard":
    st.header("📊 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}" if accuracy is not None else "N/A")
    with col2:
        st.metric("Macro F1 Score", f"{macro_f1:.4f}" if macro_f1 is not None else "N/A")

    st.markdown("---")
    st.subheader("💡 Business Insights")
    st.write("- High revenue transactions tend to have lower discounts.")
    st.write("- High revenue transactions tend to have higher ratings.")
    st.write("- High revenue products tend to have more customer engagement.")

    st.markdown("---")
    st.subheader("🎯 Business Recommendations")
    st.write("1. Increase inventory for top High revenue categories.")
    st.write("2. Reduce discounts on high-performing categories.")
    st.write("3. Improve strategy for low-performing categories.")
    st.write("4. Focus marketing on high-performing regions.")
    st.write("5. Monitor ratings and customer review activity.")

elif page == "Single Prediction":
    st.header("🎯 Single Prediction")
    st.write("Enter the transaction details below to predict the revenue tier.")

    with st.form("single_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            product_category = st.selectbox(
                "Product Category",
                [
                    "Electronics",
                    "Books",
                    "Fashion",
                    "Sports",
                    "Beauty",
                    "Home & Kitchen",
                    "Other"
                ]
            )
            customer_region = st.selectbox(
                "Customer Region",
                [
                    "North America",
                    "Europe",
                    "Asia",
                    "Middle East",
                    "North",
                    "South",
                    "East",
                    "West"
                ]
            )
            payment_method = st.selectbox(
                "Payment Method",
                ["Credit Card", "Debit Card", "UPI", "Wallet", "Cash on Delivery"]
            )
            month = st.selectbox("Month", list(range(1, 13)), index=0)

        with col2:
            discount_percent = st.slider("Discount (%)", min_value=0, max_value=100, value=15, step=1)
            rating = st.slider("Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
            review_count = st.number_input("Review Count", min_value=0, max_value=1000, value=100, step=1)
            st.write("\n")

        submit = st.form_submit_button("Predict Revenue Tier")

    if submit:
        input_data = pd.DataFrame([{
            "product_category": product_category,
            "customer_region": customer_region,
            "discount_percent": discount_percent,
            "payment_method": payment_method,
            "rating": rating,
            "review_count": review_count,
            "month": month,
        }])

        try:
            input_features = prepare_features(input_data)
            prediction = model.predict(input_features)[0]
            probabilities = model.predict_proba(input_features)[0]
            labels = model.classes_

            st.markdown("---")
            if prediction == "High":
                st.success(f"### {prediction} Revenue Tier")
                st.write("This transaction is likely to belong to the high revenue tier.")
            elif prediction == "Medium":
                st.info(f"### {prediction} Revenue Tier")
                st.write("This transaction is likely to belong to the medium revenue tier.")
            else:
                st.warning(f"### {prediction} Revenue Tier")
                st.write("This transaction is likely to belong to the low revenue tier.")

            confidence = pd.DataFrame({
                "Revenue Tier": labels,
                "Confidence": [f"{v:.1%}" for v in probabilities]
            })
            st.markdown("---")
            st.subheader("Prediction Confidence")
            st.dataframe(confidence, use_container_width=True)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

elif page == "Batch Analysis":
    st.header("📁 Batch Analysis")
    st.write("Upload a CSV file with one row per transaction to run batch prediction.")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="The file must include product_category, customer_region, discount_percent, payment_method, rating, review_count, month"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip().str.lower()
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                st.error(f"CSV is missing these columns: {', '.join(missing)}")
            else:
                st.success(f"Loaded {len(df)} rows")
                with st.expander("View sample data"):
                    st.dataframe(df.head(), use_container_width=True)

                df_features = df[required_cols].copy()
                predictions = model.predict(prepare_features(df_features))
                df["revenue_tier"] = predictions

                st.markdown("---")
                st.subheader("Prediction Distribution")
                summary = df["revenue_tier"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
                st.bar_chart(summary)

                st.markdown("---")
                st.subheader("Top Categories")
                top_high = df.loc[df["revenue_tier"] == "High", "product_category"].mode()
                top_low = df.loc[df["revenue_tier"] == "Low", "product_category"].mode()
                st.write(f"**Top High Revenue Category:** {top_high.iloc[0] if not top_high.empty else 'N/A'}")
                st.write(f"**Top Low Revenue Category:** {top_low.iloc[0] if not top_low.empty else 'N/A'}")

                st.markdown("---")
                st.subheader("Business Insights")
                avg_discount_high = df.loc[df["revenue_tier"] == "High", "discount_percent"].mean()
                avg_rating_high = df.loc[df["revenue_tier"] == "High", "rating"].mean()
                avg_reviews_high = df.loc[df["revenue_tier"] == "High", "review_count"].mean()
                st.write(f"- High revenue rows have lower average discount: **{avg_discount_high:.1f}%**")
                st.write(f"- High revenue rows have higher average rating: **{avg_rating_high:.2f}**")
                st.write(f"- High revenue rows have more customer reviews: **{avg_reviews_high:.0f}**")

                st.markdown("---")
                st.subheader("Recommendations")
                st.write(f"1. Increase inventory for {top_high.iloc[0] if not top_high.empty else 'top high-performing categories'}.")
                st.write("2. Reduce discounts on high-performing categories.")
                st.write(f"3. Improve strategy for {top_low.iloc[0] if not top_low.empty else 'low-performing categories'}.")
                st.write("4. Focus marketing on high-performing regions.")

                st.markdown("---")
                st.subheader("Download Predictions")
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download prediction results",
                    data=csv_data,
                    file_name=f"revenue_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

                with st.expander("View full results table"):
                    st.dataframe(df, use_container_width=True)
        except Exception as exc:
            st.error(f"Failed to read the file: {exc}")

st.markdown("---")
st.write("Built with Streamlit and scikit-learn. Use the Batch Analysis section to upload your own data.")

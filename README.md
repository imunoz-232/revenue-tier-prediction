# Revenue Tier Prediction for Transaction Segmentation

A beginner-friendly machine learning project that predicts whether a transaction belongs to **Low**, **Medium**, or **High** revenue tier.

## Quick Start (5 Minutes)

### Step 1: Clone the Repository
```powershell
git clone https://github.com/ingridmunoz/revenue-tier-prediction.git
cd revenue-tier-prediction
```

### Step 2: Install Python Packages
```powershell
pip install -r requirements.txt
```

### Step 3: Train the Model (First Time Only)
```powershell
python train_model.py
```

Expected output:
```
============================================================
Revenue Tier Prediction - Model Training
============================================================

[1/6] Loading dataset...
✓ Loaded X records from data/clean_amazon_sales.csv
...
============================================================
✓ TRAINING COMPLETE!
============================================================
```

### Step 4: Run the Streamlit App
```powershell
streamlit run app.py
```

### Step 5: Open in Browser
A browser window will automatically open at:
```
http://localhost:8501
```

---

## App Features

### 📊 Dashboard
- View model accuracy and macro F1 score
- See key business insights from the data
- Get automatic business recommendations

### 🎯 Single Prediction
1. Enter transaction details (category, region, discount, rating, etc.)
2. Click "Predict Revenue Tier"
3. Get instant prediction with confidence scores

**Example:**
```
Product Category: Electronics
Customer Region: North America
Discount: 15%
Rating: 4.5
Review Count: 250
Result: HIGH Revenue Tier ✅
```

### 📁 Batch Analysis
1. Upload a CSV file with multiple transactions
2. App automatically predicts revenue tier for all rows
3. View prediction distribution, top categories, insights
4. Download predictions as CSV

**Required CSV Columns:**
```
product_category, customer_region, discount_percent, payment_method, rating, review_count, month
```

**Example CSV:**
```
Electronics,North America,10,Credit Card,4.5,150,3
Books,Europe,20,Debit Card,3.8,95,5
Fashion,Asia,15,Wallet,4.2,200,7
```

---

## Project Structure

```
revenue-tier-prediction/
├── app.py                      ← Streamlit application (RUN THIS)
├── train_model.py              ← Model training script
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
├── data/
│   ├── amazon_sales.csv        ← Raw data
│   └── clean_amazon_sales.csv  ← Cleaned data used for training
└── model/
    └── revenue_model.pkl       ← Trained model (created after running train_model.py)
```

---

## Model Information

**Algorithm:** Logistic Regression (Multinomial Classification)

**Features Used (7 total):**
- product_category
- customer_region
- discount_percent
- payment_method
- rating
- review_count
- month

**Target Variable:** revenue_tier (Low / Medium / High)

**Evaluation Metrics:**
- Accuracy
- Macro F1 Score
- Confusion Matrix

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:** Make sure you installed all packages:
```powershell
pip install -r requirements.txt
```

### Issue: "FileNotFoundError: model/revenue_model.pkl"
**Solution:** The model file doesn't exist. Train it first:
```powershell
python train_model.py
```

### Issue: Browser doesn't open automatically
**Solution:** Manually open:
```
http://localhost:8501
```

### Issue: CSV upload fails
**Solution:** Make sure your CSV has these exact columns:
```
product_category, customer_region, discount_percent, payment_method, rating, review_count, month
```

---

## For Professors & Educators

This project demonstrates:

✅ Machine learning workflow (data → model → prediction)  
✅ Multi-class classification (predicting 3 categories)  
✅ Data preprocessing and feature encoding  
✅ Model training and evaluation  
✅ Web application with Streamlit  
✅ Business application of machine learning  

Perfect for:
- Introduction to Machine Learning courses
- Data Science projects
- Business Analytics demonstrations
- Python for Data Science courses

---

## Commands Reference

| Command | Purpose |
|---------|---------|
| `pip install -r requirements.txt` | Install all required packages |
| `python train_model.py` | Train the machine learning model |
| `streamlit run app.py` | Start the web application |

---

## Data Source

**Dataset:** Amazon Sales Data  
**Records:** ~49,000 transactions  
**Features:** 7 input features + 1 target variable  
**Target:** Revenue Tier (Low / Medium / High)

---

## FAQ

**Q: Do I need to modify the code?**  
A: No! Just run the commands exactly as shown above.

**Q: What if I don't have Python installed?**  
A: Download Python 3.8 or newer from https://www.python.org/

**Q: Can I use my own data?**  
A: Yes! Prepare a CSV with the same columns and use the Batch Analysis section to upload it.

**Q: Is this production-ready?**  
A: This is an educational project. For production use, add proper error handling, logging, and security measures.

---

## Next Steps

1. ✅ Run the app (you're done!)
2. 🔍 Explore different predictions in the Single Prediction section
3. 📊 Upload your own data in the Batch Analysis section
4. 📈 Try different ML models to improve accuracy
5. 🚀 Deploy online using Heroku, AWS, or Google Cloud

---

**Questions?** Check the troubleshooting section or review the code comments.

**Ready to start?**
```powershell
streamlit run app.py
```

Happy learning! 🚀

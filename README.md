# 🛋️ E-commerce Furniture Sales Prediction

This project predicts the number of furniture items sold on an e-commerce platform using product attributes such as **price, discount percentage, product title, and shipping tags**.

## 🚀 Features
- Full ML pipeline (data cleaning → EDA → feature engineering → modeling → evaluation)
- Models: Linear Regression, Ridge, Random Forest
- Word report generated with plots and metrics
- Best model saved (`rf_sales_model.pkl`)
- Prediction function for new products

## 📂 Project Structure
```
Ecommerce-Furniture-ML-Project/
├── data/             # Dataset
├── src/              # Source code
├── outputs/          # Reports & models
├── notebooks/        # (optional Jupyter)
```
## ⚙️ Usage
```bash
pip install -r requirements.txt
python src/maincode.py
```

## 📈 Output
- `furniture_sales_report.docx` (with tables + plots)
- `rf_sales_model.pkl` (saved model)

# 📉 Customer Churn Prediction using Machine Learning

Predicting customer churn for a telecom company using Machine Learning. Built with Logistic Regression and Random Forest, deployed live as an interactive web app!

🚀 **[Try the Live Demo](https://customer-churn-predictor-sk.streamlit.app)**

---

## 🎯 Business Problem
Customer churn directly impacts revenue and long-term business sustainability. This project predicts customers at high risk of churn and identifies key factors influencing churn behavior to support data-driven retention strategies.

---

## 🛠️ Tech Stack
- Python (Pandas, NumPy)
- Data Visualization (Matplotlib, Seaborn)
- Machine Learning (Scikit-learn)
- Streamlit (Live Deployment)
- Models: Logistic Regression, Random Forest

---

## 🧾 Dataset
- Source: [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Records: ~7,000 customers
- Target Variable: Churn (Yes / No)

---

## 📂 Project Structure
```
├── notebooks/                          — EDA & model training notebook
├── app.py                              — Streamlit web app
├── churn_model.pkl                     — Trained Logistic Regression model
├── scaler.pkl                          — StandardScaler for feature scaling
└── requirements.txt                    — Dependencies
```

---

## 🚀 Project Workflow
1. Data cleaning and preprocessing
2. Exploratory data analysis (EDA)
3. Feature engineering (tenure groups, charge flags)
4. Model training and comparison
5. Model evaluation and interpretation
6. Business insight generation
7. Live deployment with Streamlit

---

## 📊 Model Results
| Model | Accuracy | ROC-AUC | Recall (Churn) |
|-------|----------|---------|----------------|
| Logistic Regression | 80% | 0.842 | 56% |
| Random Forest | 79% | 0.822 | 49% |

✅ **Logistic Regression selected** — better recall for identifying churners

---

## 🔍 Key Insights
- Month-to-month contract customers show **42.71% churn rate** vs 2.83% for two-year contracts
- Short-tenure customers are significantly more likely to churn
- Higher monthly charges increase churn probability
- Electronic check payment method correlates with higher churn

---

## 💡 Business Recommendations
- Incentivize long-term contracts for high-risk customers
- Offer personalized discounts to high-charge customers
- Improve onboarding experience for first-year customers
- Target electronic check users with auto-payment incentives

---

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📩 Connect with Me
- 📧 Email: santoshkumar729629@gmail.com
- 💼 LinkedIn: [linkedin.com/in/santosh-kumar-sk](https://www.linkedin.com/in/santosh-kumar-sk)
- 🐙 GitHub: [github.com/SantoshKumar902](https://github.com/SantoshKumar902)

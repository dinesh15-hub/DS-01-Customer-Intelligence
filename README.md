
# DS-01: Customer Lifetime Value and Churn Intelligence System

An end-to-end customer analytics project designed to analyze churn behavior, estimate customer lifetime value, understand retention patterns, and evaluate retention campaign scenarios.

## Project Overview

This project uses the IBM Telco Customer Churn dataset to build a customer intelligence system.

## Technologies Used

- Python
- Pandas and NumPy
- PostgreSQL
- LightGBM
- SHAP
- Lifelines
- Streamlit
- Scikit-learn

## Key Features

- Customer churn analysis and prediction
- LightGBM classification
- SHAP feature importance
- Kaplan–Meier survival analysis
- Cox proportional hazards model
- Customer segmentation and CLV analysis
- Retention campaign analysis
- Interactive ROI simulator

## Project Structure

```text
DS-01-Customer-Intelligence/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── lightgbm_churn_model.pkl
├── customer_churn_risk.csv
├── customer_uplift_scores.csv
├── shap_feature_importance.csv
├── clv_summary.csv
├── contract_retention_summary.csv
├── retention_roi_scenario.csv
├── customer_cluster_profile.csv
├── cox_model_results.csv
├── cox_model_summary.csv
├── kaplan_meier_survival.csv
└── retention_roi_sensitivity.png
```

## Setup Instructions

1. Clone this repository.
2. Install dependencies:

   pip install -r requirements.txt

3. Configure PostgreSQL credentials using Streamlit secrets.
4. Run the application:

   streamlit run app.py

## Project Status

Core churn analytics, survival analysis, customer segmentation, and ROI simulation have been implemented. Uplift modelling outputs require validation.

## Author

DS-01 Internship Project – Nexvia Technologies
